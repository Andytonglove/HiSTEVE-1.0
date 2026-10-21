import itertools
import json
import unittest
from dataclasses import replace
from pathlib import Path

from histeve import default_subtasks
from histeve.executor import VLA, CompletionDetector, HandoffGate
from histeve.goal import GoalTask
from histeve.harness import Harness
from histeve.interfaces import (
    Command,
    DetectorPrediction,
    Frame,
    InvalidPlannerResponse,
    InvalidVisualResponse,
    NativeAction,
    RuntimeConfig,
    validate_assessment,
)
from histeve.interfaces import VisualAssessment as Assessment
from histeve.memory import (
    EpisodeRecord,
    Memory,
    MemoryIndex,
    ProcedureRecord,
    ProcedureStep,
)
from histeve.minecraft.instructions import InstructionRenderer
from histeve.planner import JsonPlanner, JsonVisualVerifier
from histeve.state import StateSnapshot
from histeve.subtasks import Subtask, Subtasks

ROOT = Path(__file__).resolve().parents[1]


def VisualAssessment(*args, **kwargs):
    kwargs.setdefault("inventory_scope", "storage_total")
    return Assessment(*args, **kwargs)


class ControllerTests(unittest.TestCase):
    def test_policy_context_and_detector_precede_action(self):
        events, windows = [], []

        def context(frame, instruction):
            events.append(("context", frame.step))
            return ("policy_prefix", frame.step, instruction)

        def query(history, current, instruction, prefix):
            self.assertEqual(prefix, ("policy_prefix", current.step, instruction))
            events.append(("detect", current.step))
            windows.append([f.step for f in history] + [current.step])
            return 0.9

        harness, policy, _ = runtime(FakeVerifier((0, 3), "success"))
        harness.registry = Subtasks([replace(skill(), default_budget_steps=32)])
        harness.detector = CompletionDetector(query, context)

        def act(frame, instruction):
            events.append(("act", frame.step))
            return NativeAction()

        policy.act = act
        result = harness.run(
            replace(task(), budget={"steps": 32, "minutes": 1}), FakeEnvironment(), Frame(0, "frame-0")
        )
        self.assertEqual(windows, [[0, 4, 8, 12, 16], [1, 5, 9, 13, 17]])
        self.assertLess(events.index(("context", 16)), events.index(("detect", 16)))
        self.assertLess(events.index(("detect", 16)), events.index(("act", 16)))
        self.assertNotIn(("act", 17), events)
        self.assertTrue(result.visual_goal_met)
        self.assertEqual(result.detector_handoffs, 1)

    def test_fixed_instruction_and_native_actions(self):
        calls = []
        policy = VLA(lambda f, i: NativeAction(("attack",)), reset=lambda: calls.append("reset"))
        policy.begin("Mine stone.")
        self.assertEqual(policy.act(Frame(0, "frame-0"), "Mine stone.").buttons, ("attack",))
        self.assertEqual(calls, ["reset"])
        with self.assertRaises(ValueError):
            policy.act(Frame(1, "frame-1"), "Craft a stick.")
        policy.infer = lambda f, i: {"script": "craft()"}
        with self.assertRaises(TypeError):
            policy.act(Frame(1, "frame-1"), "Mine stone.")

    def test_detector_does_not_reuse_cached_frame(self):
        calls = []
        detector = CompletionDetector(lambda *args: calls.append(args) or 0.9, lambda *args: None, stride=1)
        detector.reset("Mine stone.")
        for i in range(4):
            self.assertIsNone(detector.predict(Frame(i, str(i)), "Mine stone."))
        first = detector.predict(Frame(4, "4"), "Mine stone.")
        self.assertEqual(first.query_id, 1)
        self.assertIsNone(detector.predict(Frame(4, "4"), "Mine stone."))
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            detector.predict(Frame(6, "6"), "Mine stone.")

    def test_default_registry_and_gui_route(self):
        registry, criteria = default_subtasks()
        self.assertEqual(set(registry.ids()), set(criteria))
        self.assertEqual(len(registry.ids()), 15)
        calls = []
        renderer = InstructionRenderer(route_callback=lambda **kw: calls.append(kw))
        text = renderer(registry.get("craft_inventory_item"), {"item": "oak_planks", "count": 1})
        self.assertIn("oak log", text)
        self.assertEqual(calls, [{"is_gui": True, "needs_table": False}])


def skill():
    return Subtask(
        "acquire",
        "interaction",
        "Collect an item",
        {
            "required": ["item", "count"],
            "properties": {"item": {"type": "string"}, "count": {"type": "integer"}},
        },
        default_budget_steps=4,
    )


class FakeEnvironment:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return Frame(len(self.actions), f"frame-{len(self.actions)}")


class FakePolicy:
    def __init__(self):
        self.instructions = []

    def begin(self, instruction):
        pass

    def act(self, frame, instruction):
        self.instructions.append(instruction)
        return NativeAction(("forward",))


class FakePlanner:
    def __init__(self):
        self.contexts = []

    def choose(self, context):
        self.contexts.append(context)
        return Command("acquire", {"item": "oak_log", "count": 3})


class FakeVerifier:
    def __init__(self, counts=None, status="interrupted", recoverable=False):
        self.counts = iter(counts) if counts is not None else itertools.repeat(0)
        self.status = status
        self.recoverable = recoverable

    def assess(self, frames, command, before):
        return VisualAssessment(
            StateSnapshot(inventory={"oak_log": next(self.counts)}),
            (frames[-1].image_ref,),
            self.status,
            recoverable=self.recoverable,
        )


class FakeDetector:
    def reset(self, instruction):
        pass

    def predict(self, frame, instruction):
        return DetectorPrediction(frame.step, 0.9)


def task():
    return GoalTask(
        "test",
        "Obtain three oak logs.",
        {},
        ({"inventory_at_least": {"oak_log": 3}},),
        {},
        {"steps": 6, "minutes": 1},
    )


def runtime(verifier=None, config=RuntimeConfig()):
    (policy, planner) = (FakePolicy(), FakePlanner())
    harness = Harness(
        Subtasks([skill()]),
        policy,
        planner,
        verifier or FakeVerifier(),
        lambda card, args: f"Collect {args['count']} {args['item']}.",
        Memory("test-run", "123"),
        completion_criteria={"acquire": "Visible inventory gain"},
        detector=FakeDetector(),
        config=config,
    )
    return (harness, policy, planner)


class RuntimeTests(unittest.TestCase):
    def test_invalid_initial_visual_response_is_a_model_termination(self):
        (harness, policy, planner) = runtime(config=RuntimeConfig(handoff="fixed"))

        def invalid(*args):
            raise InvalidVisualResponse("invalid inventory counts")

        harness.verifier.assess = invalid
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.termination, "verifier_invalid_response")
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.policy_error, "invalid inventory counts")
        self.assertFalse(result.visual_goal_met)
        self.assertEqual(policy.instructions, [])
        self.assertEqual(planner.contexts, [])
        self.assertEqual(harness.memory.current, [])

    def test_invalid_post_action_evidence_keeps_consumed_steps_without_success_memory(self):
        (harness, policy, _) = runtime(config=RuntimeConfig(handoff="fixed"))

        def assess(frames, command, before):
            if command:
                return VisualAssessment(
                    StateSnapshot(inventory={"oak_log": 999}), ("invented-frame",), "success"
                )
            return VisualAssessment(StateSnapshot(inventory={"oak_log": 0}), (frames[-1].image_ref,))

        harness.verifier.assess = assess
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.termination, "verifier_invalid_response")
        self.assertEqual(result.steps, 4)
        self.assertEqual(len(policy.instructions), 4)
        self.assertFalse(result.visual_goal_met)
        self.assertEqual(result.assessment.state.inventory, {"oak_log": 0})
        self.assertEqual(harness.memory.current, [])
        self.assertEqual(harness.memory.procedures, [])

    def test_verifier_transport_failure_is_not_reclassified_as_model_output(self):
        (harness, _, _) = runtime(config=RuntimeConfig(handoff="fixed"))

        def timeout(*args):
            raise TimeoutError("endpoint unavailable")

        harness.verifier.assess = timeout
        with self.assertRaises(TimeoutError):
            harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))

    def test_inventory_binding_filters_choices_without_changing_registry(self):
        registry = Subtasks.from_json(ROOT / "src/histeve/minecraft/subtasks.json")
        card = registry.get("place_block")
        self.assertIsNone(card.with_visual_inventory(StateSnapshot()))
        self.assertIsNone(card.with_visual_inventory(StateSnapshot(inventory={"furnace": 0})))
        bound = card.with_visual_inventory(StateSnapshot(inventory={"furnace": 1, "dirt": 8}))
        self.assertEqual(bound.parameter_schema["properties"]["item"]["enum"], ["furnace"])
        self.assertIn("crafting_table", card.parameter_schema["properties"]["item"]["enum"])
        with self.assertRaises(ValueError):
            bound.validate_parameters({"item": "crafting_table"})
        smelt = registry.get("smelt_item").with_visual_inventory(StateSnapshot(inventory={"iron_ore": 2}))
        self.assertIn("iron_ore", smelt.parameter_schema["properties"]["input"]["enum"])
        self.assertIsNotNone(
            registry.get("smelt_item").with_visual_inventory(StateSnapshot(inventory={"iron_ingot": 1}))
        )
        self.assertIsNotNone(
            registry.get("smelt_item").with_visual_inventory(StateSnapshot(inventory={"raw_iron": 1}))
        )

    def test_runtime_checks_bound_inventory_even_without_schema_enforcing_server(self):
        (harness, policy, planner) = runtime(config=RuntimeConfig(handoff="fixed"))
        placement = Subtask(
            "place_block",
            "placement",
            "Place a block",
            {"required": ["item"], "properties": {"item": {"type": "string", "enum": ["furnace", "chest"]}}},
            default_budget_steps=4,
            inventory_parameters=("item",),
        )
        harness.registry = Subtasks([skill(), placement])
        harness.completion_criteria["place_block"] = "New visible block"
        harness.verifier.assess = lambda frames, command, before: VisualAssessment(
            StateSnapshot(inventory={"furnace": 1}), (frames[-1].image_ref,)
        )
        contexts = []

        def choose(context):
            contexts.append(context)
            return Command("place_block", {"item": "chest"})

        planner.choose = choose
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.termination, "planner_invalid_command")
        self.assertEqual(policy.instructions, [])
        offered = next((card for card in contexts[0]["schemas"] if card["skill_id"] == "place_block"))
        self.assertEqual(offered["parameter_schema"]["properties"]["item"]["enum"], ["furnace"])
        self.assertEqual(offered["inventory_parameters"], ["item"])

    def test_detector_is_handoff_not_success(self):
        (harness, policy, planner) = runtime()
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.detector_handoffs, 5)
        self.assertFalse(result.visual_goal_met)
        self.assertTrue(all((row.status == "interrupted" for row in result.episodes)))
        self.assertEqual(len(policy.instructions), 6)
        self.assertEqual(set(policy.instructions), {"Collect 1 oak_log."})
        self.assertNotIn("goal_predicates", planner.contexts[0])
        self.assertNotIn("diagnostic", json.dumps(planner.contexts[0], default=list))
        schema = planner.contexts[0]["schemas"][0]
        self.assertNotIn("executor_bindings", schema)
        self.assertNotIn("recovery_skill_ids", schema)
        self.assertEqual(schema["budget_steps"], 4)
        self.assertNotIn("terminal", planner.contexts[0]["frames"][0])

    def test_consecutive_gate_strict_threshold_and_missing(self):
        gate = HandoffGate(0.7)
        self.assertFalse(gate.update(0.7))
        self.assertFalse(gate.update(0.9))
        self.assertFalse(gate.update(None))
        self.assertFalse(gate.update(0.9))
        self.assertTrue(gate.update(0.9))
        with self.assertRaises(ValueError):
            gate.update(float("nan"))

    def test_fixed_horizon_same_policy_no_detector(self):
        (harness, policy, _) = runtime(FakeVerifier(counts=(0, 0, 0)), RuntimeConfig(handoff="fixed"))
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.detector_handoffs, 0)
        self.assertEqual([row.end_step - row.start_step for row in result.episodes], [4, 2])
        self.assertEqual(len(policy.instructions), 6)

    def test_invalid_verifier_response_stops_execution_without_committing_progress(self):
        harness, _, _ = runtime(config=RuntimeConfig(handoff="fixed"))
        env = FakeEnvironment()
        calls = []

        def query(images, request):
            calls.append(len(env.actions))
            refs = ["image_99"] if len(calls) == 2 else [request["frames"][-1]["image_ref"]]
            return {
                "inventory_scope": "storage_total",
                "inventory_gains": {},
                "state": {"inventory": {"oak_log": 0}},
                "evidence_refs": refs,
                "status": "failure",
                "reason": "no inventory gain",
                "recoverable": False,
                "landmarks": {},
            }

        harness.verifier = JsonVisualVerifier(query, {"acquire": "gain item"})
        result = harness.run(task(), env, Frame(0, "frame-0"))
        self.assertEqual(calls, [0, 4])
        self.assertEqual(result.termination, "verifier_invalid_response")
        self.assertEqual(result.steps, 4)
        self.assertEqual(harness.memory.current, [])

    def test_fixed_contract_preserves_verification_memory_and_local_recovery(self):
        config = RuntimeConfig(handoff="fixed")
        (harness, policy, planner) = runtime(FakeVerifier(status="failure", recoverable=True), config)
        longer = replace(task(), budget={"steps": 12, "minutes": 1})
        result = harness.run(longer, FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.local_retries, config.max_local_retries)
        self.assertEqual(result.planner_calls, 2)
        self.assertEqual(len(harness.memory.current), 3)
        self.assertTrue(planner.contexts[1]["retrieved"])
        self.assertTrue(all((row.status == "failure" for row in result.episodes)))
        self.assertEqual(result.detector_handoffs, 0)
        self.assertEqual(len(policy.instructions), 12)

    def test_cached_detector_sample_not_counted_twice(self):
        gate = HandoffGate(0.7)
        self.assertFalse(gate.update(0.9, 4))
        self.assertFalse(gate.update(0.9, 4))
        self.assertTrue(gate.update(0.9, 8))
        with self.assertRaises(ValueError):
            gate.update(0.9, 3)

    def test_sampled_detector_skips_are_not_negative_predictions(self):
        (harness, _, _) = runtime()
        harness.detector.predict = lambda frame, instruction: (
            DetectorPrediction(frame.step, 0.9) if frame.step % 2 == 0 else None
        )
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.detector_handoffs, 2)
        self.assertEqual([row.cost_steps for row in result.episodes], [2, 2, 2])

    def test_delta_not_old_inventory(self):
        (harness, _, planner) = runtime(FakeVerifier((1, 2, 2, 3), "success"))
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertTrue(result.visual_goal_met)
        self.assertEqual(result.episodes[1].progress_events, ())
        self.assertEqual(result.episodes[1].status, "interrupted")
        self.assertEqual(result.episodes[1].cost_steps, 1)
        self.assertEqual(planner.contexts[2]["working"]["requirements"]["oak_log"]["remaining"], 1)
        self.assertTrue(harness.memory.procedures)

    def test_no_scripts_and_no_inapplicable_skill(self):
        with self.assertRaises(ValueError):
            NativeAction(("teleport",))
        with self.assertRaises(ValueError):
            NativeAction(camera=(float("inf"), 0))
        (harness, _, planner) = runtime()
        planner.choose = lambda context: Command("unknown", {})
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.termination, "planner_invalid_command")
        self.assertFalse(result.visual_goal_met)
        self.assertEqual(result.steps, 0)

    def test_invalid_model_decisions_are_terminal_attempts_not_infra_exceptions(self):
        for command in (
            Command("acquire", {"item": "oak_log", "count": 0}),
            Command("acquire", {"item": "oak_log", "count": "1"}),
        ):
            (harness, _, planner) = runtime()
            planner.choose = lambda context: command
            result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
            self.assertEqual(result.termination, "planner_invalid_command")
            self.assertEqual(result.planner_calls, 1)
            self.assertTrue(result.policy_error)
        (harness, _, planner) = runtime()

        def bad(context):
            raise InvalidPlannerResponse("invalid JSON")

        planner.choose = bad
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.termination, "planner_invalid_command")

        def offline(context):
            raise ConnectionError("offline")

        (harness, _, planner) = runtime()
        planner.choose = offline
        with self.assertRaises(ConnectionError):
            harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))

    def test_missing_inventory_count_is_not_zero(self):
        (harness, _, _) = runtime(config=RuntimeConfig(handoff="fixed"))

        def assess(frames, command, before):
            state = StateSnapshot(inventory={"oak_log": 1} if command else {})
            return VisualAssessment(
                state,
                (frames[-1].image_ref,),
                "success",
                inventory_scope="storage_total" if command else "unknown",
            )

        harness.verifier.assess = assess
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.episodes[0].status, "interrupted")
        self.assertEqual(result.episodes[0].progress_events, ())
        self.assertEqual(harness.memory.procedures, [])

    def test_placement_claim_without_state_change_cannot_become_procedure(self):
        for initial, final, expected in (
            ({}, {}, "interrupted"),
            ({}, {"furnace": 1}, "interrupted"),
            ({"furnace": 1}, {"furnace": 1}, "interrupted"),
            ({"furnace": 0}, {"furnace": 1}, "success"),
        ):
            with self.subTest(initial=initial, final=final):
                (harness, _, planner) = runtime(config=RuntimeConfig(handoff="fixed"))
                harness.registry = Subtasks(
                    [
                        Subtask(
                            "place_block",
                            "placement",
                            "Place a block",
                            {"required": ["item"], "properties": {"item": {"type": "string"}}},
                            default_budget_steps=4,
                        )
                    ]
                )
                harness.completion_criteria = {"place_block": "New visible block"}
                harness.renderer = lambda *args: "Place a furnace."
                planner.choose = lambda context: Command("place_block", {"item": "furnace"})
                harness.verifier.assess = lambda frames, command, before: VisualAssessment(
                    StateSnapshot(placed=final if command else initial),
                    (frames[-1].image_ref,),
                    "success",
                    "The furnace was placed.",
                )
                result = harness.run(
                    replace(task(), budget={"steps": 4, "minutes": 1}), FakeEnvironment(), Frame(0, "frame-0")
                )
                self.assertEqual(result.episodes[0].status, expected)
                self.assertEqual(bool(harness.memory.procedures), expected == "success")

    def test_stale_or_fabricated_evidence_rejected(self):
        (harness, _, _) = runtime()
        harness.verifier.assess = lambda *args: VisualAssessment(
            StateSnapshot(), ("not-observed",), "success"
        )
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.termination, "verifier_invalid_response")
        self.assertEqual(result.steps, 0)
        self.assertFalse(result.visual_goal_met)
        self.assertIn("cite current evidence", result.policy_error)
        self.assertEqual(harness.memory.current, [])

    def test_before_after_citations_require_current_evidence_not_only_history(self):
        (before, after) = ([Frame(0, "before")], [Frame(4, "after")])
        assessment = VisualAssessment(StateSnapshot(), ("before", "after"), "interrupted")
        validated = validate_assessment(assessment, after, before)
        self.assertEqual(validated.observation_frames, tuple(after))
        for refs in (("before",), ("stale", "after"), ()):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                validate_assessment(replace(assessment, evidence_refs=refs), after, before)

    def test_retry_bounded_and_fresh_runtime_required(self):
        (harness, _, _) = runtime(FakeVerifier(recoverable=True))
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.local_retries, 3)
        self.assertEqual(result.planner_calls, 3)
        with self.assertRaises(ValueError):
            harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))

    def test_deadline_checks_after_policy_call(self):
        (harness, policy, _) = runtime()
        now = [0.0]
        harness.clock = lambda: now[0]

        def slow_act(frame, instruction):
            now[0] += 61
            return NativeAction()

        policy.act = slow_act
        env = FakeEnvironment()
        result = harness.run(task(), env, Frame(0, "frame-0"))
        self.assertEqual(result.termination, "time_budget")
        self.assertEqual(env.actions, [])

    def test_no_policy_decode_after_detector_exhausts_deadline(self):
        (harness, policy, _) = runtime()
        now = [0.0]
        harness.clock = lambda: now[0]

        def slow_query(frame, instruction):
            now[0] = 61.0
            return None

        harness.detector.predict = slow_query
        env = FakeEnvironment()
        result = harness.run(task(), env, Frame(0, "frame-0"))
        self.assertEqual(result.termination, "time_budget")
        self.assertEqual(policy.instructions, [])
        self.assertEqual(env.actions, [])

    def test_no_zero_cost_environment_step(self):
        (harness, _, _) = runtime()
        env = FakeEnvironment()
        env.step = lambda action: Frame(0, "frame-0")
        with self.assertRaises(ValueError):
            harness.run(task(), env, Frame(0, "frame-0"))


class MemoryTests(unittest.TestCase):
    def test_visual_state_changes_retrieval_with_same_goal(self):
        rows = [
            EpisodeRecord(
                name,
                "task",
                "old-seed",
                "acquire",
                {},
                "success",
                0,
                2,
                evidence_refs=("image",),
                context_tags=(name,),
                protocol_split="train",
            )
            for name in ("oak_log", "iron_ore")
        ]
        memory = Memory("run", "new-seed", MemoryIndex(rows))
        for item in ("oak_log", "iron_ore"):
            found = memory.retrieve("Collect resources", state=StateSnapshot(visible={item}))
            self.assertEqual(found[0]["episode_id"], item)
        self.assertEqual(memory.history.episodes["oak_log"].context_tags, ("oak_log",))

    def test_failure_retrieval_keeps_relevant_older_current_episode(self):
        memory = Memory("run", "seed")
        for n in range(6):
            memory.commit(
                EpisodeRecord(
                    f"run:{n}",
                    "task",
                    "seed",
                    "acquire",
                    {},
                    "failure",
                    n,
                    n + 1,
                    evidence_refs=("image",),
                    failure_signature="tree_obscured" if n == 0 else "no_progress",
                ),
                "Collect resources",
            )
        found = memory.retrieve("Collect resources", failure_signature="tree_obscured", task_id="task")
        self.assertEqual([row["episode_id"] for row in found], ["run:0", "run:5", "run:4", "run:3"])
        self.assertEqual(memory.procedures, [])

    def test_context_filters_same_seed_and_unverified_history(self):
        relevant = EpisodeRecord(
            "same-seed",
            "task",
            "seed",
            "acquire",
            {},
            "success",
            0,
            2,
            evidence_refs=("image",),
            context_tags=("oak_log",),
            protocol_split="train",
        )
        rows = [
            relevant,
            replace(relevant, episode_id="no-evidence", seed="other", evidence_refs=()),
            replace(relevant, episode_id="past-deployment", seed="other", protocol_split="deployment"),
        ]
        memory = Memory("run", "seed", MemoryIndex(rows))
        self.assertEqual(
            memory.retrieve("oak_log", state=StateSnapshot(visible={"oak_log"}), task_id="task"), []
        )

    def test_state_tags_are_deterministic_and_preserve_false_vs_unknown(self):
        state = StateSnapshot(
            inventory={"oak_log": 1, "stick": 0},
            visible={"tree", "grass"},
            flags={"gui_open": False},
            placed={"crafting_table": 1},
        )
        tags = Memory.state_tags(state)
        self.assertEqual(tags, tuple(sorted(tags)))
        self.assertIn("oak_log", tags)
        self.assertIn("crafting_table", tags)
        self.assertNotIn("stick", tags)
        self.assertIn("flag:gui_open=false", tags)
        self.assertNotIn("flag:gui_open=false", Memory.state_tags(StateSnapshot()))

    def test_runtime_passes_visual_working_state_to_retrieval(self):
        (harness, _, _) = runtime(FakeVerifier(status="failure"), RuntimeConfig(handoff="fixed"))
        calls = []
        original = harness.memory.retrieve

        def capture(goal, **kwargs):
            calls.append((goal, kwargs))
            return original(goal, **kwargs)

        harness.memory.retrieve = capture
        harness.verifier.assess = lambda frames, command, before: VisualAssessment(
            StateSnapshot(visible={"oak_log"}), (frames[-1].image_ref,), "failure", "tree_obscured"
        )
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1]["state"].visible, {"oak_log"})
        self.assertEqual(calls[1][1]["failure_signature"], "tree_obscured")
        self.assertEqual(calls[1][1]["task_id"], task().task_id)
        self.assertEqual(result.episodes[0].context_tags, ("oak_log",))
        self.assertFalse(result.visual_goal_met)

    def test_filter_procedure_source_seed_and_unverified(self):
        source = EpisodeRecord(
            "old",
            "task",
            "123",
            "acquire",
            {},
            "success",
            0,
            2,
            evidence_refs=("image",),
            protocol_split="train",
        )
        procedure = ProcedureRecord(
            "p", "Obtain oak logs", (ProcedureStep("s", "acquire", {}),), ("old",), protocol_split="train"
        )
        memory = Memory("run", "123", MemoryIndex([source], [procedure]))
        self.assertEqual(memory.retrieve("Obtain oak logs"), [])
        memory = Memory("run", "456", MemoryIndex([replace(source, evidence_refs=())], [procedure]))
        self.assertEqual(memory.retrieve("Obtain oak logs"), [])

    def test_verified_new_seed_retrieval_and_nested_coordinates(self):
        source = EpisodeRecord(
            "old",
            "task",
            "123",
            "acquire",
            {"item": "oak_log"},
            "success",
            0,
            2,
            evidence_refs=("image",),
            protocol_split="train",
        )
        memory = Memory("run", "456", MemoryIndex([source]))
        self.assertTrue(memory.retrieve("Obtain oak_log"))
        bad = ProcedureRecord(
            "p",
            "logs",
            (ProcedureStep("s", "acquire", {"extra": {"yaw": 12}}),),
            ("old",),
            protocol_split="train",
        )
        memory.history.add_procedure(bad)
        with self.assertRaises(ValueError):
            memory.retrieve("logs")

    def test_free_text_coordinates_are_rejected(self):
        source = EpisodeRecord(
            "old",
            "task",
            "123",
            "acquire",
            {},
            "success",
            0,
            2,
            evidence_refs=("image",),
            protocol_split="train",
        )
        bad = ProcedureRecord(
            "p",
            "logs",
            (ProcedureStep("s", "acquire", {"note": "x=12, z=-4"}),),
            ("old",),
            protocol_split="train",
        )
        memory = Memory("run", "456", MemoryIndex([source], [bad]))
        with self.assertRaises(ValueError):
            memory.retrieve("logs")
        for text in (
            "position (12, 64, -4)",
            "return to (12, 64, -4)",
            "camera facing north",
            "look up 90 degrees",
            "camera down 30 deg",
            "look left 45 degrees",
        ):
            bad = ProcedureRecord(
                "p",
                "logs",
                (ProcedureStep("s", "acquire", {"note": text}),),
                ("old",),
                protocol_split="train",
            )
            with self.assertRaises(ValueError):
                Memory("run", "456", MemoryIndex([source], [bad])).retrieve("logs")

    def test_episode_and_world_memory_cannot_leak_literal_pose(self):
        source = EpisodeRecord(
            "old",
            "task",
            "123",
            "acquire",
            {"note": "coordinates: 1, 2, 3"},
            "success",
            0,
            2,
            evidence_refs=("image",),
            protocol_split="train",
        )
        with self.assertRaises(ValueError):
            Memory("run", "456", MemoryIndex([source])).retrieve("logs")
        (harness, _, _) = runtime()
        harness.verifier.assess = lambda frames, command, before: VisualAssessment(
            StateSnapshot(), (frames[-1].image_ref,), "interrupted", landmarks={"camp": "position (1, 2, 3)"}
        )
        result = harness.run(task(), FakeEnvironment(), Frame(0, "frame-0"))
        self.assertEqual(result.termination, "verifier_invalid_response")
        self.assertEqual(result.steps, 0)
        self.assertIn("literal pose", result.policy_error)
        self.assertEqual(harness.memory.world, {})
        self.assertEqual(harness.memory.current, [])


class VisualAdapterTests(unittest.TestCase):
    def test_relative_view_descriptions_are_accepted(self):
        for description in (
            "large dirt wall visible when looking up from the tunnel",
            "grass blocks visible when looking down from the ledge",
        ):
            calls = []

            def query(images, context):
                calls.append(context)
                return {
                    "inventory_scope": "storage_total",
                    "inventory_gains": {},
                    "state": {"inventory": {}},
                    "evidence_refs": ["image_0"],
                    "status": "interrupted",
                    "reason": "No new evidence.",
                    "recoverable": True,
                    "landmarks": {"scene": description},
                }

            verifier = JsonVisualVerifier(query, {})
            value = verifier.assess((Frame(0, "current"),), None, None)
            self.assertEqual(value.landmarks["scene"], description)
            self.assertEqual(len(calls), 1)

    def test_nested_inventory_response_and_bad_json_are_model_errors(self):
        payload = {
            "inventory_scope": "storage_total",
            "inventory_gains": {},
            "state": {"inventory": {"crafting": {"output": "diamond_pickaxe"}}},
            "evidence_refs": ["image_0"],
            "status": "success",
            "reason": "claimed item",
            "recoverable": False,
            "landmarks": {},
        }
        verifier = JsonVisualVerifier(lambda *args: payload, {})
        with self.assertRaises(InvalidVisualResponse):
            verifier.assess([Frame(0, "frame-0")], None, None)

        def invalid(*args):
            return json.loads("invalid")

        with self.assertRaises(InvalidVisualResponse):
            JsonVisualVerifier(invalid, {}).assess([Frame(0, "frame-0")], None, None)

    def test_verifier_only_receives_images_and_visual_state(self):
        captured = []

        def query(images, context):
            captured.append((images, context))
            return {
                "inventory_scope": "storage_total",
                "inventory_gains": {},
                "state": {"inventory": {}},
                "evidence_refs": ["image_0"],
                "status": "interrupted",
                "reason": "not visible",
                "recoverable": False,
                "landmarks": {},
            }

        verifier = JsonVisualVerifier(query, {})
        assessment = verifier.assess([Frame(0, "frame-0")], None, None)
        self.assertEqual(assessment.status, "interrupted")
        self.assertEqual(captured[0][0], ["frame-0"])
        self.assertNotIn("diagnostic", json.dumps(captured[0][1]))
        self.assertNotIn("terminal", captured[0][1]["frames"][0])

    def test_reject_wrong_response_shapes(self):
        planner = JsonPlanner(lambda *args: {"skill_id": "acquire", "parameters": {}, "script": "attack"})
        with self.assertRaises(ValueError):
            planner.choose({"frames": []})
        verifier = JsonVisualVerifier(
            lambda *args: {
                "inventory_scope": "storage_total",
                "inventory_gains": {},
                "state": {"inventory": {"oak_log": "unknown"}},
                "evidence_refs": [],
                "status": "success",
                "reason": "",
                "recoverable": False,
                "landmarks": {},
            },
            {},
        )
        with self.assertRaises(ValueError):
            verifier.assess([Frame(0, "frame-0")], None, None)
