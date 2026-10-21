import io
import json
import tempfile
import unittest
from pathlib import Path

from test_harness import runtime

from experiments.longbench import (
    FAMILIES,
    Observation,
    bind_task,
    evaluation_plan,
    load_catalog,
    run_suite,
    summarize,
)
from experiments.longbench.evaluator import LongBenchEvidenceMonitor
from experiments.longbench.runner import ScoredEnvironment
from histeve.interfaces import Frame, RuntimeConfig, VisualGoalAssessment
from histeve.interfaces import VisualAssessment as Assessment
from histeve.state import StateSnapshot


def VisualAssessment(*args, **kwargs):
    kwargs.setdefault("inventory_scope", "storage_total")
    return Assessment(*args, **kwargs)


class Environment:
    def __init__(self):
        self.index = 0
        self.closed = False

    def initial_frame(self):
        return Frame(0, "frame-0")

    def step(self, action):
        self.index += 1
        return Frame(self.index, f"frame-{self.index}", terminal=True)

    def evaluation_observation(self):
        items = {"crafting_table": 1, "wooden_pickaxe": 1} if self.index else {}
        return Observation(self.index, f"frame-{self.index}", structured={"inventory": items})

    def close(self):
        self.closed = True


class Verifier:
    def assess(self, frames, command, before):
        ref = (frames[-1].image_ref,)
        return VisualAssessment(
            StateSnapshot(), ref, goal=VisualGoalAssessment("incomplete", "keep working", ref)
        )


class LongBenchTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()

    def test_catalog_task_counts(self):
        self.assertEqual([len(t) for t in FAMILIES.values()], [4, 4, 3, 3])
        self.assertEqual(sum(len(g.nodes) for g in self.catalog.diagnostics.values()), 49)
        plan = evaluation_plan(self.catalog, [1, 2, 3])
        self.assertEqual(len(plan["episodes"]), 42)
        self.assertTrue(all(set(row["model_input"]) == {"instruction"} for row in plan["episodes"]))

    def test_return_site_deterministic(self):
        task = self.catalog.task("resource_revisit_v1")
        self.assertEqual(bind_task(task, 2), bind_task(task, 2))
        self.assertIn(bind_task(task, 2).initial_state["requested_return"], {"tree_site", "stone_site"})

    def rows(self):
        return [
            {
                "task_id": task,
                "seed": seed,
                "success": family == "resource_crafting",
                                "reached_nodes": [],
                "model_id": "test-checkpoint",
                "planner_model": "test",
                                "benchmark": "longbench-v1",
            }
            for seed in [1, 2, 3]
            for family, tasks in FAMILIES.items()
            for task in tasks
        ]

    def test_macro_and_sample_sd(self):
        score = summarize(self.rows(), self.catalog, [1, 2, 3])
        self.assertEqual(score["goal_sr_family_macro"], 0.25)
        self.assertAlmostEqual(score["goal_sr_episode_micro"], 4 / 14)
        self.assertEqual(score["sd_sample"], 0)
        self.assertIsNone(summarize(self.rows()[:14], self.catalog, [1])["sd_sample"])

    def test_reject_missing_duplicate_mixed_receipts(self):
        rows = self.rows()
        for bad in (
            rows[:-1],
            rows + [rows[0]],
            [dict(rows[0], planner_model="changed")] + rows[1:],
            [dict(rows[0], harness_config={"handoff": "fixed"})] + rows[1:],
        ):
            with self.assertRaises(ValueError):
                summarize(bad, self.catalog, [1, 2, 3])

    def test_full_suite_uses_same_harness_and_private_terminal_score(self):
        environments = []

        def make_env(task, seed):
            env = Environment()
            environments.append(env)
            return env

        def make_harness(goal, seed):
            harness, _, _ = runtime(Verifier(), RuntimeConfig(handoff="fixed"))
            original = harness.run

            def checked(task, env, initial):
                self.assertEqual(task.goal_predicates, ())
                self.assertEqual(task.initial_state, {})
                self.assertEqual(task.constraints, {})
                return original(task, env, initial)

            harness.run = checked
            return harness

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            score = run_suite(
                make_env, make_harness, {"model_id": "test-checkpoint", "planner_model": "test"}, [1], root
            )
            self.assertEqual(score["episodes"], 14)
            rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["success"] for row in rows), 1)
            self.assertTrue(all(env.closed for env in environments))
            self.assertTrue(all(row["action_steps"] == 1 for row in rows))

    def test_flags_cannot_be_injected(self):
        env = Environment()
        env.evaluation_observation = lambda: Observation(
            0,
            "frame-0",
            structured={
                "inventory": {},
                "agentic_flags": {"charcoal_produced": True},
                "state_completion": {"confirmed": True},
            },
        )
        task = self.catalog.task("light_up_v1")
        scored = ScoredEnvironment(env, task, self.catalog.diagnostic(task.task_id), io.StringIO())
        self.assertFalse(scored.evaluation.goal_met)
        self.assertNotIn("charcoal_produced", scored.adapter._flags)

    def test_private_frame_must_match(self):
        env = Environment()
        env.evaluation_observation = lambda: Observation(1, "other", structured={"inventory": {}})
        task = self.catalog.task("wooden_age_v1")
        with self.assertRaises(ValueError):
            ScoredEnvironment(env, task, self.catalog.diagnostic(task.task_id), io.StringIO())

    def test_inventory_evidence_rejects_invalid_counts(self):
        task = self.catalog.task("wooden_age_v1")
        for value in (-1, True, 1.2, "1"):
            env = Environment()
            env.evaluation_observation = lambda: Observation(
                0, "frame-0", structured={"inventory": {"stone": value}}
            )
            with self.assertRaises(ValueError):
                ScoredEnvironment(env, task, self.catalog.diagnostic(task.task_id), io.StringIO())

    def test_visual_completion_does_not_count_as_native_success(self):
        def factory(goal, seed):
            harness, _, _ = runtime(Verifier(), RuntimeConfig(handoff="fixed"))

            def assess(frames, command, before):
                ref = (frames[-1].image_ref,)
                return VisualAssessment(
                    StateSnapshot(), ref, goal=VisualGoalAssessment("complete", "claimed", ref)
                )

            harness.verifier.assess = assess
            return harness

        with tempfile.TemporaryDirectory() as temporary:
            score = run_suite(
                lambda task, seed: Environment(),
                factory,
                {"model_id": "test-checkpoint", "planner_model": "test"},
                [1],
                Path(temporary) / "run",
            )
            self.assertEqual(score["goal_sr_family_macro"], 0)

    def test_adapter_failure_closes_environment(self):
        env = Environment()

        def fail(goal, seed):
            raise RuntimeError("model service unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            with self.assertRaises(RuntimeError):
                run_suite(
                    lambda task, seed: env,
                    fail,
                    {"model_id": "test-checkpoint", "planner_model": "test"},
                    [1],
                    root,
                )
            self.assertTrue(env.closed)
            self.assertFalse((root / "summary.json").exists())

    def test_invalid_seed_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            for seeds in ([True], [-1], [2**32], [1, 1], []):
                with self.assertRaises(ValueError):
                    run_suite(None, None, {}, seeds, root)
                self.assertFalse(root.exists())

    def test_landmark_requires_record_departure_and_return(self):
        monitor = LongBenchEvidenceMonitor(
            task_id="resource_revisit_v1", requested_return_landmark="tree_site"
        )
        monitor.begin_skill("record_landmark", {"landmark_id": "tree_site"})

        def obs(step, x, visible):
            return Observation(
                step,
                f"frame-{step}",
                structured={"inventory": {}, "pose": {"x": x, "y": 64, "z": 0}, "visible_objects": visible},
            )

        monitor.update(obs(0, 0, ["tree"]))
        self.assertFalse(monitor.flags.get("returned_to_requested_resource", False))
        monitor.end_skill()
        monitor.update(obs(1, 20, []))
        monitor.update(obs(2, 0, ["tree"]))
        self.assertTrue(monitor.flags["returned_to_requested_resource"])


if __name__ == "__main__":
    unittest.main()
