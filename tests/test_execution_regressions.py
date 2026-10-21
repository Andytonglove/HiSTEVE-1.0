"""Execution progress and landmark regression tests."""

from dataclasses import replace

import pytest
from test_harness import FakeEnvironment, runtime, task

from histeve import Command, Frame, RuntimeConfig, Subtasks, default_subtasks
from histeve.interfaces import VisualAssessment, inventory_deltas
from histeve.memory import Memory
from histeve.minecraft.instructions import InstructionRenderer
from histeve.planner import JsonVisualVerifier
from histeve.state import StateSnapshot


@pytest.mark.parametrize(
    "requested,product,count",
    [
        ("planks", "oak_planks", 4),
        ("sticks", "stick", 4),
        ("table", "crafting_table", 1),
    ],
)
def test_recipe_alias_uses_the_same_item_for_execution_and_progress(requested, product, count):
    registry, criteria = default_subtasks()
    card = replace(registry.get("craft_inventory_item"), default_budget_steps=1)
    harness, _, planner = runtime(config=RuntimeConfig(handoff="fixed"))
    harness.registry, harness.completion_criteria = Subtasks([card]), criteria
    harness.renderer = InstructionRenderer()
    planner.choose = lambda context: Command(card.skill_id, {"item": requested, "count": 1})

    def assess(frames, command, before):
        return VisualAssessment(
            StateSnapshot(inventory={product: count if command else 0}),
            (frames[-1].image_ref,),
            "success" if command else "interrupted",
            inventory_scope="storage_total",
        )

    harness.verifier.assess = assess
    result = harness.run(
        replace(task(), budget={"steps": 1, "minutes": 1}), FakeEnvironment(), Frame(0, "frame-0")
    )
    assert result.episodes[0].status == "success"
    assert result.episodes[0].parameters["item"] == product
    assert result.episodes[0].progress_events == (f"inventory_delta:{product}={count}",)


def test_smelt_can_resume_after_input_has_left_player_storage():
    registry, criteria = default_subtasks()
    card = replace(registry.get("smelt_item"), default_budget_steps=1)
    harness, _, planner = runtime(config=RuntimeConfig(handoff="fixed"))
    harness.registry, harness.completion_criteria = Subtasks([card]), criteria
    harness.renderer = InstructionRenderer()
    planner.choose = lambda context: Command(
        "smelt_item", {"input": "iron_ore", "output": "iron_ingot", "count": 1}
    )

    def assess(frames, command, before):
        refs = ("frame-0", frames[-1].image_ref) if command else ("frame-0",)
        return VisualAssessment(
            StateSnapshot(visible={"furnace"}),
            refs,
            "success" if command else "interrupted",
            inventory_scope="unknown",
            inventory_gains={"iron_ingot": 1} if command else {},
        )

    harness.verifier.assess = assess
    result = harness.run(
        replace(task(), budget={"steps": 1, "minutes": 1}), FakeEnvironment(), Frame(0, "frame-0")
    )
    assert result.steps == 1
    assert result.episodes[0].status == "success"


def test_hotbar_rearrangement_is_not_progress_but_witnessed_transfer_can_be():
    before = VisualAssessment(
        StateSnapshot(inventory={"oak_log": 1}), ("before",), inventory_scope="hotbar_lower_bound"
    )
    after = replace(before, state=StateSnapshot(inventory={"oak_log": 2}), evidence_refs=("before", "after"))
    assert inventory_deltas(before, after) == {}
    assert inventory_deltas(before, replace(after, inventory_gains={"oak_log": 1})) == {"oak_log": 1}
    # Use the inventory delta when both storage counts are available.
    assert (
        inventory_deltas(
            replace(before, inventory_scope="storage_total"),
            replace(before, inventory_scope="storage_total", inventory_gains={"oak_log": 9}),
        )
        == {}
    )


def test_full_inventory_counts_include_new_items():
    before = VisualAssessment(StateSnapshot(inventory={}), ("before",), inventory_scope="storage_total")
    after = VisualAssessment(
        StateSnapshot(inventory={"oak_log": 1}), ("after",), inventory_scope="storage_total"
    )
    assert inventory_deltas(before, after) == {"oak_log": 1}


def test_recorded_landmark_reaches_both_instruction_and_visual_comparison():
    memory = Memory("run", "seed")
    reference = Frame(0, "original-camp")
    memory.observe(
        VisualAssessment(
            StateSnapshot(),
            (reference.image_ref,),
            landmarks={"camp": "A table beside a red flower and a tall cliff"},
            observation_frames=(reference,),
        )
    )
    before = memory.observe(
        VisualAssessment(StateSnapshot(), ("far-away",), observation_frames=(Frame(40, "far-away"),))
    )
    registry, criteria = default_subtasks()
    subtask = registry.get("return_to_landmark")
    parameters = subtask.execution_parameters({"landmark_id": "camp"}, memory.world)
    assert "red flower" in InstructionRenderer()(subtask, parameters)
    assert subtask.execution_parameters(parameters, memory.world) == parameters
    captured = []

    def query(images, request):
        captured.append((images, request))
        return {
            "state": {},
            "inventory_scope": "unknown",
            "inventory_gains": {},
            "evidence_refs": [request["frames"][-1]["image_ref"]],
            "status": "interrupted",
            "reason": "Still looking",
            "recoverable": True,
            "landmarks": {},
        }

    JsonVisualVerifier(query, criteria).assess(
        (Frame(41, "current"),), Command(subtask.skill_id, parameters), before
    )
    images, request = captured[0]
    assert images == ["original-camp", "far-away", "current"]
    assert "red flower" in request["landmarks"]["camp"]["description"]
    assert request["landmarks"]["camp"]["evidence_refs"] == ["image_0"]
