"""Shared Harness construction for free-form goals and LongBench adapters."""

from ..executor import VLA, CompletionDetector
from ..goal import GoalTask
from ..harness import Harness
from ..interfaces import RuntimeConfig
from ..memory import Memory
from ..planner import JsonPlanner, JsonVisualVerifier
from .instructions import InstructionRenderer
from .subtasks import default_subtasks


def build_harness(
    goal,
    *,
    visual_query,
    action_model,
    run_id="episode",
    seed="",
    detector_model=None,
    handoff="detector",
    history=None,
    config=None,
):
    registry, criteria = default_subtasks()
    policy = VLA(action_model.act, reset=action_model.reset, route=action_model.set_route)
    detector = (
        CompletionDetector(detector_model.predict, action_model.policy_context) if detector_model else None
    )
    return Harness(
        registry,
        policy,
        JsonPlanner(visual_query),
        JsonVisualVerifier(visual_query, criteria, public_goal=goal),
        InstructionRenderer(route_callback=policy.set_route),
        Memory(run_id, str(seed), history=history),
        completion_criteria=criteria,
        detector=detector,
        config=config or RuntimeConfig(handoff=handoff),
    )


def run(goal, env, initial_frame=None, *, steps=3600, minutes=20, **adapters):
    harness = build_harness(goal, **adapters)
    task = GoalTask("goal", goal, budget={"steps": steps, "minutes": minutes})
    return harness.run(task, env, initial_frame if initial_frame is not None else env.initial_frame())
