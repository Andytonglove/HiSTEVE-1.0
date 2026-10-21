"""Run LongBench episodes and collect scores."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from histeve.harness import Harness
from histeve.interfaces import Frame, NativeAction
from .benchmark import BENCHMARK_VERSION, TASK_FAMILY, bind_task, evaluation_plan, load_catalog, summarize
from .evaluator import LongBenchEvaluator, LongBenchEvidenceMonitor, Observation, MinecraftStateAdapter


class ScoredEnvironment:
    """Pair visual observations with simulator state for evaluation."""

    def __init__(self, environment, task, graph, trace):
        self.environment, self.trace = environment, trace
        self.monitor = LongBenchEvidenceMonitor(
            task_id=task.task_id,
            requested_return_landmark=task.initial_state.get("requested_return", ""),
        )
        self.adapter = MinecraftStateAdapter(evidence_monitor=self.monitor)
        self.evaluator = LongBenchEvaluator(task, graph)
        self.frame = environment.initial_frame()
        if type(self.frame) is not Frame or self.frame.step != 0:
            raise ValueError("reset must return a Frame at step zero")
        self._observe()

    def _observe(self):
        observation = self.environment.evaluation_observation()
        if (
            not isinstance(observation, Observation)
            or observation.step != self.frame.step
            or observation.current_frame_path != self.frame.image_ref
        ):
            raise ValueError("private evidence must match the current frame and step")
        # Derive evaluation flags from simulator observations.
        allowed = {
            "inventory",
            "pose",
            "visible_objects",
            "recent_events",
            "placed_blocks",
            "placed",
        }
        structured = {k: copy.deepcopy(v) for k, v in observation.structured.items() if k in allowed}
        if not isinstance(structured.get("inventory"), dict):
            raise ValueError("native inventory evidence is required")
        if any(
            not isinstance(item, str) or type(count) is not int or count < 0
            for item, count in structured["inventory"].items()
        ):
            raise ValueError("native inventory requires named nonnegative integer counts")
        state = self.adapter.update(replace(observation, structured=structured))
        self.evaluation = self.evaluator.update(self.frame.step, state)
        record = asdict(state)
        record["visible"] = sorted(state.visible)
        self.trace.write(
            json.dumps(
                {
                    "step": self.frame.step,
                    "frame": self.frame.image_ref,
                    "state": record,
                    "goal_met": self.evaluation.goal_met,
                    "reached_nodes": sorted(self.evaluator.reached),
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.trace.flush()

    def step(self, action):
        if not isinstance(action, NativeAction):
            raise TypeError("only native actions are supported")
        frame = self.environment.step(action)
        if type(frame) is not Frame or frame.step != self.frame.step + 1:
            raise ValueError("one native action must advance exactly one frame")
        self.frame = frame
        self._observe()
        return frame


def run_suite(make_environment, make_harness, identity, seeds, output, *, catalog=None):
    """Run each episode with fresh adapters and save results in a new directory."""
    catalog = catalog or load_catalog()
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must be integers in [0, 2**32)")
    plan = evaluation_plan(catalog, seeds)
    if not identity.get("model_id") or not identity.get("planner_model"):
        raise ValueError("identity requires model_id and planner_model")
    run_info = {
        "benchmark": BENCHMARK_VERSION,
        "model_id": identity["model_id"],
        "planner_model": identity["planner_model"],
        "settings": copy.deepcopy(identity.get("settings", {})),
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (root / "run_info.json").write_text(json.dumps(run_info, indent=2, allow_nan=False) + "\n")
    rows, configuration = [], None
    with (root / "results.jsonl").open("x") as results:
        for entry in plan["episodes"]:
            task = bind_task(catalog.task(entry["task_id"]), entry["seed"])
            episode = root / f"{entry['seed']}_{task.task_id}"
            episode.mkdir()
            env = None
            try:
                env = make_environment(copy.deepcopy(task), entry["seed"])
                harness = make_harness(task.instruction, entry["seed"])
                if not isinstance(harness, Harness):
                    raise ValueError("adapter must return a Harness")
                contract = asdict(harness.config)
                if configuration is not None and configuration != contract:
                    raise ValueError("Harness configuration changed within the suite")
                configuration = contract
                with (episode / "private_evaluation.jsonl").open("x") as trace:
                    scored = ScoredEnvironment(env, task, catalog.diagnostic(task.task_id), trace)
                    original_renderer = harness.renderer

                    def render(skill, parameters):
                        scored.monitor.begin_skill(skill.skill_id, parameters)
                        return original_renderer(skill, parameters)

                    public_task = replace(task, initial_state={}, goal_predicates=(), constraints={}, tags=())
                    harness.renderer = render
                    started = time.monotonic()
                    try:
                        result = harness.run(public_task, scored, scored.frame)
                    finally:
                        harness.renderer = original_renderer
                    row = {
                        **run_info,
                        "task_id": task.task_id,
                        "seed": entry["seed"],
                        "family": TASK_FAMILY[task.task_id],
                        "success": scored.evaluation.goal_met,
                        "reached_nodes": sorted(scored.evaluator.reached),
                        "termination": result.termination,
                        "action_steps": scored.frame.step,
                        "elapsed_seconds": time.monotonic() - started,
                        "planner_calls": result.planner_calls,
                        "detector_handoffs": result.detector_handoffs,
                        "local_retries": result.local_retries,
                        "harness_config": contract,
                    }
                env.close()
                env = None
                (episode / "result.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
                results.write(json.dumps(row, allow_nan=False) + "\n")
                results.flush()
                rows.append(row)
            finally:
                if env is not None:
                    env.close()
    summary = summarize(rows, catalog, seeds)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
