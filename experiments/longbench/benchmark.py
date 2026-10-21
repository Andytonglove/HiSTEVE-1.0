from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from importlib.resources import files

from .catalog import GoalTask, LongBenchCatalog


FAMILIES = {
    "resource_crafting": ("wooden_age_v1", "stone_age_v1", "iron_tool_v1", "glass_maker_v1"),
    "exploration_memory": ("cave_survey_v1", "camp_return_v1", "field_camp_v1", "resource_revisit_v1"),
    "recovery": ("tool_break_recovery_v1", "lost_drop_recovery_v1", "fuel_alternative_v1"),
    "open_ended": ("cook_dinner_v1", "light_up_v1", "survival_iron_ingot_v2"),
}
TASK_FAMILY = {task: family for family, tasks in FAMILIES.items() for task in tasks}
BENCHMARK_VERSION = "longbench-v1"


def bind_task(task: GoalTask, seed: int) -> GoalTask:
    """Choose a return landmark deterministically from the task and seed."""
    if task.task_id != "resource_revisit_v1":
        return task
    digest = hashlib.sha256(f"longbench-paper-v1:{seed}:{task.task_id}".encode()).digest()
    target = ("tree_site", "stone_site")[digest[0] % 2]
    return replace(
        task,
        initial_state={**task.initial_state, "requested_return": target},
        instruction=task.instruction + f" The requested stored location is {target.replace('_', ' ')}.",
    )


def load_catalog(root: str | Path | None = None) -> LongBenchCatalog:
    root = Path(root) if root is not None else Path(str(files(__package__).joinpath("data")))
    catalog = LongBenchCatalog.from_json(root / "public_tasks.json", root / "hidden_diagnostics.json")
    if set(catalog.tasks) != set(TASK_FAMILY) or set(catalog.diagnostics) != set(TASK_FAMILY):
        raise ValueError("task catalog must contain the 14 registered tasks")
    if sum(len(graph.nodes) for graph in catalog.diagnostics.values()) != 49:
        raise ValueError("diagnostic catalog must contain 49 nodes")
    return catalog


def evaluation_plan(catalog: LongBenchCatalog, seeds: Sequence[int]) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    episodes = []
    for seed in seeds:
        for task_id, family in TASK_FAMILY.items():
            task = bind_task(catalog.task(task_id), seed)
            episodes.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "family": family,
                    "model_input": {"instruction": task.instruction},
                    "harness_limits": dict(task.budget),
                    "reset_only": dict(task.initial_state),
                    "evaluator_only": {
                        "goal_predicates": list(task.goal_predicates),
                        "constraints": dict(task.constraints),
                    },
                }
            )
    return {"benchmark": BENCHMARK_VERSION, "episodes": episodes}


def summarize(
    rows: Sequence[Mapping[str, Any]], catalog: LongBenchCatalog, seeds: Sequence[int]
) -> dict[str, Any]:
    """Average success across families, then summarize complete seed runs.

    Rates are fractions. Progress is weighted by diagnostic node count.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    expected = {(task, seed) for task in TASK_FAMILY for seed in seeds}
    indexed = {}
    configurations = set()
    for row in rows:
        if row.get("task_id") not in TASK_FAMILY or type(row.get("seed")) is not int:
            raise ValueError("each result requires a known task_id and integer seed")
        key = (row["task_id"], row["seed"])
        if key not in expected or key in indexed:
            raise ValueError(f"unexpected or duplicate episode: {key}")
        if type(row.get("success")) is not bool:
            raise ValueError(f"episode success must be true or false: {key}")
        ids = row.get("reached_nodes")
        known = {node.node_id for node in catalog.diagnostic(key[0]).nodes}
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or not set(ids) <= known:
            raise ValueError(f"invalid diagnostic IDs: {key}")
        if row.get("benchmark") != BENCHMARK_VERSION:
            raise ValueError(f"unexpected benchmark version: {key}")
        if not row.get("model_id") or not row.get("planner_model"):
            raise ValueError("model_id and planner_model are required")
        configurations.add(
            json.dumps(
                {
                    "model_id": row["model_id"],
                    "planner_model": row["planner_model"],
                    "settings": row.get("settings", {}),
                    "harness_config": row.get("harness_config"),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(f"incomplete suite: missing {len(expected - set(indexed))} episodes")
    if len(configurations) != 1:
        raise ValueError("cannot combine results from different model or harness configurations")
    family_rates = {
        family: statistics.mean(indexed[task, seed]["success"] for task in tasks for seed in seeds)
        for family, tasks in FAMILIES.items()
    }
    repeat_family = {
        family: [statistics.mean(indexed[task, seed]["success"] for task in tasks) for seed in seeds]
        for family, tasks in FAMILIES.items()
    }
    repeat_macro = [
        statistics.mean(repeat_family[family][index] for family in FAMILIES) for index in range(len(seeds))
    ]
    from .statistics import summarize_repeats

    nodes_per_repeat = sum(len(catalog.diagnostic(task).nodes) for task in TASK_FAMILY)
    repeat_progress = [
        sum(len(indexed[task, seed]["reached_nodes"]) for task in TASK_FAMILY) / nodes_per_repeat
        for seed in seeds
    ]
    progress_statistics = summarize_repeats(
        dict(zip(map(str, seeds), repeat_progress)), list(map(str, seeds))
    )
    reached = sum(len(row["reached_nodes"]) for row in indexed.values())
    total = sum(len(catalog.diagnostic(task).nodes) for task, _ in indexed)
    return {
        "episodes": len(indexed),
        "family_goal_sr": family_rates,
        "goal_sr_family_macro": statistics.mean(family_rates.values()),
        "goal_sr_episode_micro": statistics.mean(row["success"] for row in indexed.values()),
        "diagnostic_progress_node_weighted": reached / total,
        "repeat_diagnostic_progress_node_weighted": repeat_progress,
        "diagnostic_progress_statistics": progress_statistics,
        "reached_nodes": reached,
        "total_nodes": total,
        "repeat_family_goal_sr": repeat_family,
        "family_sd_population": {
            family: statistics.pstdev(values) for family, values in repeat_family.items()
        },
        "family_sd_sample": {
            family: statistics.stdev(values) if len(values) > 1 else None
            for family, values in repeat_family.items()
        },
        "repeat_family_macro": repeat_macro,
        "sd_population": statistics.pstdev(repeat_macro),
        "sd_sample": statistics.stdev(repeat_macro) if len(seeds) > 1 else None,
        "sem": statistics.stdev(repeat_macro) / len(seeds) ** 0.5 if len(seeds) > 1 else None,
        "sd_unit": "fraction",
        "repeat_unit": "complete_seed_suite",
        "run_config": json.loads(next(iter(configurations))),
    }
