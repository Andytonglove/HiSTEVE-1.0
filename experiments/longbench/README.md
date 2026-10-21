# LongBench (experimental)

## LongBench

| Family | Tasks |
| --- | --- |
| Resource & Crafting | wooden age, stone age, iron tool, glass maker |
| Exploration & Memory | cave survey, camp return, field camp, resource revisit |
| Recovery | tool break recovery, lost drop recovery, fuel alternative |
| Open-ended | cook dinner, light up, survival iron ingot |

The suite contains 14 final-goal tasks and 49 evaluator-only diagnostic nodes. Eleven tasks start from empty inventory in natural survival; three recovery tasks have controlled reset states. Setup commands are confined to reset. Tasks and per-episode step/time budgets live in `experiments/longbench/data/public_tasks.json`; Minecraft 1.16 reset configurations live in `data/env/`.

```bash
python -m experiments.longbench plan --seeds 260727 260728 260729
python -m experiments.longbench run --adapter my_adapter \
  --seeds 260727 260728 260729 --output runs/longbench
python -m experiments.longbench score --seeds 260727 260728 260729 \
  --results runs/longbench/results.jsonl
```

`my_adapter` is your integration module. It exports:

- `identity`: `model_id`, `planner_model`, and optional `settings` with relevant model/runtime parameters.
- `make_environment(task, seed)`: reset a fresh environment according to the task. Return an adapter with `initial_frame() -> Frame`, `step(NativeAction) -> Frame`, `evaluation_observation() -> experiments.longbench.Observation`, and `close()`.
- `make_harness(goal, seed)`: return a fresh `Harness` with `JsonVisualVerifier(public_goal=goal)`; use `build_harness` from `histeve.minecraft.runtime`.

The environment adapter alone receives reset/scoring fields. Its private `Observation` contains the matching frame path and step plus native inventory, pose, visible objects and cumulative event counters in `structured`. Use `inventory`, `pose`, `visible_objects`, `recent_events`, and `placed_blocks` as applicable. Native inventory is an item-to-count mapping; pose is `{x, y, z}`; events use `kind:item=count`. The benchmark derives navigation and recovery flags from simulator observations.

The runner passes only the goal, budgets and screenshots to the Harness; scoring predicates and native state remain evaluator-side. Each run writes `results.jsonl`, `summary.json`, `run_info.json`, and per-episode evaluator evidence. Use a new output directory for each run. Infrastructure errors stop the run; unsuccessful episodes remain in the denominator. The adapter is responsible for simulator provisioning, reset validation and frame recording.

## Scoring

Three seeds yield 42 episodes. Within each seed, compute success separately for the four families (4/4/3/3 tasks), then average the four rates equally. Report the mean and sample SD (`ddof=1`) across complete seed suites. Episode-micro success is also included.

Progress counts reached diagnostic nodes out of 49 per complete suite. Success uses the terminal native state. Missing or duplicate episodes are rejected, and results from different model/harness settings are not combined. JSON rates and SDs are fractions; multiply by 100 for percentages and percentage points. One repeat has undefined sample SD.


## Scope

This source release provides task specifications, a suite runner, and scoring code. The examples use an adapter supplied by the user; they do not reproduce the paper's historical evaluation runs. The included tests use synthetic observations, and real-environment scoring requires an independently validated MineStudio adapter.
