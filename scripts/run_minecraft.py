"""Run a user-provided Minecraft/model integration.

python -m scripts.run_minecraft --adapter my_adapter --goal "Obtain one iron ingot."
The module exports visual_query, action_model, detector_model and make_environment(seed).
"""

import argparse
import importlib

from histeve.minecraft.runtime import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--minutes", type=float, default=20)
    args = parser.parse_args()
    adapter = importlib.import_module(args.adapter)
    env = adapter.make_environment(args.seed)
    try:
        result = run(
            args.goal,
            env,
            steps=args.steps,
            minutes=args.minutes,
            visual_query=adapter.visual_query,
            action_model=adapter.action_model,
            detector_model=adapter.detector_model,
            seed=args.seed,
        )
        print(f"{result.termination}: {result.steps} actions, {result.planner_calls} planner calls")
    finally:
        env.close()


if __name__ == "__main__":
    main()
