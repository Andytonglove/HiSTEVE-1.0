import argparse
import importlib
import json
from pathlib import Path

from . import evaluation_plan, load_catalog, run_suite, summarize


def main():
    parser = argparse.ArgumentParser(description="HiSTEVE LongBench")
    parser.add_argument("command", choices=("plan", "run", "score"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--adapter", help="module exposing make_environment, make_harness and identity")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be integers in [0, 2**32)")
    catalog = load_catalog()
    if args.command == "run":
        if not args.adapter or not args.output:
            parser.error("run requires --adapter and --output")
        adapter = importlib.import_module(args.adapter)
        result = run_suite(
            adapter.make_environment,
            adapter.make_harness,
            adapter.identity,
            args.seeds,
            args.output,
            catalog=catalog,
        )
    elif args.command == "score":
        if not args.results:
            parser.error("score requires --results")
        rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
        result = summarize(rows, catalog, args.seeds)
    else:
        result = evaluation_plan(catalog, args.seeds)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
