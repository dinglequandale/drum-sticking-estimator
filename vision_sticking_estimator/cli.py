#!/usr/bin/env python3
"""Drum Sticking Estimator CLI."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Estimate per-hit limb assignment (L/R/F) from a drum audio/video file."
    )
    parser.add_argument("input", help="Audio or video file path")
    parser.add_argument("-o", "--output-dir", default=None, help="Directory for intermediate files")
    parser.add_argument("--json", default=None, help="Write JSON output to this file")
    parser.add_argument("--grid", action="store_true", help="Print human-readable grid to stdout")
    parser.add_argument("--no-separate", action="store_true", help="Skip Demucs separation")
    parser.add_argument("--force-separate", action="store_true", help="Always run Demucs")
    parser.add_argument("--handedness", default="right", choices=["right", "left"])
    parser.add_argument("--style", default="crossed", choices=["crossed", "open"])
    parser.add_argument("--ambiguity-threshold", type=float, default=0.65,
                        help="Confidence below this → flagged ambiguous (default: 0.65)")
    parser.add_argument("--self-prior", action="store_true",
                        help="Enable Tier-2 per-video self-prior (experimental, off by default)")
    parser.add_argument("--eval", default=None,
                        help="Path to ground-truth annotation JSON for evaluation")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from src.types import PipelineConfig
    config = PipelineConfig(
        handedness=args.handedness,
        style=args.style,
        ambiguity_threshold=args.ambiguity_threshold,
        self_prior_enabled=args.self_prior,
    )

    separate = "auto"
    if args.no_separate:
        separate = False
    elif args.force_separate:
        separate = True

    from src.pipeline import run
    result = run(
        args.input,
        config=config,
        output_dir=args.output_dir,
        separate=separate,
    )

    if args.json:
        with open(args.json, "w") as f:
            f.write(result.json_output)
        print(f"JSON written to {args.json}")
    else:
        print(result.json_output)

    if args.grid:
        print("\n" + result.grid_output)

    print("\n" + result.pattern_summary)

    n_flagged = sum(1 for h in result.assigned_hits if h.flagged_ambiguous)
    n_total = len(result.assigned_hits)
    print(f"\n{n_total} hits | {n_flagged} flagged ambiguous ({100*n_flagged//max(n_total,1)}%)")

    if args.eval:
        from src.eval import load_annotations, report_all_tiers
        annotations = load_annotations(args.eval)
        print("\n" + report_all_tiers(result, annotations))


if __name__ == "__main__":
    main()
