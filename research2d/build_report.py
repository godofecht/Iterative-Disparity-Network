"""Generate the public two-dimensional research report from measured experiments.

Runs seeded 2D stereo and orientation studies in CI and includes full data for
seed 42 plus the complete, unfiltered seed-wise summary. Never inserts a
preselected published result into measured arrays.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from experiment import run as train_stereo
from orientation import run as train_orientation


def checked(values: list[dict], field: str) -> dict:
    numbers = [float(d[field]) for d in values]
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError(f"non-finite reported metric: {field}")
    mean = sum(numbers) / len(numbers)
    sd = math.sqrt(sum((x - mean) ** 2 for x in numbers) / len(numbers))
    return {"mean": mean, "sd": sd, "values": numbers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--orientation-epochs", type=int, default=1000)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected distinct and nonempty numeric seeds")
    stereo_results, orientation_results = [], []
    for seed in seeds:
        stereo = train_stereo(seed, args.epochs)
        orientation = train_orientation(seed, args.orientation_epochs, 600)
        if not (stereo["metrics"]["final_merit"] > stereo["metrics"]["initial_merit"] and
                orientation["metrics"]["final_merit"] > orientation["metrics"]["initial_merit"]):
            raise AssertionError("training objective must improve in each independent seed")
        stereo_results.append(stereo)
        orientation_results.append(orientation)
        print(f"SEED {seed}: 2D stereo train |r|={stereo['metrics']['train_abs_r']:.6f}, "
              f"unseen eggbox |r|={stereo['metrics']['test_abs_r']:.6f}, "
              f"orientation train R2={orientation['metrics']['training_orientation_r2']:.6f}, "
              f"unseen orientation R2={orientation['metrics']['test_orientation_r2']:.6f}", flush=True)
    summary = {
        "stereo_train_abs_r": checked([d["metrics"] for d in stereo_results], "train_abs_r"),
        "stereo_test_abs_r": checked([d["metrics"] for d in stereo_results], "test_abs_r"),
        "orientation_train_r2": checked([d["metrics"] for d in orientation_results], "training_orientation_r2"),
        "orientation_test_r2": checked([d["metrics"] for d in orientation_results], "test_orientation_r2"),
        "stereo_final_merit": checked([d["metrics"] for d in stereo_results], "final_merit"),
        "orientation_final_merit": checked([d["metrics"] for d in orientation_results], "final_merit"),
    }
    report = {
        "schema_version": 1,
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "run_url": args.run_url, "commit": args.commit,
            "seeds": seeds, "stereo_epochs": args.epochs,
            "orientation_epochs": args.orientation_epochs,
            "implementation": "NumPy/SciPy FFT-gradient spatial invariance; scipy nonlinear CG",
            "paper": "Eglen, Bray & Stone (1997), Network 8:441–452",
            "note": "Reconstructed deterministic stimuli/optimizer, not the authors original random draws or Williams optimizer",
        },
        "stereo": stereo_results[0], "orientation": orientation_results[0],
        "replications": {
            "stereo": [{"seed": s, **result["metrics"],
                        "epochs_completed": result["meta"]["epochs_completed"]}
                       for s, result in zip(seeds, stereo_results)],
            "orientation": [{"seed": s, **result["metrics"],
                             "epochs_completed": result["meta"]["epochs_completed"]}
                            for s, result in zip(seeds, orientation_results)],
            "summary": summary,
        },
        "paper_comparison": {
            "stereo_train_single_run_abs_r": 0.92,
            "stereo_unseen_single_run_abs_r": 0.89,
            "stereo_ten_run_mean_train_abs_r": 0.905,
            "stereo_ten_run_mean_unseen_abs_r": 0.869,
            "orientation_fifteen_run_mean_merit": 0.903,
            "reference_only": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    print("2D benchmark written:", args.out, "bytes=", args.out.stat().st_size,
          "train |r| mean=", summary["stereo_train_abs_r"]["mean"],
          "unseen |r| mean=", summary["stereo_test_abs_r"]["mean"], flush=True)


if __name__ == "__main__":
    main()
