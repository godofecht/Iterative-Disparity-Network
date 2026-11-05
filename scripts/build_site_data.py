#!/usr/bin/env python3
"""Build a fully static GitHub Pages report from a *completed CI training run*.

No network calls, no seeded/fake browser data, no ML dependencies. Fail closed
when experiment files are missing, inconsistent, or non-finite.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil


def raw_values(path: Path) -> list[float]:
    numbers = [float(v) for v in path.read_text(encoding="utf-8").split()]
    ensure(numbers, f"empty input: {path}")
    ensure(all(math.isfinite(v) for v in numbers), f"non-finite input: {path}")
    return numbers


def column(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    ensure(len(rows) >= 2 and len(rows[0]) == 1, f"invalid column CSV: {path}")
    numbers = [float(row[0]) for row in rows[1:]]
    ensure(all(math.isfinite(v) for v in numbers), f"non-finite CSV: {path}")
    return numbers


def ensure(condition: object, error: str) -> None:
    if not condition:
        raise ValueError(error)


def pearson(a: list[float], b: list[float]) -> float:
    ensure(len(a) == len(b) and len(a) > 2, "incompatible correlation arrays")
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    ensure(va > 0 and vb > 0, "constant output cannot be correlated")
    return cov / math.sqrt(va * vb)


def read_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream))
    ensure(data and int(data[0]["epoch"]) == 0, "history is missing epoch zero")
    keys = ("merit", "short_variance", "long_variance", "gradient_norm", "step")
    result = []
    for item in data:
        values = {k: float(item[k]) for k in keys}
        ensure(all(math.isfinite(v) for v in values.values()), "non-finite history entry")
        values["epoch"] = int(item["epoch"])
        result.append(values)
    ensure(all(result[i]["epoch"] > result[i - 1]["epoch"]
               for i in range(1, len(result))), "history epochs are unordered")
    return result


def read_checkpoints(path: Path, width: int, truth: list[float]) -> list[dict]:
    checkpoints = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            ensure(len(row) == width + 1, "checkpoint has wrong number of outputs")
            epoch = int(row[0])
            values = [float(x) for x in row[1:]]
            ensure(all(math.isfinite(v) for v in values), "checkpoint contains non-finite values")
            ensure(not checkpoints or epoch > checkpoints[-1]["epoch"],
                   "checkpoint epochs must be strictly increasing")
            checkpoints.append({"epoch": epoch, "predictions": values,
                                "correlation": pearson(values, truth)})
    ensure(checkpoints and checkpoints[0]["epoch"] == 0, "missing epoch-zero checkpoint")
    return checkpoints


def build(input_dir: Path, site: Path, output: Path, run_url: str,
          sha: str, stride: int, seed: int, epochs: int) -> dict:
    ensure(stride >= 5, "patch stride must be at least five")
    left_file = input_dir / "image1.csv"
    right_file = input_dir / "image2.csv"
    truth_file = input_dir / "disparity.csv"
    results = input_dir / "research-results"
    left = raw_values(left_file)
    right = raw_values(right_file)
    truth = column(truth_file)
    weights = column(results / "weights.csv")
    predictions = column(results / "predictions.csv")
    history = read_history(results / "training_metrics.csv")
    ensure(len(left) == len(right) == len(truth) * stride,
           "stereo source must consist of N consecutive stride-sized patches")
    ensure(len(weights) == 120 and len(predictions) == len(truth),
           "11-10-1 model or prediction shape is inconsistent")
    ensure(history[-1]["epoch"] <= epochs and history[-1]["merit"] >= history[0]["merit"],
           "objective did not improve or completed too many steps")
    checkpoints = read_checkpoints(results / "checkpoints.csv", len(truth), truth)
    ensure(checkpoints[-1]["epoch"] == history[-1]["epoch"],
           "final checkpoint does not match final training epoch")
    ensure(max(abs(a - b) for a, b in zip(predictions, checkpoints[-1]["predictions"])) < 1e-10,
           "final checkpoint must equal final predictions")

    activations = [[] for _ in range(10)]
    for sample in range(len(truth)):
        start = sample * stride
        patch = left[start:start + 5] + right[start:start + 5] + [1.0]
        for hidden in range(10):
            offset = hidden * 11
            activations[hidden].append(
                math.tanh(sum(weights[offset + i] * patch[i] for i in range(11))))
    from_hidden = [
        sum(weights[110 + h] * activations[h][i] for h in range(10))
        for i in range(len(truth))
    ]
    ensure(max(abs(a - b) for a, b in zip(from_hidden, predictions)) < 1e-9,
           "the documented 11-10-1 forward pass does not reproduce output")

    corr = pearson(predictions, truth)
    fingerprints = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (left_file, right_file, truth_file)
    }
    result = {
        "schema_version": 1,
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "run_url": run_url,
            "commit": sha,
            "seed": seed,
            "requested_epochs": epochs,
            "completed_epochs": history[-1]["epoch"],
            "stride": stride,
            "dataset_sha256": fingerprints,
            "experiment": "one-dimensional shared-weight stereo micro-networks",
            "validation": "CI generated, checked gradients, validated R parity",
        },
        "metrics": {
            "samples": len(truth),
            "input_values_per_eye": len(left),
            "trainable_weights": len(weights),
            "initial_merit": history[0]["merit"],
            "final_merit": history[-1]["merit"],
            "correlation": corr,
            "absolute_correlation": abs(corr),
            "short_variance": history[-1]["short_variance"],
            "long_variance": history[-1]["long_variance"],
        },
        "signals": {
            "left": left, "right": right, "truth": truth,
            "final_prediction": predictions, "weights": weights,
            "final_hidden_activations": activations,
        },
        "history": history,
        "checkpoints": checkpoints,
    }
    ensure(site.is_dir() and (site / "index.html").is_file(), "site source unavailable")
    shutil.copytree(site, output, dirs_exist_ok=True)
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "experiment.json").write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    for name in ("weights.csv", "predictions.csv", "training_metrics.csv", "checkpoints.csv"):
        shutil.copyfile(results / name, data / name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    result = build(args.input_dir, args.site_dir, args.out,
                   args.run_url, args.commit, args.stride, args.seed, args.epochs)
    print("Pages experiment bundle validated: "
          f"{result['metrics']['samples']} positions, "
          f"{len(result['checkpoints'])} real network snapshots, "
          f"|r|={result['metrics']['absolute_correlation']:.9f}, "
          f"F={result['metrics']['final_merit']:.9f}")


if __name__ == "__main__":
    main()
