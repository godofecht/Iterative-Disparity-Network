#!/usr/bin/env python3
"""Regression check for genuinely learned disparity, not just a runnable build."""
import csv
import math
import sys


def values(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return [float(row[0]) for row in rows[1:]]


def correlation(a, b):
    if len(a) != len(b) or len(a) < 2:
        raise AssertionError("Reference and prediction lengths differ")
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    da = [x - mean_a for x in a]
    db = [x - mean_b for x in b]
    denom = math.sqrt(sum(x*x for x in da) * sum(x*x for x in db))
    if denom == 0:
        raise AssertionError("The predictions or references are constant")
    return sum(x*y for x, y in zip(da, db)) / denom


def main():
    pred, reference, metrics = sys.argv[1:4]
    r = correlation(values(pred), values(reference))
    with open(metrics, newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    initial = float(history[0]["merit"])
    final = float(history[-1]["merit"])
    print(f"Real-data benchmark: epochs={history[-1]['epoch']} "
          f"initial_F={initial:.9g} final_F={final:.9g} r={r:.9g} |r|={abs(r):.9g}")
    if not all(math.isfinite(v) for v in (initial, final, r)):
        raise AssertionError("Non-finite model metrics")
    if abs(r) < 0.90:
        raise AssertionError(f"Learned disparity is insufficiently correlated: |r|={abs(r):.6f}")
    if final - initial < 1.0:
        raise AssertionError(f"Objective failed to improve materially: ΔF={final-initial:.6f}")


if __name__ == "__main__":
    main()
