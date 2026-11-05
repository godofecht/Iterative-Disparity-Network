#!/usr/bin/env python3
"""Check numerical agreement of independent C++ and R forward implementations."""
import csv
import math
import sys


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return [float(row[0]) for row in rows[1:]]


def main():
    a, b = map(read_csv, sys.argv[1:3])
    if len(a) != len(b) or not a:
        raise AssertionError("mismatched reference predictions")
    error = max(abs(x - y) for x, y in zip(a, b))
    if not math.isfinite(error) or error > 1e-9:
        raise AssertionError(f"R/C++ mismatch: maximum absolute error {error}")
    print(f"R/C++ parity passed: {len(a)} predictions, maximum error={error:.3g}")


if __name__ == "__main__":
    main()
