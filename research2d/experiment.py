"""Two-dimensional, unsupervised spatial-invariance stereo experiment.

Research basis: Eglen, Bray & Stone (1997), Network 8:441-452, figure 2.
The deterministic stimulus synthesis and SciPy conjugate-gradient line search
are documented reconstructions, not the authors' original code or random draws.
Ground truth is used exclusively after network forward passes for evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize

PIXELS = 120
PATCH = 3
HIDDEN = 5
HALF_LIFE = 2.0
TRAIN_PIXELS = 0.2


def stimulus(scene: str, seed: int, size: int = PIXELS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized stereo views and ground truth, never used in fitting.

    A deterministic high-resolution (10x) upsampling of 20% black random dots
    provides 0.1-pixel disparity increments. The 120x120 stereo images are
    smoothed by Gaussian sigma=1 and individually standardized.
    """
    if size % PATCH != 0 or size < 6:
        raise ValueError("image size must be divisible by 3 and >= 6")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    if scene == "gaussian":
        radius_sq = (xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2
        depth = 2.0 * np.exp(-radius_sq / (2.0 * (size / 4.0) ** 2)) - 1.0
    elif scene == "eggbox":
        depth = np.sin(2 * np.pi * xx / (size / 2.5)) * np.sin(2 * np.pi * yy / (size / 2.5))
    else:
        raise ValueError("scene must be gaussian or eggbox")
    disparity = np.round(depth * 10.0) / 10.0
    base = (rng.random((size, size)) >= TRAIN_PIXELS).astype(np.float64)
    high = np.repeat(np.repeat(base, 10, axis=0), 10, axis=1)
    coordinates_y = yy * 10.0 + 4.5
    coordinates_x = xx * 10.0 + 4.5
    left = map_coordinates(high, [coordinates_y, coordinates_x], order=1, mode="grid-wrap")
    right = map_coordinates(high, [coordinates_y, coordinates_x + disparity * 10.0],
                            order=1, mode="grid-wrap")
    left = gaussian_filter(left, 1.0, mode="wrap")
    right = gaussian_filter(right, 1.0, mode="wrap")
    for data in (left, right):
        sd = float(data.std())
        if sd <= 0 or not np.isfinite(sd):
            raise ValueError("degenerate random-dot stimulus")
        data -= float(data.mean())
        data /= sd
    return left, right, disparity


def patches(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] % PATCH or left.shape[1] % PATCH:
        raise ValueError("stereo images must have equal 2D dimensions divisible by three")
    h, w = left.shape

    def cut(data: np.ndarray) -> np.ndarray:
        return data.reshape(h // PATCH, PATCH, w // PATCH, PATCH).transpose(0, 2, 1, 3).reshape(-1, 9)

    return np.column_stack((cut(left), cut(right), np.ones((h // PATCH) * (w // PATCH))))


def average_disparity(disparity: np.ndarray) -> np.ndarray:
    h, w = disparity.shape
    return disparity.reshape(h // PATCH, PATCH, w // PATCH, PATCH).mean(axis=(1, 3))


def spectral_kernels(rows: int, cols: int, half_life: float = HALF_LIFE) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues squared of (Kshort-I) and (Kglobal-I) on a 2D torus."""
    if rows < 2 or cols < 2 or half_life <= 0:
        raise ValueError("invalid lattice or half life")
    yy, xx = np.ogrid[:rows, :cols]
    wrapped_r = np.hypot(np.minimum(yy, rows - yy), np.minimum(xx, cols - xx))
    short = np.exp(-math.log(2.0) * wrapped_r / half_life)
    short /= short.sum()
    long = np.full((rows, cols), 1.0 / (rows * cols))
    ku = np.abs(np.fft.fft2(short) - 1.0) ** 2
    kv = np.abs(np.fft.fft2(long) - 1.0) ** 2
    return ku, kv


class SpatialModel:
    """A weight-shared tanh micro-network with an exact 2D spectral gradient."""

    def __init__(self, data: np.ndarray, width: int,
                 half_life: float = HALF_LIFE, hidden_units: int = HIDDEN):
        if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] % width:
            raise ValueError("expected NxD patches with width dividing N")
        self.x = data
        self.cols = width
        self.features = data.shape[1]
        self.hidden_units = hidden_units
        self.rows = data.shape[0] // width
        self.ku, self.kv = spectral_kernels(self.rows, self.cols, half_life)
        self.n = data.shape[0]

    @staticmethod
    def initial_weights(seed: int, features: int = 19,
                        hidden_units: int = HIDDEN) -> np.ndarray:
        return np.random.default_rng(seed).normal(0, 0.12, features * hidden_units + hidden_units)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        cut = self.features * self.hidden_units
        w1 = weights[:cut].reshape(self.features, self.hidden_units)
        return (np.tanh(self.x @ w1) @ weights[cut:]).reshape(self.rows, self.cols)

    def evaluate(self, weights: np.ndarray) -> tuple[float, np.ndarray]:
        cut = self.features * self.hidden_units
        w1 = weights[:cut].reshape(self.features, self.hidden_units)
        w2 = weights[cut:]
        hidden = np.tanh(self.x @ w1)
        y = (hidden @ w2).reshape(self.rows, self.cols)
        spectrum = np.fft.fft2(y)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        u = 0.5 / self.n * float(np.sum(power * self.ku))
        v = 0.5 / self.n * float(np.sum(power * self.kv))
        epsilon = 1e-12
        merit = math.log((v + epsilon) / (u + epsilon))
        ratio_spectrum = spectrum * (self.kv / (v + epsilon) - self.ku / (u + epsilon))
        output_gradient = np.fft.ifft2(ratio_spectrum).real.ravel()
        hidden_gradient = (output_gradient[:, None] * w2[None, :]) * (1.0 - hidden ** 2)
        derivative = np.concatenate(((self.x.T @ hidden_gradient).ravel(),
                                     hidden.T @ output_gradient))
        if not np.isfinite(merit) or not np.all(np.isfinite(derivative)):
            raise FloatingPointError("non-finite spatial objective or derivative")
        return -merit, -derivative  # SciPy minimizes; the research maximizes F.


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size or a.size < 2:
        raise ValueError("invalid correlation arrays")
    centered_a, centered_b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)
    return float(centered_a @ centered_b / denom) if denom > 1e-14 else 0.0


def run(seed: int = 42, epochs: int = 1000, size: int = 120,
        checkpoint_every: int = 50) -> dict:
    left, right, training_disparity = stimulus("gaussian", seed + 100, size)
    test_l, test_r, testing_disparity = stimulus("eggbox", seed + 101, size)
    grid = size // 3
    model = SpatialModel(patches(left, right), grid)
    heldout = SpatialModel(patches(test_l, test_r), grid)
    truth = average_disparity(training_disparity)
    heldout_truth = average_disparity(testing_disparity)
    weights0 = SpatialModel.initial_weights(seed)
    history: list[dict] = []
    snapshots: list[dict] = []
    index = 0

    def snapshot(step: int, weights: np.ndarray) -> None:
        train = model.predict(weights)
        test = heldout.predict(weights)
        neg_f, gradient = model.evaluate(weights)
        row = {"epoch": step, "merit": -neg_f,
               "gradient_norm": float(np.linalg.norm(gradient)),
               "train_r": pearson(train, truth),
               "test_r": pearson(test, heldout_truth)}
        history.append(row)
        if step == 0 or step % checkpoint_every == 0 or step == epochs:
            snapshots.append({"epoch": step, "train": train.ravel().tolist(),
                              "test": test.ravel().tolist()})

    snapshot(0, weights0)

    def callback(w: np.ndarray) -> None:
        nonlocal index
        index += 1
        snapshot(index, w)

    result = minimize(model.evaluate, weights0, jac=True, method="CG", callback=callback,
                      options={"maxiter": epochs, "gtol": 1e-7})
    if not snapshots or snapshots[-1]["epoch"] != index:
        snapshots.append({"epoch": index,
                          "train": model.predict(result.x).ravel().tolist(),
                          "test": heldout.predict(result.x).ravel().tolist()})
    final_train = model.predict(result.x)
    final_test = heldout.predict(result.x)
    return {
        "meta": {"seed": seed, "size": size, "patch": PATCH, "grid": grid,
                 "epochs_requested": epochs, "epochs_completed": index,
                 "optimizer": "scipy.optimize.minimize(CG)",
                 "converged": bool(result.success), "optimizer_message": str(result.message),
                 "train_texture_seed": seed + 100, "test_texture_seed": seed + 101,
                 "half_life": HALF_LIFE, "architecture": "19-5-1 with explicit hidden bias"},
        "metrics": {"initial_merit": history[0]["merit"],
                    "final_merit": history[-1]["merit"],
                    "train_r": pearson(final_train, truth),
                    "test_r": pearson(final_test, heldout_truth),
                    "train_abs_r": abs(pearson(final_train, truth)),
                    "test_abs_r": abs(pearson(final_test, heldout_truth))},
        "scenes": {
            "train": {"left": left.ravel().tolist(), "right": right.ravel().tolist(),
                      "truth": truth.ravel().tolist(), "output": final_train.ravel().tolist()},
            "test": {"left": test_l.ravel().tolist(), "right": test_r.ravel().tolist(),
                     "truth": heldout_truth.ravel().tolist(),
                     "output": final_test.ravel().tolist()},
        },
        "history": history, "checkpoints": snapshots,
        "weights": result.x.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="2D stereo disparity spatial-invariance reconstruction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("build/research-2d.json"))
    args = parser.parse_args()
    result = run(seed=args.seed, epochs=args.epochs, size=args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(json.dumps({"meta": result["meta"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
