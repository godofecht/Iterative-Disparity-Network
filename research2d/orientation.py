"""Unsupervised 2D feature-orientation experiment based on Eglen et al. Fig. 3.

Paper-scale 600x600 image, 12x12 elliptical texture elements, 5x5 input
patches, 120x120 virtual output lattice, nine tanh hidden units, and global
versus local spatial variance with half-life one. Evaluation-only orientation
fitting uses measured principal axes, never supervision during training.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiment import SpatialModel


def oriented_image(seed: int, size: int = 600) -> tuple[np.ndarray, np.ndarray]:
    if size % 60 or size < 60:
        raise ValueError("orientation image dimensions must be multiples of 60")
    rng = np.random.default_rng(seed)
    nt = size // 12
    image = np.zeros((size, size), dtype=np.float64)
    tile_y, tile_x = np.mgrid[:12, :12].astype(float)
    p = tile_x - 5.5
    q = tile_y - 5.5
    for ty in range(nt):
        for tx in range(nt):
            theta = (0.85 * np.sin(2 * np.pi * (tx + .5) / nt)
                     + 0.45 * np.cos(2 * np.pi * (ty + .5) / nt))
            major = rng.uniform(4, 11)
            minor = major / 3
            co, si = math.cos(theta), math.sin(theta)
            u = co * p + si * q
            v = -si * p + co * q
            ellipse = (u / (major / 2)) ** 2 + (v / (minor / 2)) ** 2 <= 1
            image[ty * 12:(ty + 1) * 12, tx * 12:(tx + 1) * 12] = ellipse

    # Explicit per-patch normalization avoids optimizing average brightness.
    h = size // 5
    patch = image.reshape(h, 5, h, 5).transpose(0, 2, 1, 3).reshape(-1, 25)
    patch = patch - patch.mean(axis=1, keepdims=True)
    stdev = patch.std(axis=1, keepdims=True)
    patch = np.divide(patch, stdev, out=np.zeros_like(patch), where=stdev > 0)
    return image, np.column_stack((patch, np.ones(h * h)))


def measured_orientation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal-axis orientation and conservative ellipse-detection mask.

    Include patches with >=5 nonzero pixels and moment roundness <0.2.
    These orientation estimates are evaluation-only, not given to the model.
    """
    h = image.shape[0] // 5
    patch = image.reshape(h, 5, h, 5).transpose(0, 2, 1, 3).reshape(-1, 25)
    yy, xx = np.mgrid[:5, :5]
    vx, vy = xx.ravel() - 2.0, yy.ravel() - 2.0
    count = patch.sum(axis=1)
    ax = (patch @ vx) / np.maximum(count, 1)
    ay = (patch @ vy) / np.maximum(count, 1)
    dx = vx[None, :] - ax[:, None]
    dy = vy[None, :] - ay[:, None]
    cxx = (patch * dx ** 2).sum(axis=1) / np.maximum(count, 1)
    cyy = (patch * dy ** 2).sum(axis=1) / np.maximum(count, 1)
    cxy = (patch * dx * dy).sum(axis=1) / np.maximum(count, 1)
    trace = cxx + cyy
    discr = np.sqrt(np.maximum(0, (cxx - cyy) ** 2 + 4 * cxy ** 2))
    eig_large = .5 * (trace + discr)
    eig_small = .5 * (trace - discr)
    roundness = eig_small / np.maximum(eig_large, 1e-10)
    angle = .5 * np.arctan2(2 * cxy, cxx - cyy)
    mask = (count >= 5) & (roundness < .2)
    return angle, mask


def fit_orientation_curve(angle: np.ndarray, response: np.ndarray, mask: np.ndarray):
    x = np.column_stack((np.sin(2 * angle), np.cos(2 * angle), np.ones_like(angle)))
    return np.linalg.lstsq(x[mask], response.ravel()[mask], rcond=None)[0]


def variance_explained(angle: np.ndarray, response: np.ndarray,
                       mask: np.ndarray, coefficients) -> float:
    predicted = (coefficients[0] * np.sin(2 * angle)
                 + coefficients[1] * np.cos(2 * angle) + coefficients[2])
    actual = response.ravel()
    denom = ((actual[mask] - actual[mask].mean()) ** 2).sum()
    return float(1.0 - ((actual[mask] - predicted[mask]) ** 2).sum() / denom) if denom > 1e-12 else 0.0


def run(seed: int = 42, epochs: int = 1000, size: int = 600) -> dict:
    image, data = oriented_image(seed + 1000, size)
    unseen_image, unseen_data = oriented_image(seed + 1001, size)
    grid = size // 5
    net = SpatialModel(data, grid, half_life=1.0, hidden_units=9)
    heldout = SpatialModel(unseen_data, grid, half_life=1.0, hidden_units=9)
    initial = SpatialModel.initial_weights(seed + 2000, features=26, hidden_units=9)
    history = []
    index = 0

    def save(weight):
        f, _ = net.evaluate(weight)
        history.append({"epoch": index, "merit": -f})

    save(initial)

    def callback(weight):
        nonlocal index
        index += 1
        save(weight)

    result = minimize(net.evaluate, initial, jac=True, method="CG", callback=callback,
                      options={"maxiter": epochs, "gtol": 1e-7})
    response = net.predict(result.x)
    test_response = heldout.predict(result.x)
    angle, mask = measured_orientation(image)
    test_angle, test_mask = measured_orientation(unseen_image)
    coefficients = fit_orientation_curve(angle, response, mask)
    return {
        "meta": {"seed": seed, "size": size, "grid": grid,
                 "epochs_completed": index, "epochs_requested": epochs,
                 "architecture": "26-9-1 with 5x5 normalized patches",
                 "train_texture_seed": seed + 1000,
                 "test_texture_seed": seed + 1001,
                 "optimizer": "scipy.optimize.minimize(CG)",
                 "converged": bool(result.success),
                 "optimizer_message": str(result.message)},
        "metrics": {"initial_merit": history[0]["merit"],
                    "final_merit": history[-1]["merit"],
                    "training_orientation_r2": variance_explained(angle, response, mask, coefficients),
                    "test_orientation_r2": variance_explained(test_angle, test_response, test_mask, coefficients),
                    "train_evaluable_patches": int(mask.sum()),
                    "test_evaluable_patches": int(test_mask.sum())},
        "images": {"train_input": image.ravel().tolist(),
                   "test_input": unseen_image.ravel().tolist(),
                   "train_output": response.ravel().tolist(),
                   "test_output": test_response.ravel().tolist(),
                   "train_orientation": angle.tolist(),
                   "test_orientation": test_angle.tolist()},
        "history": history,
        "weights": result.x.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Unsupervised 2D orientation experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--out", type=Path, default=Path("build/research-orientation.json"))
    args = parser.parse_args()
    result = run(seed=args.seed, epochs=args.epochs, size=args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(json.dumps({"meta": result["meta"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
