# Iterative Disparity Network

A reproducible C++17 experiment in **unsupervised stereo-disparity learning** via spatial invariance. This is a repaired, runnable implementation of the repository's original one-dimensional 1,000-position experiment, rather than a claim of reproducing the full two-dimensional benchmark.

## Research basis

Eglen, Bray & Stone (1997), *Unsupervised discovery of invariances*, Network: Computation in Neural Systems 8:441–452, [author-hosted paper](https://www.damtp.cam.ac.uk/user/sje30/papers/network97.pdf). Related earlier work: Eglen, Stone & Barrow (1996), *Learning perceptual invariances: A spatial model*.

Each virtual micro-network has 11 inputs (five stereo intensity samples from each view plus an explicit bias), 10 tanh hidden units, and one linear output. All 1,000 micro-networks share the same 120 weights. Input patches are selected every seven samples from the provided 7,000-sample stereo sequences.

The **spatial** objective is to maximize

F = log((V + epsilon) / (U + epsilon))

where U = 0.5 * sum((K_short * y - y)^2) and V = 0.5 * sum((K_long * y - y)^2). The asterisk denotes circular convolution on the one-dimensional output lattice. The short and long kernels are normalized exponentials of toroidal distance, with half-lives 32 and 3,200 by default, matching the legacy code's configuration. Epsilon = 1e-12 prevents divisions by zero. This stabilizer makes the numerical objective a tiny regularized variant of the paper's exact ratio.

All 120 analytic derivatives include the derivative of each spatial average with respect to every output. Their correctness is checked independently by central finite differences. Optimization uses nonlinear **Polak–Ribière+ conjugate-gradient ascent** and Armijo backtracking. The 1993 Williams conjugate-gradient implementation remains preserved in the legacy sources, but is not used in the new build; PR+ is not claimed to be the same optimizer.

Ground-truth disparity is **never used to train**. The optional reference is used only to report Pearson r and |r|, since learned representations are sign-indeterminate. Meaningful scientific replication additionally requires evaluation on unseen random-dot stereo pairs.

## Build and run

Requires a C++17 compiler and CMake 3.16+.

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --parallel
    ctest --test-dir build -C Release --output-on-failure

From the build directory on Linux or macOS:

    cd build
    ./DisparityNetwork --epochs 2000 --seed 42 --output-dir results

On Windows the executable is typically under build/Release; run it with build as the current working directory. CLI options include --left, --right, --reference, --stride, --epochs, --seed, --short-half-life, --long-half-life, and --output-dir. Run with --help for usage.

The generated directory contains weights.csv (120 parameters in **hidden-major** order: 11 incoming weights per hidden unit, then 10 output weights), predictions.csv (one value per sample), and training_metrics.csv (objective, short and long variances, gradient norm and accepted step). The legacy logs are retained but ignored by the new training executable.

## Independent R reference

The R implementation computes the same patch extraction and forward pass **independently**. With an R runtime and Python 3 available, run from the repository root after the C++ executable:

    Rscript r_script.R image1.csv image2.csv build/results/weights.csv build/results/r_predictions.csv
    python3 tests/check_r_parity.py build/results/predictions.csv build/results/r_predictions.csv

Predictions must agree within 1e-9 absolute error. The GitHub Actions matrix builds and tests Linux, macOS, and Windows; Linux additionally installs R and checks this cross-language parity using the repository's actual data.

## Verified baseline on repository data

The 5 November 2026 CI run on the original 7,000-value left/right images (1,000 stereo patches), with seed 42 and 1,000 Polak–Ribière+ iterations, produced **F = 2.406892828** (from initial **0.009222471354**) and **Pearson r = +0.9572882731** against the *evaluation-only* supplied disparity. The historical `outputs.csv` snapshot has **|r| = 0.9615181** against the same `disparity.csv`; these are related baselines, not identical optimization trajectories or generalization experiments.

GitHub Actions [run 41944098113](https://github.com/godofecht/Iterative-Disparity-Network/actions/runs/41944098113) passed on Linux, macOS, and Windows, with Linux performing an R/C++ cross-check (maximum absolute difference approximately **5e-16**) and an objective/correlation regression assertion. Its **disparity-research-results** artifact contains the numerical predictions, model weights and all per-iteration metrics; it can be downloaded directly from the run. The performance numbers here are for the 1D dataset only.

## Fidelity and limitations

This repository's original dataset is a **1D sequence of stereo patches**, whereas Eglen et al. (1997) used two-dimensional output lattices, among other differences in architecture, stimulus generation, and smoothing. The repaired 1D experiment tests the fundamental spatial-invariance learning rule, but must **not** be treated as a reproduction of the paper's published 2D correlation or generalization results. Such a replication needs its own 2D random-dot generator, 2D toroidal masks, train/test split, repeated seeds, and preregistered acceptance criteria.

The original NN.cpp, Network.cpp, disparityNet.h, dispnet.h, and cg_williams_module.c remain in the repository for provenance, but are superseded by the isolated C++17 core. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) before redistributing legacy third-party code.

## License

See [LICENSE](LICENSE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
