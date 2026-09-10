# Iterative-Disparity-Network

Stereo disparity learned by a small network, optimised with conjugate gradient
instead of plain gradient descent, and checked against an independent
implementation in R.

This is the second pass at
[Disparity-Network](https://github.com/godofecht/Disparity-Network). Three
things changed.

**Topology.** 11 inputs (five pixels from each image plus a bias unit), 10
hidden, 1 output. The bias unit is explicit rather than folded into the weight
matrix.

**Optimiser.** `cg_williams_module.c` is Peter Williams' conjugate gradient
method. It takes the parameter vector, a function to minimise, its derivative,
and a convergence test, and it does its own line search and second-derivative
estimate. Training runs 2000 epochs and logs every 50.

**A second opinion.** `r_script.R` computes the same forward pass and objective
in R, reading the same `image1`/`image2` data. When the C++ and the R disagree,
one of them is wrong, and that is the point of having both.

Intermediate state is written out as it trains: `batch_inputs.csv`,
`batch_weights.csv`, `batch_disparity.csv`, `weights.h`, `outputs.csv`.

## Build

```bash
cmake -B build && cmake --build build
./build/DisparityNetwork
```

## Licensing and provenance

The network, the disparity task and the harness are mine, under
[LICENSE](LICENSE). The conjugate gradient module is not. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
