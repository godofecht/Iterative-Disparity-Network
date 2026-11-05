# Two-dimensional spatial-invariance experiments

This directory implements two *actual learning experiments* following the experimental designs in Stephen Eglen, Alistair Bray and Jim Stone, **Unsupervised discovery of invariances**, *Network: Computation in Neural Systems* **8** (1997), 441–452, [PDF](https://www.damtp.cam.ac.uk/user/sje30/papers/network97.pdf). These are independent, documented reconstructions, **not** a claim that the original authors' code, random draws, or numerical results have been reproduced exactly.

The [interactive 2D research dashboard](https://godofecht.github.io/Iterative-Disparity-Network/2d.html) is deployed **directly from GitHub Actions output**, alongside the original [1D dashboard](https://godofecht.github.io/Iterative-Disparity-Network/).

## Stereo disparity (paper's figure 2)

**Training input:** independent pseudo-random binary dot texture (20% black), repeated onto a 1200×1200 high-resolution grid, sampled into a **120×120 stereo pair** with a radial Gaussian-disparity function. Disparity ranges approximately ±1 pixel, quantized to **0.1 pixel**. Gaussian image blur σ=1 pixel and zero-mean/unit-variance normalization are applied separately to each eye. The reconstruction's high-resolution dot placement and interpolation are documented in the source, and may differ from the paper's original generator.

**Network:** disjoint 3×3 pixels from each eye plus one bias input; **19 inputs, five tanh hidden units, one linear output**, and a **40×40 virtual output lattice** with shared weights (100 trainable weights). Training maximizes the spatial-invariance merit function

\[
F(w)=\log\left(\frac{V(w)+10^{-12}}{U(w)+10^{-12}}\right).
\]

Here \(U=\frac12\sum((K_{\rm short}*y)-y)^2\), \(V=\frac12\sum((K_{\rm global}*y)-y)^2\), and the short-range kernel is an exponential with radial half-life **2 output positions**. All convolution is **toroidal** (circular). The global averaging kernel is uniform. The numerically efficient implementation uses Fourier-domain spectral kernels to evaluate the exact analytic gradient of this objective, with central-finite-difference verification of **all 100 parameters** in a smaller test grid. It uses SciPy's nonlinear conjugate-gradient implementation, not Williams' original line search.

**Generalization:** after training on a radial Gaussian disparity, weights are frozen and evaluated against a **different random-dot texture** and an **egg-box** (product of two sinusoids) disparity field. The exact sinusoidal spatial frequency is a documented reconstruction rather than a demonstrated match to the authors' test function. Ground-truth disparity enters **only** evaluation and visualization; it is not passed into the network or its gradient.

CI runs **1000 optimization iterations** for each of **three predeclared seeds (42, 43, 44)** and reports the raw and absolute Pearson r, training merit, and full epoch progression. One complete seed's actual output matrices and 21 model checkpoints are embedded into the static public dashboard; other seeds are fully reported as rows and aggregate statistics.

## Feature orientation (paper's figure 3)

**Input:** a **600×600** black background divided into **12×12** regions containing white, random-length **ellipses**, major axis 4–11 pixels, aspect ratio 3, with orientations varying smoothly across the input. A second, independently seeded ellipse image is used for evaluation. The precise angular field and ellipse rasterization are reconstructions.

**Network:** non-overlapping **5×5** patches individually normalized for mean and standard deviation; 25 intensity inputs plus bias, **nine tanh hidden units**, one linear output, and a **120×120** virtual output lattice. The same spatial objective is optimized with short-range half-life **1** and uniform global averaging. There are **243** parameters.

**Orientation evaluation** is deliberately separate from training: after freezing the weights, an angle is estimated from the principal axis of a patch's intensity second moment, retaining patches with **at least five nonzero pixels and moment roundness below 0.2**. A periodic diagnostic function \(a\sin(2\theta)+b\cos(2\theta)+c\) is fitted to **training outputs only**. The fit is then applied unchanged to the second, unseen texture; the reported \(R^2\) measures explained *output variance*, not stereo correlation and not classification accuracy.

The original study reported orientation learning merit of about 0.9 after 1000 steps, with 15-run mean 0.903; this implementation reports its own measured objective and orientation score without overwriting the historical benchmark.

## Running

Requires Python 3.12 and the pinned NumPy/SciPy versions:

\`\`\`sh
python -m pip install -r research2d/requirements.txt
python -m unittest discover -s research2d -p 'test_*.py' -v
python research2d/experiment.py --seed 42 --epochs 1000 --size 120 --out build/stereo-2d.json
python research2d/orientation.py --seed 42 --epochs 1000 --size 600 --out build/orientation-2d.json
python research2d/build_report.py --seeds 42,43,44 --epochs 1000 --orientation-epochs 1000 --commit LOCAL --run-url https://github.com/godofecht/Iterative-Disparity-Network/actions --out build/research-2d-report.json
\`\`\`

GitHub Actions' **Build and validate** workflow rebuilds this research report on every \`main\` push. CI runs all C++ and R tests, then the 2D gradient/stimulus tests and all seeded experiments, and only **after every platform succeeds** publishes the report to GitHub Pages. Download \`research-two-dimensional\` from the matching GitHub Actions run to obtain all experiment arrays and metadata.

## Scientific limitations

This is a **working 2D research extension with independent held-out testing**, not a mathematically identical reproduction of every published outcome. In particular, the original stimulus texture distribution, the exact egg-box waveform, ellipse orientation field, original Williams optimizer and weight initialization are not available to us as executable specifications. The dashboard displays actual discrepancies with the published benchmarks, not adjusted or cherry-picked outputs.

No claims about convergence to a unique optimum or generalization to natural images are warranted. Three seeds measure basic robustness but cannot replace the paper's ten or fifteen runs or confidence intervals from large replications. Additional model ablations, reproducing the authors' exact random-dot generator, and pre-registered threshold tests belong in follow-up research.
