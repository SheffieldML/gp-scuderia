# R frameworks: kergp, DiceKriging and GauPro

Both packages run through the `kind: r` runner. The harness starts `Rscript --vanilla` on the adapter script, with the task in `GPSCUDERIA_TASK` and the result in `GPSCUDERIA_OUT`. Adapters share `R/gpscuderia/gaussian.R`, which refuses what is outside the exact Gaussian slice and converts wire quantities to and from the normalised scale. R packages install into a private library with `R/install.R`, from `ci/pinned-r.txt` or `ci/latest-r.txt`.

## Conversions

| Wire | kergp | DiceKriging |
|---|---|---|
| `kern.rbf`, `kern.matern32`, `kern.matern52` | `covRadial` with `k1Fun1Gauss`, `k1Fun1Matern3_2`, `k1Fun1Matern5_2`, unit variance (`cov = "corr"`), `iso = 1` when d > 1 | `covtype` `gauss`, `matern3_2`, `matern5_2`; `coef.cov` is the lengthscale replicated per input, `coef.var` the variance |
| `variance` | A constant `covMan` kernel multiplying the unit-variance factor | `coef.var` |
| `kern.bias`, `kern.add`, `kern.prod` | `covComp` formula over the terms (sum of products distributed from the wire tree) | refused: one stationary kernel per model |
| Likelihood noise | `varNoise` | `noise.var` (never `nugget`: a nugget makes a prediction at a training input noise-free) |
| Log likelihood | `-0.5 (n log 2π + 2 Σ log diag L + sseStar)` from the fitted factor, since `gp(estim = FALSE)` leaves it `NA` | the same expression from `mod@T` and `mod@z` |
| Predictive variance | `predict(type = "SK")$sd^2` is latent, so the runner adds the noise | the same |
| Standardised targets | the runner standardises, then adds the `-n log s` Jacobian to the log likelihood and rescales predictions | the same |

Mean and normaliser are mutually exclusive, as on the wire. Both packages take `mean.constant` through the trend, with no coefficient fitted.

## Refusals

The adapters refuse, and never substitute:

- non-exact approximations and non-Gaussian likelihoods;
- `kern.linear`, `kern.white` (for GauPro, its `White` kernel fails when parameters are fixed in release 0.2.17), `kern.bias` (GauPro has no constant kernel) and any kernel outside the table above (folding white noise into the likelihood noise would change the model's parameters);
- ARD and `active_dims` (not yet supported by the runner);
- DiceKriging's Matérn kernels when the input dimension exceeds one: it uses a product of one-dimensional kernels, which is not the isotropic wire kernel. The Gaussian kernel is the same either way, so `rbf` is accepted. The capability `multi_input` records this in the framework files.

`scripts/check_r_refusals.py <framework>` exercises each refusal and a control that must run.

## GauPro conversions

Verified to 1e-15 against the wire reference at fixed parameters, including sums and products of kernels.

- Lengthscale: `beta = -log10(2 l^2)` for the Gaussian kernel and `beta = -2 log10(l)` for Matérn 3/2 and 5/2, repeated for every input dimension (isotropic).
- Noise: GauPro's nugget is relative, `K = s2 (R + nug I)`, where `s2` is the kernel's total variance (for a sum or product, the sum or product of the parts' variances, which GauPro tracks in `kernel$s2`). The runner sets `nug = noise / kernel$s2` with `nug.min` lowered to 1e-12.
- Predictive variance includes the noise; the runner subtracts it before the shared helper adds it back.

## Known deviations from the oracle

At small noise (1e-4) and for duplicated inputs (noise 1e-3), all three packages agree with each other and with an independent numpy Cholesky, and differ from the GPy reference. GPy adds 1e-8 to the noise variance in exact inference. These cases are recorded as known in `benchmarks/conditioning.yaml`.

## Not yet covered

Gradients (no `operations` are declared, so gradient benchmarks skip), fitting, ARD and `active_dims`.
