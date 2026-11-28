# CIP-0007 phase 0 probe notes

Environment: R 4.5.2 (Homebrew, arm64), DiceKriging 1.6.1, kergp 0.5.8, GauPro 0.2.17.
Reference: `R/probe/wire.R`, an independent dense-R implementation of the wire semantics,
agrees with the GPy `x-reference` to ~3e-8 (loglik) and ~2e-8 (mean).

## DiceKriging (probe: `R/probe/dicekriging.R`)

Matches the goldens `ftc_standardize` and `ftc_rbf_white` to machine precision
(loglik 2e-15, mean 1e-15, predictive variance 2e-15).

- Kernels: `gauss` with `coef.cov = lengthscale`, `coef.var = variance` equals `kern.rbf`
  (same factor 1/2 convention). `matern3_2`, `matern5_2` equal `kern.matern32`, `kern.matern52`.
  The lengthscale caveat in CIP-0007 for `gauss` is settled: no conversion is needed.
- Noise: use `noise.var`, not `nugget`. With `nugget`, a prediction at a training input
  returns the observation with zero variance (`ftc_standardize` has Xnew = 3.0, a training
  input; mean 3.8 and variance 0 instead of 3.97 and 0.26). That is data-dependent and wrong
  for the wire, so the adapter must not use `nugget`.
- Predictive variance: `predict(type = "SK")$sd^2` is the latent variance without noise.
  The wire variance includes noise, so the runner adds the noise back.
- Likelihood: `km` does not store a log-likelihood when every parameter is given.
  `logLikFun(c(lengthscale, variance), model)` on a model with known trend and `noise.var`
  (case `LLconcentration_beta`) returns the full Gaussian log-likelihood, with no profiling.
  The runner needs a fitted-structure carrier model for this call (built with
  `optim.method = "BFGS"`, which runs a short optimisation) and a separate fully fixed model
  for prediction, because fitting changes the stored covariance parameters.
- Standardise: parameters live on the normalised scale. The log-likelihood needs the
  `-n log s` Jacobian added.
- `kern.white` plus likelihood noise folds exactly into `noise.var`. Precedent for GPyTorch
  is to refuse rather than fold silently; a decision is needed for DiceKriging.
- Not expressible, must refuse: `kern.add`/`kern.prod` of distinct kernels, `kern.bias`,
  coregionalisation, sparse approximations, non-Gaussian likelihoods. `mean.constant` maps to
  a known `coef.trend` (formula `~1`).

## kergp (probe: `R/probe/kergp.R`)

Matches `ftc_standardize` and `ftc_prod_matern52` (`add(prod(matern52, rbf), bias)`, 2-D)
to ~1e-15 on log-likelihood, mean and latent predictive variance.

- Kernels: `covRadial` with `k1Fun1Gauss`, `k1Fun1Matern3_2`, `k1Fun1Matern5_2`; `theta` is the
  lengthscale and `sigma2` the variance, equal to the wire kernels with no conversion (Gram
  agreement 1e-16). `iso = 1` gives one shared lengthscale; `cov = "corr"` gives a unit-variance
  factor, so a product of two variances is carried once on one factor.
- Composition: `covComp(~ a() * b() + cc(), where = environment())` expresses `kern.add` and
  `kern.prod` of distinct kernels. The kernels must be written `a()`; `~ a * b` fails.
  `kern.bias` is a constant `covMan` kernel (`function(x1, x2, par) par[1]`).
- Fixed parameters: `gp(..., estim = FALSE, varNoise = v, beta = b)` does no estimation. It leaves
  `logLik` as NA, but the log-likelihood follows from `fit$L` and `fit$sseStar`:
  `-0.5 (n log 2 pi + 2 sum log diag L + sseStar)`.
- Noise and prediction: `varNoise` is added to the diagonal. `predict(type = "SK")$sd^2` is the
  latent variance without noise (the runner adds noise back). `forceInterp = TRUE` returns the
  observation with zero variance at training inputs, so the runner must keep the default `FALSE`.
- Mean: a known `beta` (formula `~ 1`, `beta = c`) maps `mean.constant`; zero mean is `beta = 0`.
  Standardise needs the `-n log s` Jacobian added, as for DiceKriging.
- Not yet checked: `kern.white` (a diagonal `covMan` would need the same fold-or-refuse decision),
  ARD (`iso = 0`), and active dimensions. Still refuse: coregionalisation, sparse approximations,
  non-Gaussian likelihoods.

## GauPro (probe: `R/probe/gaupro.R`)

Matches `ftc_standardize` and an `ftc_rbf_white` variant with `mean.constant = 1.5` to ~1e-15.

- Kernels: `Gaussian`, `Matern32`, `Matern52` with `s2` the variance and `theta = 10^beta`.
  `beta = -log10(2 l^2)` for `Gaussian`; `beta = -2 log10(l)` for the Matérns (Gram agreement 1e-16).
- Fixed parameters: `GauPro_kernel_model$new(..., param.est = FALSE, nug.est = FALSE, normalize = FALSE)`.
  The nugget is relative: `K = s2 (R + nug I)`, so `nug = noise / variance`; `nug.min` must be
  lowered (1e-12) for small noise.
- Mean: `trend_0$new()` for zero mean, `trend_c$new(m = c, m_est = FALSE)` for `mean.constant`.
- Likelihood: `model$loglikelihood()` is the full Gaussian value; add `-n log s` for standardise.
- Prediction: `predict(se.fit = TRUE)$s2` already includes the noise, unlike `DiceKriging` and `kergp`.
- Not yet checked: composite kernels (`*`, `+` on kernel objects), ARD, active dimensions, gradients.
