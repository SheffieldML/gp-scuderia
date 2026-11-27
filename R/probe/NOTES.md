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

## kergp, GauPro

Not yet probed.
