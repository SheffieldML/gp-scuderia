---
id: "2026-11-28_linear-basis-adapters"
title: "mean.linear_basis: framework adapters"
status: "In Progress"
priority: "Medium"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["0009"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_linear-basis-reference-goldens"]
tags:
- backlog
- cip0009
- adapters
---

# Task: mean.linear_basis adapters

## Description

GPy first (`fixed` natively; estimated modes through GLS computed from the model's `K_y`), then GPc and GPmat through bias and linear kernels where the basis allows, refusing otherwise. R adapters follow CIP-0007. Every other framework refuses.

## Acceptance Criteria

- [x] GPy matches the reference in every mode
- [x] GPc and GPmat match (code and refusals in place; not executed, see progress) where realisable and refuse elsewhere
- [x] No adapter maps a mode onto a constant bias unless the basis is the intercept alone
- [ ] `frameworks/*.yaml` declarations updated only for tested modes

## Related

- CIP: 0009, 0007 (gp-scuderia)

## Progress Updates

### 2026-11-28

Task created with CIP-0009 Accepted.

### 2026-11-28

In Progress. GPy done: `fixed` through a fixed-weight `_BasisMean` mapping (round trips through `to_wire`); estimated modes through GPy's K, K* and noise with the GLS algebra of the NumPy reference, and `from_wire` refuses them because no GPy model realises them. GPy adds 1e-8 to the noise, so its log likelihood sits about 1e-7 from the exact reference; the golden tolerance for log likelihood is 1e-6. GPc, GPmat and R remain.

### 2026-11-28 (GPc and GPmat)
Added `python/gp_scuderia/kernel_form.py` (pure Python) and `octave/gpscuderia/gpscuderiaKernelForm.m`. A Gaussian prior on beta is the GP with mean F m0 and kernel F S0 F^T, so it is rewritten exactly as bias + linear kernels when the basis is an intercept plus linear columns covering every input, S0 is diagonal with equal linear variances and m0 is zero off the intercept. An intercept-only fixed beta becomes `mean.constant`. Profiled, flat_prior, correlated, nonlinear, partial-linear and unequal-variance cases refuse. Capability options (`beta.*`, `beyond_kernel_form`) let `frameworks/gpc.yaml` and `gpmat.yaml` declare exactly that; refusal snapshots updated. New goldens `ftc_linear_gaussian_prior_diag` and `ftc_linear_intercept_fixed`.

Verified: the rewrite, fed through GPy's own bias and linear kernels, reproduces both new x-references (`conformance_gpy.py`); seven refusal cases are tested in `check_linear_basis.py`; `compare.py` is clean for GPy, GPflow, GPJax and GPyTorch. Not verified: `gpc` is not installed and Octave fails to start here (Homebrew hdf5 missing), so neither GPc nor GPmat has run the new goldens. Registry entries stay unverified for them.

### 2026-11-28 (DiceKriging)
Added the linear-basis path to the merged `R/dicekriging/runDicekriging.R` (rebased onto CIP-0007) and `mean.linear_basis` to `frameworks/dicekriging.yaml`; `R/gpscuderia/gaussian.R` now refuses mean types a runner does not declare. Conventions were read from the 1.6.1 source before any comparison: `logLikFun` is the concentrated likelihood, `predict.km` variance is latent only, `gauss` is exp(-h^2/(2 theta^2)) so theta equals the wire lengthscale. `km()` with known `coef.cov`, `coef.var` and `noise.var` fits nothing, so the runner builds the log likelihood from km's own Cholesky factor and whitened residual (`logLik()` fails on an unfitted model). `profiled` is native, `fixed` uses `coef.trend` with SK prediction, `flat_prior` subtracts log|M'M|/2, wire var is sd^2 + noise. `gaussian_prior` refuses. Result: `compare.py dicekriging` matches all five supported linear goldens within 1e-8 of tolerance and refuses the rest as the snapshot says; `check_framework_files.py` and `validate.py` pass. No CI job yet (`ci.modes: []`) until CIP-0007 is accepted; `kergp` not done.
