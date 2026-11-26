---
id: "2026-11-26_non-gaussian-likelihoods"
title: "Non-Gaussian likelihoods (Bernoulli, Poisson) with Laplace inference"
status: "Completed"
priority: "Medium"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- interop
---

# Task: Non-Gaussian likelihoods and inference tags

## Description

Scope agreed with the owner on 2026-11-26: Bernoulli (probit link) and Poisson (log link) with `inference: laplace`. Student-t, EP and sparse non-Gaussian models are out of scope for 0.1 (`ep` is reserved, not defined). An adapter that cannot realise the combination raises.

## Acceptance Criteria

- [x] Schema, registry and goldens (`laplace_bernoulli`, `laplace_poisson`)
- [x] Reference values come from an independent implementation (`scripts/check_laplace_independent.py`, numpy, Rasmussen and Williams Alg 3.1/3.2)

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).

### 2026-11-26 (completed)
- GPy is the oracle for the goldens and agrees with the independent numpy Laplace to about 1e-8 (log likelihood to 7e-10).
- GPy's own predictive variance of y is nan for Bernoulli and negative for Poisson, so `mean`/`var` are derived from the latent Laplace moments (`latent_mean`, `latent_var`) in closed form.
- Correction: GPc does implement Laplace (CIP-0033, CIP-003A: `CLaplaceInference`, `CProbitNoise`, `CPoissonNoise`). `CInterchange` now reads and runs `models.gp` with these likelihoods; GPc matches the goldens including the latent moments.
- GPc's Laplace defaults mirror GPy (30 Newton iterations, objective tolerance 1e-4, jitter 1e-6), which leaves about 1e-6 error against the converged answer. `CInterchange` solves to 1e-13 with no jitter so a document has one meaning (`CGp::setLaplaceSolver`).
- GPmat, GPyTorch, GPflow and GPJax refuse these models: none implements Laplace, and GPflow/GPJax offer different approximations that must not be substituted silently. Refusals are tested.
- GPc export path done (GPc `9913b63`): `gp export` writes Laplace models, and the `.model` stream records `laplaceInference`. Tested by `testInterchange` and `scripts/conformance_gpc_cli.sh`.
- Not done: EP (GPc has it), Student-t (GPc has it). Variational and MCMC are tracked in `2026-11-26_variational-and-mcmc`.
