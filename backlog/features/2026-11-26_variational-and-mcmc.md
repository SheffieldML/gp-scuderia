---
id: "2026-11-26_variational-and-mcmc"
title: "Variational and MCMC approximations in the wire format"
status: "Proposed"
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

# Task: Variational and MCMC approximations in the wire format

## Description

Implement the design in CIP-0001 ("Variational and MCMC approximations"): `inference: vi` with a `variational` block, and `inference: mcmc` with `prior` and `mcmc` blocks. GPc, GPy and GPmat refuse them with `UnsupportedByFramework`; GPflow, GPJax and GPyTorch adapters are the first to realise them.

## Acceptance Criteria

- [ ] Schema accepts `vi` and `mcmc` with their blocks, and rejects them for Gaussian likelihoods where the posterior is exact
- [ ] Golden `vi_bernoulli` (GPflow oracle) with `elbo`, predictive moments and the quadrature rule
- [ ] Golden `vi_gaussian_z_equals_x` whose ELBO equals the collapsed bound
- [ ] Golden `mcmc_hyperparameters` with Monte Carlo error tolerances from an independent chain
- [ ] Negative fixtures for each new rule
- [ ] Registry moves `reserved_inference` entries into `types` once verified
- [ ] GPc, GPy and GPmat adapters refuse the tags explicitly

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Design written into CIP-0001 and tags reserved in the registry. No schema change yet.
