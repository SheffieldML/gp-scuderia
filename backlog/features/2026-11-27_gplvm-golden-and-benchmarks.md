---
id: "2026-11-27_gplvm-golden-and-benchmarks"
title: "bgplvm golden and the fixed-parameter benchmarks"
status: "In Progress"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0006"]
owner: "Neil Lawrence"
dependencies: ["2026-11-27_gplvm-runner-ops"]
tags:
- backlog
- cip0006
---

# Task: bgplvm golden and the fixed-parameter benchmarks

## Description

Golden `bgplvm_rbf_ard` with GPy reference values, then benchmarks gplvm_bound_agreement, gplvm_collapse_to_sparse, gplvm_gradient_agreement, psi_statistics and gplvm_timing, in that order (CIP-0006 steps 3 and 4).

## Acceptance Criteria

- [x] Golden (examples/latent/bgplvm_rbf_ard) with GPy oracle for bound_without_kl, kl and psi statistics
- [x] gplvm_bound_agreement, gplvm_collapse_to_sparse and psi_statistics added (GPy passes all 13 cases)
- [x] gplvm_collapse_to_sparse: S to 0 reproduces the dtcvar log likelihood at inputs mu
- [x] gplvm_gradient_agreement (GPy passes all 5 cases)
- [ ] gplvm_timing (scheduled tier)
- [ ] Each benchmark meets the CIP-0005 admission rule and the latent-gauge rule

## Related

- CIP: 0006 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0006.

### 2026-11-27
In Progress. Golden added under `examples/latent/` with GPy values; `scripts/check_latent_ops.py` verifies it. Benchmarks remain.

### 2026-11-27 (later)
Added `gplvm_bound_agreement`, `gplvm_collapse_to_sparse` and `psi_statistics`. The psi oracle is a 48-node Gauss-Hermite quadrature; 24 nodes left 1e-7 in psi2, which the generator's GPy cross-check caught. Frameworks without a `bound` or `psi` operation in their framework file are skipped. Gradient and timing remain.

### 2026-11-27 (gradient)
Added `gplvm_gradient_agreement`. The gradient is that of `bound_without_kl - kl` in wire units, with `latent.mean`, `latent.variance` and `inducing_inputs` added to `params.py`. GPy leaves updates off after constructing a `BayesianGPLVM`, so its gradients (not its values) were stale until `update_model(True)`; the adapter now refreshes them. GPy returns about 4e-6 for the inducing-input gradient of a linear kernel, where the exact value is zero, so the case relies on an absolute tolerance of 1e-4. Only `gplvm_timing` remains.
