---
id: "2026-11-27_gplvm-golden-and-benchmarks"
title: "bgplvm golden and the fixed-parameter benchmarks"
status: "Ready"
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

- [ ] Golden with GPy oracle for bound_without_kl, kl and psi statistics
- [ ] gplvm_collapse_to_sparse: S to 0 reproduces the dtcvar log likelihood at inputs mu
- [ ] Each benchmark meets the CIP-0005 admission rule and the latent-gauge rule

## Related

- CIP: 0006 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0006.
