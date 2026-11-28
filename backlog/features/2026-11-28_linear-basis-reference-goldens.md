---
id: "2026-11-28_linear-basis-reference-goldens"
title: "mean.linear_basis: NumPy reference, property checks and goldens"
status: "In Progress"
priority: "High"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["0009"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_linear-basis-schema"]
tags:
- backlog
- cip0009
- conformance
---

# Task: mean.linear_basis reference, property checks and goldens

## Description

A NumPy reference for all four `beta.mode` values, independent of any framework. Property checks for the relations stated in CIP-0009. One golden per mode, plus a quadratic case.

## Acceptance Criteria

- [x] Reference gives log likelihood and predictive moments for each mode
- [x] `gaussian_prior` with large `prior_cov` approaches `flat_prior` up to the dropped term
- [x] `profiled` and `flat_prior` predict identically and differ in log likelihood by `-0.5 log|A|`
- [x] A deliberately wrong reference fails the checks
- [ ] Goldens added, with `tests/expected_refusals.yaml` updated

## Related

- CIP: 0009 (gp-scuderia)

## Progress Updates

### 2026-11-28

Task created with CIP-0009 Accepted.

### 2026-11-28

In Progress. `python/gp_scuderia/linear_basis.py` and `scripts/check_linear_basis.py` done: ten property checks, each against a different computation than the reference, and a broken reference is caught. Goldens and `expected_refusals.yaml` remain.
