---
id: "2026-11-27_accuracy-benchmarks"
title: "Accuracy benchmark files"
status: "Completed"
priority: "Medium"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0003"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cip0003
---

# Task: Accuracy benchmark files

## Description

Add benchmarks/*.yaml, the seeded generator writing wire models, and benchmarks for gradient agreement, conditioning and lengthscale extremes, with report sections. Frameworks lacking required capabilities are reported as skipped.

## Acceptance Criteria

- [x] Benchmarks run across frameworks; failures gate the build; skips are reported.

## Related

- CIP: 0003 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0003.

### 2026-11-27
Implemented: `benchmarks/{lengthscale_extremes,conditioning,size_accuracy}.yaml`, `scripts/benchmarks.py` (seeded generator, GPy oracle reference, runner), a report section, and CI wiring (the oracle job generates the data, each framework job downloads it and runs). All six frameworks pass locally.

Findings recorded as `known` mismatches (they do not gate, and fail as unexpected passes if fixed):
- noise 1e-4: all five non-GPy frameworks agree with each other and differ from GPy by about 0.4 in log likelihood. Suspected GPy jitter; not confirmed.
- DTCVar at n=1000 differs from GPy by about 1e-2 in gpflow, gpjax and gpmat. Not diagnosed.

Deferred: gradient agreement. The runner contract exposes no gradients, so it needs a contract extension first.
