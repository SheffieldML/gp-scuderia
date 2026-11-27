---
id: "2026-11-27_gpy-jitter-feature-detection"
title: "GPy adapter detects the jitter options"
status: "Completed"
priority: "Medium"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0005"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cip0005
---

# Task: GPy adapter detects the jitter options

## Description

GPy gained `jitter` on `ExactGaussianInference` (PR 1332) and `const_jitter` on `VarDTC` and `FITC` (PR 1333). The adapter detects these options and uses `jitter=0` for exact inference and `const_jitter=1e-6` for VarDTC when they exist. The oracle features are recorded in each golden (`x-reference.oracle_features`). A `known:` entry written as `{why, fixed_by}` stops applying when the oracle lists that feature, so the known cases lift automatically on a new GPy and remain on the pinned one.

## Acceptance Criteria

- [x] `gpy_adapter.FEATURES` lists `exact_jitter` and `kuu_jitter` when available.
- [x] `benchmarks.py` unions `oracle_features` and skips `fixed_by` known entries.
- [x] Old GPy (1.13.2): all gpflow comparisons pass with the known cases applied.
- [x] New GPy (both options): all gpflow comparisons pass with the known cases lifted.
- [x] The duplicated-inputs gap is confirmed as the 1e-8 jitter and marked `fixed_by`.

## Related

- CIP: 0005
