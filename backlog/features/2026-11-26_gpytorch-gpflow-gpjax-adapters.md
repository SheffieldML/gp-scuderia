---
id: "2026-11-26_gpytorch-gpflow-gpjax-adapters"
title: "GPyTorch, GPflow and GPJax adapters"
status: "Completed"
priority: "Medium"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: ["2026-11-26_gpy-to-wire"]
tags:
- backlog
- interop
---

# Task: GPyTorch, GPflow and GPJax adapters

## Description

Write one adapter per framework and verify against the goldens before setting `verified: true` in `registry/types.yaml`. Raise on kernels or noise structures the framework cannot realise (for example `kern.white` in GPyTorch).

## Acceptance Criteria

- [x] Each adapter reproduces every golden it can realise within tolerance, and refuses the rest
- [x] Registry records per-framework verification (`framework_verification`)

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).

## Progress Updates

### 2026-11-26
Completed. `scripts/check_adapter.py <fw>` checks each adapter against every golden.
- GPflow: reproduces ftc and dtcvar goldens; refuses fitc and coregionalisation.
- GPyTorch: reproduces `ftc_prod_matern52` and `ftc_standardize`; refuses `kern.white`, sparse approximations and coregionalisation.
- GPJax: as GPflow. Needs jitter 0 for ftc (1e-6 breaks the 1e-7 mean tolerance) and 1e-6 for the sparse bound.
