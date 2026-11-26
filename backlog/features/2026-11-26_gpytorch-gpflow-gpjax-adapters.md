---
id: "2026-11-26_gpytorch-gpflow-gpjax-adapters"
title: "GPyTorch, GPflow and GPJax adapters"
status: "Proposed"
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

- [ ] Each adapter reproduces both goldens within tolerance
- [ ] Registry `verified` updated per entry

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).
