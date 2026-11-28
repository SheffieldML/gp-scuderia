---
id: "2026-11-28_linear-basis-schema"
title: "mean.linear_basis: schema, overview and registry"
status: "Completed"
priority: "High"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["0009"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cip0009
- schema
---

# Task: mean.linear_basis schema, overview and registry

## Description

Add `mean.linear_basis` with `basis.polynomial` and the four `beta.mode` values to the model schema, `docs/overview.md` and `registry/types.yaml`. Exact GP only in 0.1.

## Acceptance Criteria

- [x] Schema accepts the four modes and rejects missing `value`, `prior_mean` or `prior_cov` where required
- [x] Schema rejects the type with a sparse approximation
- [x] Invalid fixtures cover each rejection
- [x] `scripts/validate.py` passes
- [x] Overview and registry updated

## Related

- CIP: 0009 (gp-scuderia)

## Progress Updates

### 2026-11-28

Task created with CIP-0009 Accepted.

### 2026-11-28

Completed. Schema, overview, registry and six invalid fixtures; each fixture differs from a valid counterpart only by its one defect, and all four modes validate. `validate.py` and `check_framework_files.py` pass.
