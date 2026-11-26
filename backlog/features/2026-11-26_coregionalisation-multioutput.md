---
id: "2026-11-26_coregionalisation-multioutput"
title: "Coregionalisation and multi-output models"
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

# Task: Coregionalisation and multi-output models

## Description

Add `kern.coregionalize` and `output_dim > 1` to the schema, matching GPc CIP-0038, with a GPy-generated golden.

## Acceptance Criteria

- [x] Schema, registry, golden and negatives
- [x] GPy and GPc agree within tolerance; GPmat rejects with a clear error (scope B)

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).

### 2026-11-26 (scope B, completed)

Scope: ICM and LCM for GPy and GPc; GPmat rejects. Delivered a stacked layout (index column in `data.X`, `output_dim` 1) rather than `output_dim > 1`, because GPy and GPc both model multi-output that way. Added `active_dims` support to the GPc reader, `kern.coregionalize`, and `likelihoods.mixed_gaussian` (GPc realises it as an index-white kernel plus a frozen 1e-6 likelihood variance, subtracted on import and added back on export). Goldens `ftc_icm` and `ftc_lcm` agree with GPy to 6e-8 in log likelihood. GPc `.model` streams now carry an optional `activeDims=` line so ICM models survive a save and reload. The Kronecker form (`output_dim > 1`) is left for a later task.
