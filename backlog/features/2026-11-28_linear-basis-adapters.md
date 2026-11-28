---
id: "2026-11-28_linear-basis-adapters"
title: "mean.linear_basis: framework adapters"
status: "Proposed"
priority: "Medium"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["0009"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_linear-basis-reference-goldens"]
tags:
- backlog
- cip0009
- adapters
---

# Task: mean.linear_basis adapters

## Description

GPy first (`fixed` natively; estimated modes through GLS computed from the model's `K_y`), then GPc and GPmat through bias and linear kernels where the basis allows, refusing otherwise. R adapters follow CIP-0007. Every other framework refuses.

## Acceptance Criteria

- [ ] GPy matches the reference in every mode
- [ ] GPc and GPmat match where realisable and refuse elsewhere
- [ ] No adapter maps a mode onto a constant bias unless the basis is the intercept alone
- [ ] `frameworks/*.yaml` declarations updated only for tested modes

## Related

- CIP: 0009, 0007 (gp-scuderia)

## Progress Updates

### 2026-11-28

Task created with CIP-0009 Accepted.
