---
id: "2026-11-26_gpy-to-wire"
title: "GPy to_wire and from_wire adapter"
status: "Ready"
priority: "High"
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

# Task: GPy to_wire and from_wire adapter

## Description

Make GPy, the oracle, read and write the format. Implement `to_wire`/`from_wire` for GPRegression and SparseGPRegression covering the kernels in the 0.1 registry, and round-trip both goldens.

## Acceptance Criteria

- [ ] `to_wire(model)` output validates against `schemas/model.schema.json`
- [ ] `from_wire` of each golden reproduces its `x-reference` within tolerance
- [ ] Unsupported GPy kernels raise rather than substitute
- [ ] Added to `scripts/conformance.sh`

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).
