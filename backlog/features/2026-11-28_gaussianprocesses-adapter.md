---
id: "2026-11-28_gaussianprocesses-adapter"
title: "GaussianProcesses.jl adapter and framework file (phase 2)"
status: "Completed"
priority: "Low"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["000A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_juliagps-adapter"]
tags:
- backlog
- julia
---

# Task: GaussianProcesses.jl adapter and framework file (phase 2)

## Description

Add the second Julia adapter, reusing `julia/GPScuderia/`. `GaussianProcesses.jl` has its own kernel hierarchy and stores log parameters, so the conversions differ from `juliagps`. Add `frameworks/gaussianprocesses.yaml` and the registry entries.

## Acceptance Criteria

- [x] Declared goldens match the oracle within tolerance
- [x] Unsupported models are refused with a reason
- [x] Log-parameter conversions documented in `docs/`
- [x] Framework file and registry updated

## Related

- CIP: 000A (gp-scuderia)
- Model: `2026-11-27_r-gaupro`

## Progress Updates

### 2026-11-28
Created from the accepted CIP-000A.

### 2026-11-28
Adapter and framework file done; goldens match, refusals verified. kern.white refused (no white kernel in the package).
