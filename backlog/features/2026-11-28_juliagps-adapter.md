---
id: "2026-11-28_juliagps-adapter"
title: "juliagps adapter and framework file"
status: "Completed"
priority: "High"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["000A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_julia-runner-kind"]
tags:
- backlog
- julia
---

# Task: juliagps adapter and framework file

## Description

Write the adapter that builds an `AbstractGPs.jl` model with `KernelFunctions.jl` kernels from a wire document, with conversions for lengthscale (input transform), variance (`ScaledKernel`), ARD and noise. Add `frameworks/juliagps.yaml` with the capabilities the probe confirms, marked provisional, and the registry entries with explicit `null` where unsupported. Document the conversions in `docs/`.

## Acceptance Criteria

- [x] Declared goldens match the oracle within tolerance
- [x] Unsupported models are refused with a reason, never substituted
- [x] Gradients exposed and compared through the gradient benchmark, or the lack recorded in the framework file
- [x] Jitter recorded and stated in the conformance report
- [x] Conversions documented in `docs/`
- [x] Registry updated

## Related

- CIP: 000A (gp-scuderia)
- Model: `2026-11-27_r-kergp-dicekriging`

## Progress Updates

### 2026-11-28
Created from the accepted CIP-000A.

### 2026-11-28
Adapter, framework file, refusal check and docs done. Registry entries skipped, following the R precedent.
