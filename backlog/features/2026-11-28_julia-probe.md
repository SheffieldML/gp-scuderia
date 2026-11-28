---
id: "2026-11-28_julia-probe"
title: "Probe AbstractGPs, KernelFunctions and GaussianProcesses against the goldens"
status: "Completed"
priority: "High"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["000A"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- julia
---

# Task: Probe AbstractGPs, KernelFunctions and GaussianProcesses against the goldens

## Description

Install `AbstractGPs.jl`, `KernelFunctions.jl` and `GaussianProcesses.jl`, build each golden by hand in Julia, and record match or refusal with reasons. Record parameterisations (lengthscale, variance, ARD, noise), likelihood and prediction conventions, jitter and defaults, and the JSON and YAML packages that read the wire format with the fewest dependencies.

## Acceptance Criteria

- [x] Probe note written under `julia/probe/NOTES.md`, with Julia and package versions recorded
- [x] Each golden reproduced at fixed parameters, or refused with a reason
- [x] Small-noise behaviour and jitter recorded
- [x] Capability lists and the survey in CIP-000A corrected from the probe
- [x] JSON and YAML package choice recorded

## Related

- CIP: 000A (gp-scuderia)
- Model: `2026-11-27_r-probe`

## Progress Updates

### 2026-11-28
Created from the accepted CIP-000A.

### 2026-11-28
Probe done; findings in julia/probe/NOTES.md.
