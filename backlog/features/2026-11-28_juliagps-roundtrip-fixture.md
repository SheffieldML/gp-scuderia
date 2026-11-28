---
id: "2026-11-28_juliagps-roundtrip-fixture"
title: "Round-trip fixture between GPy and juliagps"
status: "Completed"
priority: "Medium"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["000A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_juliagps-adapter"]
tags:
- backlog
- julia
- round-trip
---

# Task: Round-trip fixture between GPy and juliagps

## Description

Add a fixture for `models.gp_regression` with an RBF kernel and a Gaussian likelihood, written by GPy and read by `juliagps`, and the reverse. Until it passes, the `juliagps` entry stays provisional.

## Acceptance Criteria

- [x] GPy to juliagps direction passes
- [x] juliagps to GPy direction passes
- [x] Provisional marker on `juliagps` removed, or a recorded reason for keeping it (no marker was ever written into `frameworks/juliagps.yaml`, so there is nothing to remove)

## Related

- CIP: 000A (gp-scuderia)

## Progress Updates

### 2026-11-28
Created from the accepted CIP-000A.

### 2026-11-28
Completed. `scripts/roundtrip_juliagps.py` (run with the GPy interpreter) covers both directions for an RBF Gaussian regression; `julia/juliagps/exportModel.jl` is the Julia-side writer. Log likelihood, mean and variance agree to within 1e-6 (observed differences about 1e-7). It is not in CI because it needs GPy and Julia together; run it by hand.
