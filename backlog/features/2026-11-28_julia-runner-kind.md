---
id: "2026-11-28_julia-runner-kind"
title: "Add kind: julia to the runner contract"
status: "Ready"
priority: "High"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["000A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_julia-probe"]
tags:
- backlog
- julia
---

# Task: Add kind: julia to the runner contract

## Description

Add a Julia runner kind with a stub Julia runner exercised by `scripts/check_runner_contract.py`, and a `julia/GPScuderia/` helper for reading and writing wire documents. Add pinned and latest install lists, with a `Manifest.toml` for the pinned mode. The runner records `JULIA_NUM_THREADS` and BLAS thread count, and reports the first call as cold time.

## Acceptance Criteria

- [ ] Stub Julia runner passes the contract check
- [ ] `ci/pinned-julia.txt` and `ci/latest-julia.txt` (or the Manifest equivalent) exist
- [ ] Contract documentation in `docs/` names `kind: julia`
- [ ] Thread counts and dtype recorded in runner output

## Related

- CIP: 000A (gp-scuderia)
- Model: `2026-11-27_r-runner-kind`

## Progress Updates

### 2026-11-28
Created from the accepted CIP-000A.
