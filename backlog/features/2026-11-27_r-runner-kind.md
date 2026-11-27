---
id: "2026-11-27_r-runner-kind"
title: "Add kind: r to the runner contract"
status: "Completed"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0007"]
owner: "Neil Lawrence"
dependencies: ["2026-11-27_r-probe"]
tags:
- backlog
- r
---

# Task: Add kind: r to the runner contract

## Description

Add an R runner kind with a stub R runner exercised by `scripts/check_runner_contract.py`, and `R/gpscuderia/` helpers for reading and writing wire documents.

## Acceptance Criteria

- [x] Stub R runner passes the contract check
- [x] Install lists for pinned and latest R packages exist

## Related

- CIP: 0007 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0007.

### 2026-11-27
Completed. `kind: r` launches `Rscript` with the Octave environment variables. `R/gpscuderia/run.R` is the shared driver (read, refuse, error, timing, write; gradient is refused unless a runner supplies it). The stub passes crash isolation, error, refusal and intact-batch checks. `ci/pinned-r.txt`, `ci/latest-r.txt` and `R/install.R` install the packages. The contract check skips the R half when Rscript is absent, and the R job joins the CI matrix with the adapters.
