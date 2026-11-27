---
id: "2026-11-27_runner-contract"
title: "Process-level JSON runner contract"
status: "Completed"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0003"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cip0003
---

# Task: Process-level JSON runner contract

## Description

Define task and result schemas, add gp_scuderia.run for python entries, move the Octave scripts onto the same task file, and make compare.py a client of the contract. Batch with per-task catch and isolated retry on crash.

## Acceptance Criteria

- [x] All frameworks run via the contract with unchanged results.

## Related

- CIP: 0003 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0003.

### 2026-11-27
Completed. `gp_scuderia.run` (Python) and the Octave scripts read `gp-scuderia-task/1` and write `gp-scuderia-result/1` (schemas under `schemas/`). `compare.py` is a client: it launches the runner as a subprocess, and on a process crash retries each task alone. Results for gpy, gpflow, gpytorch, gpjax, gpml and gpmat are identical cell for cell to before; `check_runner_contract.py` covers crash isolation with a stub. gpc still uses its `gpc_adapter` entry through the same Python runner.
