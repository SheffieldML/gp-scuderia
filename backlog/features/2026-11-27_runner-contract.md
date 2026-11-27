---
id: "2026-11-27_runner-contract"
title: "Process-level JSON runner contract"
status: "Ready"
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

- [ ] All frameworks run via the contract with unchanged results.

## Related

- CIP: 0003 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0003.
