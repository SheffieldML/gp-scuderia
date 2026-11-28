---
id: "2026-11-28_julia-ci-matrix"
title: "Julia job in the generated CI matrix"
status: "Ready"
priority: "Medium"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "infrastructure"
related_cips: ["000A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_juliagps-adapter"]
tags:
- backlog
- julia
- ci
---

# Task: Julia job in the generated CI matrix

## Description

Add a Julia job to the generated CI matrix using `julia-actions/setup-julia`, with pinned mode on pull requests and latest mode on the weekly schedule. Include the unsupported-model refusal check.

## Acceptance Criteria

- [ ] Generated matrix includes the Julia framework in pinned and latest modes
- [ ] Refusal check runs for Julia
- [ ] Workflow passes on a pull request

## Related

- CIP: 000A (gp-scuderia)

## Progress Updates

### 2026-11-28
Created from the accepted CIP-000A.
