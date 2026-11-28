---
id: "2026-11-28_juliagps-roundtrip-fixture"
title: "Round-trip fixture between GPy and juliagps"
status: "Ready"
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

- [ ] GPy to juliagps direction passes
- [ ] juliagps to GPy direction passes
- [ ] Provisional marker on `juliagps` removed, or a recorded reason for keeping it

## Related

- CIP: 000A (gp-scuderia)

## Progress Updates

### 2026-11-28
Created from the accepted CIP-000A.
