---
id: "2026-11-26_approximation-tags"
title: "PITC and further approximation tags"
status: "Completed"
priority: "Low"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- interop
---

# Task: PITC and further approximation tags

## Description

Keep PITC and any further tags in the registry as unsupported until a second implementation exists, then add a golden.

## Acceptance Criteria

- [x] Registry records support per framework
- [x] Adapters raise on unsupported tags

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).

### 2026-11-26

Completed. GPy adapter now raises `UnsupportedByFramework` (was `NotImplementedError`) for dtc and pitc. New `scripts/check_unsupported_approximations.py` runs for all four Python adapters inside `conformance.sh`. Registry comment separates native classes from verified paths.
