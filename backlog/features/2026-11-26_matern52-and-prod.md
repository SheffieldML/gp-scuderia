---
id: "2026-11-26_matern52-and-prod"
title: "kern.matern52 and kern.prod"
status: "Ready"
priority: "High"
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

# Task: kern.matern52 and kern.prod

## Description

Add `kern.matern52` and `kern.prod` to the schema, registry, goldens (GPy x-reference) and negative fixtures. Implement in GPc (`CInterchange`, using the tensor kernel for products), GPmat and the GPc Python adapter.

## Acceptance Criteria

- [ ] Golden and negative fixture for each kernel
- [ ] GPy, GPmat and GPc match the x-reference
- [ ] Registry entries per framework updated

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).
