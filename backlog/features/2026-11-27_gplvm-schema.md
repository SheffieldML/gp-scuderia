---
id: "2026-11-27_gplvm-schema"
title: "GP-LVM schema, registry and fixtures"
status: "Completed"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0006"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cip0006
---

# Task: GP-LVM schema, registry and fixtures

## Description

Add the `latent` block, the reserved `models.gplvm` and `models.bayesian_gplvm` tags, and fixtures for each rule (CIP-0006 step 1).

## Acceptance Criteria

- [x] Schema accepts both latent types and rejects each violation with a fixture
- [x] Cross-field rules checked by `scripts/validate.py`
- [x] Existing documents still validate

## Related

- CIP: 0006 (gp-scuderia)

## Progress Updates

### 2026-11-27
Implemented on branch cip0006-gplvm-tier.
