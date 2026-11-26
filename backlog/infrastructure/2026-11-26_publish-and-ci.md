---
id: "2026-11-26_publish-and-ci"
title: "Publish gp-scuderia and add CI"
status: "Proposed"
priority: "High"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "infrastructure"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- interop
---

# Task: Publish gp-scuderia and add CI

## Description

Publish the repository (needs `gh auth login`), add CI running `scripts/conformance.sh` with tool locations taken from the environment (removing home-directory defaults), and pin the GPy environment so the oracle does not move silently.

## Acceptance Criteria

- [ ] Repository public
- [ ] CI green on the goldens
- [ ] No hard-coded home paths in scripts

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).
