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

- [x] Repository public
- [ ] CI green on the goldens
- [x] No hard-coded home paths in scripts

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).

### 2026-11-26

Repository was already public with one commit; pushed main (`b1a8f5b`). Added `.github/workflows/conformance.yml` (jobs `oracle` and `gpc`), `requirements-gpy.txt` (pinned oracle), and made `scripts/conformance.sh` take tool locations from `GPC_ROOT`, `PYTHON`, `GPY_PY`, `GPMAT` and skip sections whose tool is unset. Untracked committed `.pyc` files. CI is not yet confirmed green: the `gpc` job checks out the GPc branch `cip0016-interchange-adapter`, which exists only locally, so that job fails until the branch is pushed or merged.
