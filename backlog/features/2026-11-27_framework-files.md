---
id: "2026-11-27_framework-files"
title: "Framework YAML files and derived expectations"
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

# Task: Framework YAML files and derived expectations

## Description

Write frameworks/*.yaml for gpy, gpflow, gpytorch, gpjax, gpc, gpml and gpmat. Add the capability schema including optional per-name options (coarse by default, refinable). Derive golden features; compare.py reads the files instead of PACKAGES, OCTAVE_RUNNERS and framework_verification; remove the registry block.

## Acceptance Criteria

- [x] Report identical to today's cell for cell; deliberately wrong declarations fail in both directions; unknown options fail validation.

## Related

- CIP: 0003 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0003.

### 2026-11-27 (delivery)
Added `frameworks/*.yaml` (7), `scripts/frameworks.py` (loader, feature extraction, derived expectations, registry-vocabulary and option validation), `scripts/check_framework_files.py` with the snapshot `tests/expected_refusals.yaml`, and `gp_scuderia/gpc_adapter.py`. `compare.py` now reads the files and the registry's `framework_verification` block is replaced by `capability_options`. Old and new `compare.py` output is identical cell for cell for gpy, gpflow, gpytorch, gpjax, gpml and gpmat (gpc not run locally; CI covers it). The CI matrix is still hand-written; that is task `generated-ci-matrix`.
