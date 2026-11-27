---
id: "2026-11-27_framework-files"
title: "Framework YAML files and derived expectations"
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

# Task: Framework YAML files and derived expectations

## Description

Write frameworks/*.yaml for gpy, gpflow, gpytorch, gpjax, gpc, gpml and gpmat. Add the capability schema including optional per-name options (coarse by default, refinable). Derive golden features; compare.py reads the files instead of PACKAGES, OCTAVE_RUNNERS and framework_verification; remove the registry block.

## Acceptance Criteria

- [ ] Report identical to today's cell for cell; deliberately wrong declarations fail in both directions; unknown options fail validation.

## Related

- CIP: 0003 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0003.
