---
id: "2026-11-26_yaml-in-cpp"
title: "Optional YAML reading in C++"
status: "Proposed"
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

# Task: Optional YAML reading in C++

## Description

Read YAML directly in `CInterchange` behind an optional yaml-cpp build flag; keep convert-at-boundary as the default.

## Acceptance Criteria

- [ ] Build flag off by default
- [ ] Goldens load from YAML when enabled

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).
