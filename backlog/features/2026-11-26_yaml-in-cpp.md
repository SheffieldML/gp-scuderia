---
id: "2026-11-26_yaml-in-cpp"
title: "Optional YAML reading in C++"
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

# Task: Optional YAML reading in C++

## Description

Read YAML directly in `CInterchange` behind an optional yaml-cpp build flag; keep convert-at-boundary as the default.

## Acceptance Criteria

- [x] Build flag off by default
- [x] Goldens load from YAML when enabled

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).

### 2026-11-26

Completed (GPc `cd9775e`). `make YAML=1 [YAMLROOT=prefix]` defines `GPC_YAML` and links yaml-cpp; `CJson::parseFile` dispatches on `.yaml`/`.yml`. Quoted scalars stay strings, plain scalars follow the YAML core schema, non-finite numbers are rejected. `testInterchange` checks that all nine YAML goldens parse to the same tree as their JSON twins and give the same log likelihood; in the default build it checks the refusal message. Homebrew's yaml-cpp (clang/libc++) does not link with g++-16; I built 0.8.0 with g++-16 into `~/.local/yaml-cpp-gcc`.
