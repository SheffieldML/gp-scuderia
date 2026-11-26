---
id: "2026-11-26_gpc-allow-white-in-model-stream"
title: "Record the allow-white opt-in in the GPc .model stream"
status: "Completed"
priority: "Medium"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: ["2026-11-26_gpy-to-wire"]
tags:
- backlog
- interop
---

# Task: Record the allow-white opt-in in the GPc .model stream

## Description

The `.model` stream does not store the CIP-0027 allow-white flag, so exact GPs with a White kernel cannot round-trip. Raise a GPc CIP-0027 amendment to persist the flag (versioned field), then lift the import/export refusals.

## Acceptance Criteria

- [x] GPc CIP amendment agreed
- [x] Old `.model` files still load
- [x] `gp import`/`gp export` handle exact+White

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).

### 2026-11-26

Completed. Additive optional stream key `allowWhiteAndLikelihoodNoise` (written only when set; old streams load as false). Rather than a separate GPc CIP-0027 amendment, the change is recorded in GPc CIP-0016 because it does not alter the CIP-0027 default. `gp import`/`gp export` handle exact+White; White without the flag, and sparse+White, are still refused on export.
