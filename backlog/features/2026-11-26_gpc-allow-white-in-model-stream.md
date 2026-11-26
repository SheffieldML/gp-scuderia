---
id: "2026-11-26_gpc-allow-white-in-model-stream"
title: "Record the allow-white opt-in in the GPc .model stream"
status: "Proposed"
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

- [ ] GPc CIP amendment agreed
- [ ] Old `.model` files still load
- [ ] `gp import`/`gp export` handle exact+White

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).
