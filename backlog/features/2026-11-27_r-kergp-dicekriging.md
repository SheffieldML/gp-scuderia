---
id: "2026-11-27_r-kergp-dicekriging"
title: "kergp and DiceKriging adapters and framework files"
status: "Ready"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0007"]
owner: "Neil Lawrence"
dependencies: ["2026-11-27_r-runner-kind"]
tags:
- backlog
- r
---

# Task: kergp and DiceKriging adapters and framework files

## Description

Adapters, `frameworks/kergp.yaml`, `frameworks/dicekriging.yaml`, registry entries, parameter-conversion notes in `docs/`, and an R job in the generated CI matrix.

## Acceptance Criteria

- [ ] Goldens match or are refused as declared, in both directions
- [ ] Linear trends and unsupported smoothness are refused, never substituted
- [ ] Accuracy and gradient benchmarks run where capabilities allow

## Related

- CIP: 0007 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0007.
