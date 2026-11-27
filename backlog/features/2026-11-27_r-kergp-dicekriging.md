---
id: "2026-11-27_r-kergp-dicekriging"
title: "kergp and DiceKriging adapters and framework files"
status: "Completed"
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

- [x] Goldens match or are refused as declared, in both directions
- [x] Linear trends and unsupported smoothness are refused, never substituted
- [x] Accuracy and gradient benchmarks run where capabilities allow

## Related

- CIP: 0007 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0007.

### 2026-11-27
Completed. Adapters, framework files, snapshot entries, CI matrix and workflow steps, `scripts/check_r_refusals.py` and `docs/r-frameworks.md` are in. Findings: DiceKriging's Matern kernel is a product kernel and differs from the isotropic wire kernel for d > 1, so it is refused (new capability option `multi_input`, claimed by gpml, kergp, and DiceKriging's rbf); small-noise and duplicated-input mismatches are the GPy 1e-8 noise term and are recorded as known. Gradients are not declared; `kern.white` is refused rather than folded (decision open with the owner).
