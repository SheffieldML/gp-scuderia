---
id: "2026-11-27_accuracy-benchmarks"
title: "Accuracy benchmark files"
status: "Ready"
priority: "Medium"
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

# Task: Accuracy benchmark files

## Description

Add benchmarks/*.yaml, the seeded generator writing wire models, and benchmarks for gradient agreement, conditioning and lengthscale extremes, with report sections. Frameworks lacking required capabilities are reported as skipped.

## Acceptance Criteria

- [ ] Benchmarks run across frameworks; failures gate the build; skips are reported.

## Related

- CIP: 0003 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0003.
