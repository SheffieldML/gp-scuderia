---
id: "2026-11-26_mean-and-normalizer"
title: "Optional mean and normalizer fields"
status: "Ready"
priority: "High"
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

# Task: Optional mean and normalizer fields

## Description

Implement the CIP-0001 decision: optional `mean` (`mean.constant`) and `normalizer` (`normalizer.center`, `normalizer.standardize`). Log likelihood and predictions are reported on the original scale with the `-n log s` Jacobian term. Absent fields keep the zero-mean, unscaled meaning.

## Acceptance Criteria

- [ ] Schema and `docs/overview.md` updated
- [ ] Golden generated with GPy's normalizer
- [ ] GPc maps `normalizer.center` onto the stored bias and drops the export refusal for centred models; scaling checked against GPc's scale handling before it is claimed
- [ ] GPmat maps `bias`/`scale` likewise
- [ ] Adapters without support raise

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).
