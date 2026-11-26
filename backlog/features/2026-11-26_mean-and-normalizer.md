---
id: "2026-11-26_mean-and-normalizer"
title: "Optional mean and normalizer fields"
status: "Completed"
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

- [x] Schema and `docs/overview.md` updated
- [x] Golden generated with GPy's normalizer
- [x] GPc maps `normalizer.center` onto the stored bias and drops the export refusal for centred models; scaling checked against GPc's scale handling before it is claimed
- [x] GPmat maps `bias`/`scale` likewise
- [x] Adapters without support raise

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).

### 2026-11-26

Completed. Goldens `ftc_standardize` and `dtcvar_mean_constant` (GPy x-reference); schema, overview and registry updated; GPc, GPmat and GPy adapters agree.

Decisions beyond the CIP text:
- `mean` and `normalizer` are mutually exclusive in 0.1 (the order of a mean function and a normalizer is unspecified). Schema, GPc, GPmat and GPy all reject both.
- GPc's `-0.5 * 2 log|scale|` objective term (only when the scale is learnt) is one log scale per output, not per datum, so it is not the density of y. `CInterchange` keeps the learnt flag off and adds `-N log s` itself. Export refuses a model whose output scale is learnt.
- `mean.constant` imports to GPc bias, so a `.model` export of it reads back as `normalizer.center`; log likelihood and predictions are identical.
- The GPy adapter uses a `Standardize` subclass with the wire mean and std fixed, and reports `log_likelihood() - n log std`.
