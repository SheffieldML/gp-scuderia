---
id: "2026-11-26_matern52-and-prod"
title: "kern.matern52 and kern.prod"
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

# Task: kern.matern52 and kern.prod

## Description

Add `kern.matern52` and `kern.prod` to the schema, registry, goldens (GPy x-reference) and negative fixtures. Implement in GPc (`CInterchange`, using the tensor kernel for products), GPmat and the GPc Python adapter.

## Acceptance Criteria

- [x] Golden and negative fixture for each kernel
- [x] GPy, GPmat and GPc match the x-reference
- [x] Registry entries per framework updated

## Related

- CIP: 0001 (gp-scuderia)

## Progress Updates

### 2026-11-26

Task created from CIP-0001 (Accepted).

### 2026-11-26

Completed. Goldens `ftc_prod_matern52` and `fitc_matern52` (GPy x-reference); GPc
(`CInterchange` via `CTensorKern`), GPmat (`tensor`) and GPy adapters match. Negatives:
prod without parts, white inside prod, ARD matern52 (GPc C++); unknown kernel (GPc python, GPmat).
The FITC golden uses mean tolerance 1e-6 (GPc whitened path differs ~3e-7).

Bugs found and fixed on the way:
- GPc `CGp::updateAlpha` overwrote the member `scaledM` for whitened FITC, so predict changed the
  objective (regression check added to `testInterchange`).
- GPmat `gpLogLikelihood` FITC counted the 2*pi constant twice.
