---
id: "2026-11-27_gradient-agreement"
title: "Contract version 2 and gradient agreement"
status: "Completed"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0005"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cip0005
---

# Task: Contract version 2 and gradient agreement

## Description

Add operations to the task schema, a gradient operation to every adapter and Octave runner, the oracle finite-difference cross-check and the gradient_agreement benchmark.

## Acceptance Criteria

- [x] A deliberately wrong gradient conversion in a test adapter fails the benchmark.
- [x] Version 1 task and result files still validate.

## Related

- CIP: 0005 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0005.

### 2026-11-27
Implemented on branch cip0005-gradients: contract /2, gradient in the GPy, GPflow, GPyTorch, GPJax, GPML and GPmat runners, `benchmarks/gradient_agreement.yaml`, `scripts/check_gradient.py` (wrong chain rule rejected, /1 files still validate).
