---
id: "2026-11-27_timing-benchmarks"
title: "Timing benchmarks"
status: "Completed"
priority: "Medium"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0003", "0005"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cip0003
---

# Task: Timing benchmarks

## Description

Add in-runner timing with warmup, the scaling benchmark, hardware recording and a report table. Timing never gates the build.

## Acceptance Criteria

- [x] In-runner timing (cold, median, IQR) in every Python and Octave runner; `timing` is optional so the contract stays /1.
- [x] Grid expansion with pr and scheduled tiers; three timing benchmarks (exact, sparse, Laplace).
- [x] Thread modes (`--threads 1|all`, best effort) and hardware recorded.
- [x] Report table with ratios to the fastest framework at the same thread setting; never gates the build.
- [x] `scripts/check_timing.py` in CI.
- [ ] Medians repeatable across two runs on a quiet machine (hosted runners are noisy; reported, not asserted).

## Related

- CIP: 0005 defines the benchmark files.

- CIP: 0003 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0003.

### 2026-11-27
Implemented in the cip0005-benchmarks branch. Observation: GPy sparse and Laplace take about 1 s with all threads against about 7 ms with one thread; probable BLAS oversubscription on small matrices, to be confirmed.
