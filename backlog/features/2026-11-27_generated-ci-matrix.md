---
id: "2026-11-27_generated-ci-matrix"
title: "Generate the CI matrix from framework files"
status: "Completed"
priority: "High"
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

# Task: Generate the CI matrix from framework files

## Description

A first job reads frameworks/*.yaml and emits the job matrix for fromJSON, replacing per-framework jobs including pinned, latest and Octave fetch handling; update conformance.sh to loop over the files.

## Acceptance Criteria

- [x] All current jobs reproduced from YAML; adding a framework needs no workflow edit.

## Related

- CIP: 0003 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0003.

## Progress Updates

### 2026-11-27
Completed. `scripts/ci_matrix.py` emits the matrix from `frameworks/*.yaml`; one `frameworks` job covers Python and Octave runners, pinned always and latest on schedule. `scripts/ci_fetch.py` replaces the inline GPML and GPmat fetch steps. gpc keeps its bespoke job (`ci.job: gpc`) and the oracle job keeps pinned GPy. `conformance.sh` loops over the files. Octave report columns are now labelled `pinned` rather than `local`. Not yet exercised on GitHub.
