---
id: "2026-11-27_generated-ci-matrix"
title: "Generate the CI matrix from framework files"
status: "Ready"
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

- [ ] All current jobs reproduced from YAML; adding a framework needs no workflow edit.

## Related

- CIP: 0003 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0003.
