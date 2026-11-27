---
id: "2026-11-27_gplvm-runner-ops"
title: "Runner bound and psi operations"
status: "Ready"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0006"]
owner: "Neil Lawrence"
dependencies: ["2026-11-27_gplvm-schema"]
tags:
- backlog
- cip0006
---

# Task: Runner bound and psi operations

## Description

Add `bound` (with `kl` reported separately) and optional `psi` operations to the runner contract; implement in the GPc and GPy adapters (CIP-0006 step 2).

## Acceptance Criteria

- [ ] Contract version bump; earlier task and result files still validate
- [ ] GPc and GPy adapters return bound_without_kl, kl and psi0/1/2
- [ ] Frameworks without the operation are reported as skipped

## Related

- CIP: 0006 (gp-scuderia)

## Progress Updates

### 2026-11-27
Created from the accepted CIP-0006.
