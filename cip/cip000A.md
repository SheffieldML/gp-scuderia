---
author: "Neil Lawrence"
created: "2026-11-28"
id: "000A"
last_updated: "2026-11-28"
status: "Accepted"
compressed: false
related_requirements: []
related_cips: ["0003", "0004", "0005", "0007"]
tags:
- cip
- julia
- frameworks
- adapters
title: "Julia frameworks: AbstractGPs and KernelFunctions, then GaussianProcesses"
---

# CIP-000A: Julia frameworks: AbstractGPs and KernelFunctions, then GaussianProcesses

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted
- [ ] In Progress
- [ ] Implemented
- [ ] Closed

## Summary

Add a Julia runner kind to gp-scuderia and use it to bring in Julia Gaussian process packages as frameworks. Phase one covers the JuliaGPs stack (`AbstractGPs.jl` with `KernelFunctions.jl`) as a single framework, `juliagps`. Phase two adds `GaussianProcesses.jl` as a second, independent implementation. `ApproximateGPs.jl`, `GPLikelihoods.jl`, `Stheno.jl` and `TemporalGPs.jl` are recorded as candidates and deferred until the wire format can name what they do.

The route follows CIP-0007: a new `kind` in the runner contract, a thin shared helper that reads a wire document and builds the native model, and one adapter per package. The harness does not change.

## Motivation

- **Julia has an active GP community that the comparison does not reach.** The JuliaGPs organisation maintains a shared interface for GPs, and several research groups use it. A shared interchange format with no Julia target leaves that community out.
- **Julia is a third language family for conformance.** Python, Octave and C++ are covered, and CIP-0007 adds R. A Julia implementation that matches the GPy oracle is further evidence that the wire format is not shaped by one language or one library.
- **The Julia kernel layer differs structurally from the others.** In `KernelFunctions.jl` a lengthscale is an input transform applied to a base kernel, and a variance is a `ScaledKernel` wrapper. Neither is a native parameter of the kernel, as in GPy or GPflow. This is the kind of convention gap the adapters exist to expose.
- **The machinery is ready.** CIP-0003 made a framework a YAML file plus a runner that speaks JSON, and CIP-0007 showed that a new language needs a runner kind and an adapter and no change to the harness.

This follows the interoperability tenet: speak the formats the community already uses, and do not require a new framework to change the harness.

## Detailed Description

### Candidate survey

The survey is from the packages' documentation as we remember it. We have not installed them for this CIP, and the claims about coverage below are unverified. Phase 0 exists to check every one before anything is committed to a framework file.

| Package | Model | Kernels | Decision |
|---|---|---|---|
| `AbstractGPs.jl` with `KernelFunctions.jl` | Exact GP behind a common interface (`GP`, `posterior`, `logpdf`, `marginals`); the kernel library is shared with most of the JuliaGPs packages | Squared exponential, Matérn, linear, constant, white-like, sums and products, input transforms for lengthscale | Phase one, as one framework. The widest kernel coverage and the stable interface. |
| `GaussianProcesses.jl` | Exact GP with its own kernel hierarchy and stateful model objects | Its own types, including Matérn, ARD variants, sums and products | Phase two. Independent of `KernelFunctions.jl`, and closest to GPy in style. |
| `ApproximateGPs.jl` | Variational and Laplace approximations on top of `AbstractGPs.jl` | Via `KernelFunctions.jl` | Deferred. Needs the approximation and `vi` vocabulary to be settled; we believe it offers a VFE or DTC-style collapsed bound and SVGP, and not FITC or PITC. |
| `GPLikelihoods.jl` | Non-Gaussian likelihoods | | Deferred with `ApproximateGPs.jl`, because non-Gaussian goldens require an approximate inference route. |
| `Stheno.jl` | Older composable GP package, largely superseded by `AbstractGPs.jl` | Own types | Deferred. Revisit only if a user needs it. |
| `TemporalGPs.jl` | State-space GPs | | Deferred. Kalman-form kernels have no wire tag. |

Wrappers that call a Python framework from Julia (for example through `PythonCall`) are excluded, because they would re-test frameworks already covered.

### One framework or two for the JuliaGPs stack

`AbstractGPs.jl` and `KernelFunctions.jl` are layers of one stack, and no user chooses one without the other. We record them as a single framework, `juliagps`, whose `versions` list names both. This matches how `gpytorch` is treated, and it avoids a framework entry that cannot run alone. `ApproximateGPs.jl` and `GPLikelihoods.jl`, when they arrive, extend `juliagps` through its capabilities, unless the probe shows that their behaviour is different enough to warrant a separate entry.

### What the wire format can express

Expected capabilities, subject to phase 0:

```yaml
# frameworks/juliagps.yaml (expected)
capabilities:
  approximations: [ftc]
  kernels:
    - {rbf: {options: [ard]}}
    - {matern32: {options: [ard]}}
    - {matern52: {options: [ard]}}
    - white
    - bias
    - linear
    - add
    - prod
  likelihoods: [gaussian]
  inference: [exact]
  extras: []                 # mean and normalizer handled in the adapter only if the probe shows it is faithful

# frameworks/gaussianprocesses.yaml (expected)
capabilities:
  approximations: [ftc]
  kernels:
    - {rbf: {options: [ard]}}
    - {matern32: {options: [ard]}}
    - {matern52: {options: [ard]}}
    - white
    - add
    - prod
  likelihoods: [gaussian]
  inference: [exact]
  extras: [mean.constant]
```

Declared capabilities decide which goldens each runner must match and which it must refuse. Where a package cannot express a wire model, the adapter refuses with a reason and never substitutes.

### Parameterisation hazards

Each of these has to be settled in phase 0 and written into `docs/`, because a mismatch produces silent wrong answers, not errors.

- **Lengthscale convention.** `KernelFunctions.jl` applies a lengthscale through a transform on the input, and the unit convention (multiply or divide, factor of 2 in the exponent) has to be matched to the wire format. `GaussianProcesses.jl` often takes log lengthscales as its stored parameters.
- **Variance.** In `KernelFunctions.jl`, variance is a `ScaledKernel` wrapper around a kernel, and nesting it inside sums and products changes what the wrapper means. The adapter must place it consistently with the wire tree.
- **ARD.** Per-dimension lengthscales are an input transform with a vector, and the adapter must check the shape rule against the wire's `ard` option.
- **Noise.** `AbstractGPs.jl` takes the observation noise as an argument when building a finite-dimensional distribution, and `GaussianProcesses.jl` takes a log noise standard deviation. Both must be converted to the wire variance.
- **Log likelihood and prediction quantities.** The adapter reports what the wire runner contract defines: the marginal log likelihood, and the predictive variance with or without noise as the contract states, not the package default.
- **Jitter.** Julia packages add small diagonal terms in different places, and some expose the amount. The small-noise gap found in CIP-0005 (GPy's 1e-8 term) is a warning that this affects agreement at low noise. Each adapter records its jitter, and the conformance report states it.
- **Floating point.** Julia defaults to `Float64`, but some paths accept `Float32`. The runner asserts and records the dtype.

### Julia runner kind

Add `kind: julia` to the runner contract in CIP-0003, in the manner of `kind: r` in CIP-0007:

```yaml
name: juliagps
runners:
  - id: julia
    kind: julia
    script: julia/juliagps/run.jl
    install: {pinned: ci/pinned-julia.txt, latest: ci/latest-julia.txt}
    versions_from_run: true
```

- The script reads the JSON task file and writes the JSON result file, with the same `{log_likelihood, mean, var}`, `{refused}` and `{error}` shapes as every other runner.
- A thin shared helper (`julia/GPScuderia/`, mirroring `octave/gpscuderia/` and `R/gpscuderia/`) reads a wire document, builds the native model, and converts parameters to and from the wire convention. Package-specific code lives under `julia/<package>/`.
- Packages are installed from a pinned `Manifest.toml` (the Julia equivalent of a lockfile), with a `latest` mode that resolves the newest compatible versions, as for Python and R. CI installs Julia with the standard `julia-actions/setup-julia` action.
- JSON handling uses `JSON3.jl` or `JSON.jl`, and YAML uses `YAML.jl`. Which one is chosen is a phase 0 decision, and we prefer whichever has fewer dependencies.
- **Start-up and compilation cost are not part of timing.** Julia compiles on first call. Timing is measured inside the runner after warmup, as in CIP-0003, and the first call is reported separately as cold time, as CIP-0005 already does for JIT compilation in JAX. Cold time is a real cost to users and the report should show it.
- **Thread count.** Julia's thread count is set with `JULIA_NUM_THREADS`, and its BLAS threads are set separately. The runner records both, and the thread rule of CIP-0005 applies.

### Relation to the oracle and to CIP-0004

The oracle remains GPy. Julia frameworks are checked against it in the same way as the others. In CIP-0004's matrix, Julia packages supply rows for family A1 (exact regression) in this CIP, and later for further families once `ApproximateGPs.jl` is admitted. Until a training protocol exists, conformance and benchmarks at fixed parameters are the scope.

### Out of scope

- Approximations other than FTC, until the approximation vocabulary can name what the Julia packages offer. `ApproximateGPs.jl` would be a follow-up CIP.
- Non-Gaussian likelihoods.
- Training or optimiser comparisons (CIP-0004).
- Autodiff comparisons across Julia backends (`ForwardDiff.jl`, `Zygote.jl`, `Enzyme.jl`). The gradient benchmark of CIP-0005 applies when a package exposes likelihood gradients, and the choice of backend is the adapter's business.
- A user-facing Julia package that reads and writes the wire format. This may follow as a by-product of the adapters.

## Implementation Plan

### Phase 0: probe

Install `AbstractGPs.jl`, `KernelFunctions.jl` and `GaussianProcesses.jl` and check, for each wire kernel and each golden, what the package can and cannot express, in the manner of `2026-11-27_r-probe`. Record parameterisations, likelihood conventions, jitter and defaults. The output is a probe note and corrected capability lists. Amend this CIP's expectations if the probe contradicts any claim above.

### Phase 1: runner kind and `juliagps`

1. Add `kind: julia` to the runner contract, with a stub Julia runner exercised by `scripts/check_runner_contract.py`.
2. Add `julia/GPScuderia/` (read, write, convention conversion) and the `juliagps` adapter.
3. Add the framework file and pinned and latest install lists.
4. Add the Julia job to the generated CI matrix.
5. Document parameter conversions in `docs/`.
6. Run the goldens and the accuracy and gradient benchmarks that the declared capabilities allow.
7. Add a round-trip fixture for `models.gp_regression` with an RBF kernel and a Gaussian likelihood, written by GPy and read by `juliagps`, and the reverse.

### Phase 2: `GaussianProcesses.jl`

Add the second adapter once phase 1 is stable, reusing the shared helper.

## Backward Compatibility

Additive. Existing frameworks, goldens and the runner contract for `python`, `octave`, `r` and `command` are unchanged. The registry gains entries for the new frameworks, with explicit `null` for unsupported types, as before. The CI matrix gains a Julia job. No wire schema change is needed for phase one.

## Testing Strategy

- **Conformance.** Every golden matches within tolerance if declared as supported, and is refused with a reason otherwise. The harness checks both directions, as in CIP-0003.
- **Cross-check.** Agreement with the GPy oracle on log likelihood and predictions at fixed parameters, at several noise levels including the small-noise regime that exposed the jitter gap in CIP-0005.
- **Round trip.** The phase 1 fixture is the first cross-language round trip for a framework entry that was not derived from GPy conventions. Until it passes, the `juliagps` entry is marked provisional, as the GPyTorch, GPflow and GPJax entries were.
- **Gradients.** Where a package exposes likelihood gradients, compare them with the oracle through the existing gradient benchmark. Where it does not, say so in the framework file and skip, and do not finite-difference through a black box and call it a package result.
- **Refusals.** Unsupported kernels, approximations and non-Gaussian likelihoods must be refused, with tests that fail if an adapter substitutes.
- **Runner contract.** The stub Julia runner passes the same contract check as the stub Python, Octave and R runners.

## Related Requirements

None recorded. This CIP serves the shared model interchange work tracked in GPc under REQ-000A and CIP-0016.

## Implementation Status

- [ ] Phase 0: probe `AbstractGPs.jl`, `KernelFunctions.jl` and `GaussianProcesses.jl`
- [ ] Add `kind: julia` to the runner contract, with a stub test
- [ ] `julia/GPScuderia/` helper and conversion documentation
- [ ] `juliagps` adapter and framework file
- [ ] Julia job in the generated CI matrix
- [ ] Benchmarks run for `juliagps`
- [ ] Round-trip fixture between GPy and `juliagps`
- [ ] `GaussianProcesses.jl` adapter and framework file (phase 2)

## References

- CIP-0003: declarative framework files and the runner contract
- CIP-0004: standardised evaluation and a learned guide
- CIP-0005: first benchmark suite
- CIP-0007: R frameworks, the model for the new runner kind
- JuliaGPs organisation: `AbstractGPs.jl`, `KernelFunctions.jl`, `ApproximateGPs.jl`, `GPLikelihoods.jl`, `Stheno.jl`, `TemporalGPs.jl`
- `GaussianProcesses.jl` (Fairbrother, Nemeth, Rischard, Brea and Pinder, *Journal of Statistical Software*, 2022)
