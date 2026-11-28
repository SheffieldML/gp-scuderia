# Interchange overview (0.1 draft)

## Encodings

- **YAML** — primary authoring and golden fixtures. GPc reads it directly when built with `make YAML=1` (needs yaml-cpp built with the same compiler as GPc; Homebrew's clang build does not link against g++). The default GPc build reads JSON and refuses `.yaml` with a hint to convert (`gp_scuderia.io.to_json_twin`). Rebuild `CJson.o` when you change the flag.
- **JSON** — isomorphic machine encoding of the same tree

## Units

Parameters on the wire use **GPy-natural** values (e.g. RBF `lengthscale`, not GPc `inverseWidth`). Toolbox adapters convert at the boundary.

## Parameter conventions

The wire carries natural, positive, untransformed values: `variance` is a variance, `lengthscale` is a length scale, a Gaussian likelihood `variance` is a noise variance. Adapters convert at the boundary. Known mismatches:

| Framework | Mismatch | Conversion |
|---|---|---|
| GPc | `inverseWidth`, noise precision `beta` | CIP-0014 table; `1/variance` |
| GPyTorch | `RBFKernel` has no variance | wrap in `ScaleKernel`; `variance` ↔ `outputscale` |
| GPyTorch | no white kernel; noise is in the likelihood | reject `kern.white`, do not fold silently |
| GPflow | softplus-transformed variables | adapters store the constrained value; wire is constrained |
| GPJax | Gaussian likelihood uses `obs_stddev` | `sqrt` / square |

## Semantics fixed by the format

- **Noise.** The Gram matrix is K(kernel, including any `kern.white`) + `likelihood.variance` on the diagonal. Both add.
- **Mean and target scaling.** With neither field the model is zero-mean and unscaled. Two optional fields change that, and are mutually exclusive in 0.1 because the order in which a mean function and a normalizer apply is not specified:
  - `mean: {type: mean.constant, value: c}` — prior mean `c` on the original target scale. Equivalent to fitting a zero-mean GP to `y - c`; no Jacobian term.
  - `mean: {type: mean.linear_basis, basis: {type: basis.polynomial, powers: [[0], [1]]}, beta: {mode: profiled}}` — prior mean `F(x) beta` with one basis column per multi-index (here `1` and `x`). `beta.mode` is part of the meaning, because the modes report different log likelihoods: `fixed` (needs `value`), `profiled` (generalised least squares estimate), `flat_prior` (restricted likelihood, subtracting `0.5 log|F' K_y^-1 F|`) and `gaussian_prior` (needs `prior_mean`, `prior_cov`). `profiled` and `flat_prior` predict identically; predictive variance includes the universal-kriging term. Exact GP only in 0.1. See CIP-0009.
  - `normalizer: {type: normalizer.center, mean: m}` or `{type: normalizer.standardize, mean: m, std: s}` — the model was fitted to `(y - m) / s`. `likelihood.variance` and kernel parameters live on that normalized scale. Log likelihood and predictions are reported on the original scale, so the log likelihood includes the Jacobian `-n log s` with `n` the number of target values; predictive variance is multiplied by `s^2`.
  - Adapters that cannot realise a field raise. GPc and GPmat store `m = (y - bias) / scale`, so both fields map onto `bias` and `scale`. Neither toolbox's own objective includes the `-n log s` term, so the adapters add it when reporting.
- **Numerical references.** Goldens carry `x-reference` (GPy-generated log likelihood and predictions, with tolerances). Toolboxes that add jitter to sparse models are covered by the stated tolerances.

## Stacked multi-output models (ICM and LCM)

Several outputs share one dataset, as in GPy's `GPCoregionalizedRegression`. `data.X` carries the output index as an integer column (here column `q`, after the `q` input columns), `data.Y` has one column, and `output_dim` stays 1.

- `active_dims` on any kernel leaf lists the `X` columns it reads. A leaf without it reads every column; `lengthscale` lists must match the number of columns read.
- `kern.coregionalize` has `output_dim`, `W` (`output_dim x rank`), `kappa` (`output_dim` positive entries) and exactly one active column holding the index. Its covariance is `B[i, j]` with `B = W W' + diag(kappa)`.
- ICM is `kern.prod` of a spatial kernel and a `kern.coregionalize`. LCM is `kern.add` of ICM terms.
- `likelihoods.mixed_gaussian` gives one noise variance per output and names the index column with `output_index_column`. The Gram is `K` plus `diag(variance[index])`.

Only GPy and GPc implement these in 0.1. GPmat raises `gpscuderia:unsupported` on `active_dims` and on `likelihoods.mixed_gaussian`. Goldens: `ftc_icm`, `ftc_lcm`. The Kronecker form (`output_dim > 1` with a shared input set) is not yet in the schema.

## Approximations

`approximation` is an enumerated tag. A framework that cannot realise a tag must raise, not substitute. `registry/types.yaml` records which frameworks support which tags (provisional for GPyTorch, GPflow, GPJax).

## Extension fields

Unknown top-level fields are rejected. Use an `x-` prefix for experimental fields.

## Versioning

Bump `schema_version` when removing or renaming fields. Additive optional fields may stay on the same minor draft until 0.1 is tagged stable.

## Related

- GPc CIP-0016 / REQ-000A
- GPy `to_dict` / `from_dict`
- GPmat `gpDeconstruct` / `gpReconstruct`
- GPyTorch `state_dict`, GPflow `gpflow.utilities.parameter_dict`, GPJax `Module` pytrees (native persistence; none is portable across frameworks)
