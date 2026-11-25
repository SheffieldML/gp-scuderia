# Interchange overview (0.1 draft)

## Encodings

- **YAML** — primary authoring and golden fixtures
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
