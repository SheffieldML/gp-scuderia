# gp-scuderia

Shared **model interchange** for SheffieldML Gaussian process toolboxes ([GPc](https://github.com/SheffieldML/GPc), [GPy](https://github.com/SheffieldML/GPy), [GPmat](https://github.com/SheffieldML/GPmat)).

- **YAML primary**, JSON isomorphic (same abstract tree)
- **Portable type tags** inspired by GPy class strings (`GPy.kern.RBF` → `kern.rbf`)
- Schema versioned independently of any one toolbox

GPc tracks this work under [REQ-000A](https://github.com/SheffieldML/GPc/blob/main/requirements/req000A_shared-model-interchange.md) / [CIP-0016](https://github.com/SheffieldML/GPc/blob/main/cip/cip0016.md).

## Layout

| Path | Role |
|------|------|
| `schemas/` | JSON Schema for the interchange tree (validates YAML and JSON) |
| `registry/types.yaml` | Portable tag ↔ GPy class ↔ GPc stream type ↔ GPmat kern type |
| `examples/` | Golden `.yaml` models (and optional `.json` twins) |
| `docs/` | Human notes: units, approximation tags, versioning |

## Type tags

Strip the `GPy.` prefix and lowercase:

| GPy `class` | Portable `type` |
|-------------|-----------------|
| `GPy.kern.RBF` | `kern.rbf` |
| `GPy.kern.White` | `kern.white` |
| `GPy.kern.Add` | `kern.add` |
| `GPy.likelihoods.Gaussian` | `likelihoods.gaussian` |
| `GPy.models.GPRegression` | `models.gp_regression` |

Writers emit portable tags. Readers **may** accept full `GPy.*` strings as aliases.

## Status

Schema **0.1** scaffold — not yet a stable contract. Do not rely on field names until 0.1 is tagged.

## License

BSD-3-Clause (aligned with SheffieldML GP toolboxes), unless noted otherwise when published to GitHub.
