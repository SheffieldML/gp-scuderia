# gp-scuderia

Shared, framework-neutral **model interchange** for Gaussian process toolboxes: [GPc](https://github.com/SheffieldML/GPc), [GPy](https://github.com/SheffieldML/GPy), [GPmat](https://github.com/SheffieldML/GPmat), and the wider ecosystem ([GPyTorch](https://gpytorch.ai), [GPflow](https://github.com/GPflow/GPflow), [GPJax](https://github.com/JaxGaussianProcesses/GPJax)).

- **YAML primary**, JSON isomorphic (same abstract tree)
- **Portable type tags** inspired by GPy class strings (`GPy.kern.RBF` → `kern.rbf`)
- Schema versioned independently of any one toolbox
- No framework is normative: the registry maps each wire tag to GPy, GPc, GPmat, GPyTorch, GPflow and GPJax, with explicit `null` where a framework has no equivalent

GPc tracks this work under [REQ-000A](https://github.com/SheffieldML/GPc/blob/main/requirements/req000A_shared-model-interchange.md) / [CIP-0016](https://github.com/SheffieldML/GPc/blob/main/cip/cip0016.md).

## Layout

| Path | Role |
|------|------|
| `schemas/` | JSON Schema for the interchange tree (validates YAML and JSON) |
| `registry/types.yaml` | Portable tag ↔ native type and parameter map for each framework; per-framework approximation support |
| `examples/` | Golden `.yaml` models (and optional `.json` twins) |
| `docs/` | Human notes: units, approximation tags, versioning |
| `tests/invalid/` | Documents that must be rejected |
| `scripts/validate.py` | Validates examples, rejects invalid fixtures, checks registry coverage |

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

Schema **0.1** draft — not yet a stable contract. Do not rely on field names until 0.1 is tagged. Registry entries for GPyTorch, GPflow and GPJax are provisional and unverified; no cross-framework round-trip fixture exists yet.

Run `python scripts/validate.py` (needs `pyyaml`, `jsonschema`).

## License

BSD-3-Clause (aligned with SheffieldML GP toolboxes), unless noted otherwise when published to GitHub.
