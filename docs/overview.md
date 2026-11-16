# Interchange overview (0.1 draft)

## Encodings

- **YAML** — primary authoring and golden fixtures
- **JSON** — isomorphic machine encoding of the same tree

## Units

Parameters on the wire use **GPy-natural** values (e.g. RBF `lengthscale`, not GPc `inverseWidth`). Toolbox adapters convert at the boundary.

## Versioning

Bump `schema_version` when removing or renaming fields. Additive optional fields may stay on the same minor draft until 0.1 is tagged stable.

## Related

- GPc CIP-0016 / REQ-000A
- GPy `to_dict` / `from_dict`
- GPmat `gpDeconstruct` / `gpReconstruct`
