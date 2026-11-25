"""Load and dump interchange documents (YAML primary, JSON isomorphic)."""
import json
import pathlib

import yaml


def load(path):
    path = pathlib.Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def dump(doc, path):
    path = pathlib.Path(path)
    if path.suffix == ".json":
        path.write_text(json.dumps(doc, indent=2) + "\n")
    else:
        path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))


def to_json_twin(yaml_path):
    """Write the JSON twin next to a YAML file (Octave/MATLAB read JSON)."""
    yaml_path = pathlib.Path(yaml_path)
    twin = yaml_path.with_suffix(".json")
    dump(load(yaml_path), twin)
    return twin
