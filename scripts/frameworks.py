"""Framework declarations (frameworks/*.yaml) and the expectations derived from them (CIP-0003).

A golden's features are compared with a framework's declared capabilities:
all features declared -> the runner must match; any missing -> it must refuse.
Capability entries are bare names (all options) or {name: {options: [...]}}.
"""
import pathlib
import yaml

root = pathlib.Path(__file__).resolve().parents[1]
FAMILIES = {"approximations": "", "kernels": "kern.", "likelihoods": "likelihoods.", "inference": "", "extras": ""}


def load_registry():
    return yaml.safe_load((root / "registry/types.yaml").read_text())


def vocabulary(registry):
    """Names each capability family may use, taken from the registry."""
    tags = [t["portable"] for t in registry["types"]]
    return {
        "approximations": set(registry["approximations"]),
        "kernels": {t[5:] for t in tags if t.startswith("kern.")},
        "likelihoods": {t[12:] for t in tags if t.startswith("likelihoods.")},
        "inference": {"exact", "laplace"} | set(registry.get("reserved_inference", {})),
        "extras": {t for t in tags if t.startswith(("mean.", "normalizer."))},
    }


def _entries(items):
    """Normalise a capability list to {name: None | set(options)}; None claims every option."""
    out = {}
    for it in items or []:
        if isinstance(it, str):
            out[it] = None
        else:
            (name, spec), = it.items()
            out[name] = set((spec or {}).get("options", []))
    return out


def load_frameworks(directory=None, registry=None):
    registry = registry or load_registry()
    vocab, options = vocabulary(registry), registry.get("capability_options", {})
    fws = {}
    for p in sorted((directory or root / "frameworks").glob("*.yaml")):
        d = yaml.safe_load(p.read_text())
        if d["name"] != p.stem:
            raise ValueError("%s: name '%s' does not match the file name" % (p.name, d["name"]))
        caps = {fam: _entries(d["capabilities"].get(fam)) for fam in FAMILIES}
        unknown = set(d["capabilities"]) - set(FAMILIES)
        if unknown:
            raise ValueError("%s: unknown capability families %s" % (p.name, sorted(unknown)))
        for fam, entries in caps.items():
            for name, opts in entries.items():
                if name not in vocab[fam]:
                    raise ValueError("%s: '%s' is not a registry %s name" % (p.name, name, fam))
                if opts:
                    bad = opts - set(options.get("kern." + name, []))
                    if bad:
                        raise ValueError("%s: unknown options %s for %s" % (p.name, sorted(bad), name))
        d["_caps"] = caps
        fws[d["name"]] = d
    return fws


def features(doc):
    """What a golden exercises, as {family: {name: set(options)}}."""
    kernels = {}

    def walk(k):
        opts = kernels.setdefault(k["type"][5:], set())
        if k.get("ARD"):
            opts.add("ard")
        if doc.get("input_dim", 1) > 1 and k["type"] in ("kern.rbf", "kern.matern32", "kern.matern52"):
            opts.add("multi_input")
        for part in k.get("parts", []):
            walk(part)
    walk(doc["kernel"])
    extras = {doc[k]["type"]: set() for k in ("mean", "normalizer") if k in doc}
    return {"approximations": {doc.get("approximation", "ftc"): set()}, "kernels": kernels,
            "likelihoods": {doc["likelihood"]["type"][12:]: set()},
            "inference": {doc.get("inference", "exact"): set()}, "extras": extras}


def missing(caps, doc):
    """Reasons the framework's declared capabilities do not cover the golden (empty list: covered)."""
    reasons = []
    for fam, used in features(doc).items():
        for name, opts in used.items():
            claimed = caps[fam].get(name, False)
            if claimed is False:
                reasons.append("%s '%s'" % (fam[:-1] if fam != "inference" else fam, name))
            elif claimed is not None and not opts <= claimed:
                reasons.append("%s '%s' option %s" % (fam[:-1], name, sorted(opts - claimed)))
    return reasons


def expected(framework, doc):
    """'match' when the framework declares everything the golden uses, else 'refused'."""
    return "refused" if missing(framework["_caps"], doc) else "match"
