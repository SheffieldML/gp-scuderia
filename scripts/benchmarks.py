"""Accuracy benchmarks (CIP-0003): seeded models, oracle references, every framework.

    python scripts/benchmarks.py generate --out DIR          # needs GPy (the oracle)
    python scripts/benchmarks.py run FRAMEWORK --data DIR [--mode M] [--json out.json]

`generate` expands benchmarks/*.yaml into wire models (YAML and JSON twin) with an x-reference
computed by the oracle. `run` sends the cases a framework can run (capabilities from
frameworks/<name>.yaml, features derived from the model as in compare.py) through the runner
contract and compares with the references. A case outside the framework's capabilities is
`skipped`. A benchmark may list `known: {framework: {case: reason}}`; a mismatch recorded there is
`known` and does not fail the build, but a known case that starts to match is reported as an
unexpected pass so the entry gets removed.
"""
import argparse, copy, json, pathlib, platform, sys, time
import numpy as np, yaml

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python")); sys.path.insert(0, str(root / "scripts"))
from gp_scuderia import load, dump, to_json_twin
import frameworks as F
from harness import launch, versions


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def benchmarks():
    return [yaml.safe_load(p.read_text()) for p in sorted((root / "benchmarks").glob("*.yaml"))]


def make_model(bench, case, index):
    """A wire model for one case. The seed depends on benchmark seed and case index only."""
    c = deep_merge(bench["base"], {k: v for k, v in case.items() if k != "id"})
    rng = np.random.default_rng([bench["seed"], index])
    n, d = c["n"], c["input_dim"]
    if c.get("x") == "clustered":
        centres = rng.uniform(-3, 3, size=(4, d))
        X = centres[rng.integers(0, 4, n)] + 1e-3 * rng.standard_normal((n, d))
    elif c.get("x") == "duplicated":
        base = rng.uniform(-3, 3, size=((n + 1) // 2, d))
        X = np.repeat(base, 2, axis=0)[:n]
    else:
        X = rng.uniform(-3, 3, size=(n, d))
    Y = np.sin(X.sum(1, keepdims=True)) + 0.1 * rng.standard_normal((n, 1))
    doc = {"schema_version": "0.1", "type": "models.gp_regression", "input_dim": d, "output_dim": 1,
           "approximation": c["approximation"], "kernel": c["kernel"],
           "likelihood": {"type": "likelihoods.gaussian", "variance": c["noise"]},
           "data": {"X": X.tolist(), "Y": Y.tolist()}}
    if c.get("inducing"):
        doc["inducing_inputs"] = X[rng.choice(n, c["inducing"], replace=False)].tolist()
    Xnew = rng.uniform(-3, 3, size=(5, d)).tolist()
    return doc, Xnew


COND_MAX_N = 4000


def _stationary_K(kern, X):
    """Covariance matrix for rbf, matern32 and matern52 (ARD allowed); None for anything else."""
    t = kern["type"].split(".")[-1]
    if t not in ("rbf", "matern32", "matern52") or "parts" in kern:
        return None
    ell = np.broadcast_to(np.asarray(kern["lengthscale"], float), (X.shape[1],))
    Z = X / ell
    r2 = np.maximum(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1), 0.0)
    r = np.sqrt(r2)
    v = kern["variance"]
    if t == "rbf":
        return v * np.exp(-0.5 * r2)
    if t == "matern32":
        return v * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    return v * (1 + np.sqrt(5) * r + 5.0 / 3.0 * r2) * np.exp(-np.sqrt(5) * r)


def descriptors(doc, generator=None):
    """Meta-features of a generated case (CIP-0005). One implementation for every benchmark, so
    that real tasks and the guide in CIP-0004 can use the same numbers.

    signal_to_noise: kernel variance over noise variance at the generating parameters.
    lengthscale_over_spacing: median nearest-neighbour distance is measured in units of the
      lengthscale (per dimension, so ARD is respected) and inverted; large means smooth data.
    log10_condition: of K + noise I, only for n <= COND_MAX_N and stationary kernels.
    """
    X = np.asarray(doc["data"]["X"], float)
    n, d = X.shape
    kern = doc["kernel"]
    noise = doc["likelihood"]["variance"]
    out = {"n": n, "input_dim": d,
           "signal_to_noise": float(kern["variance"] / noise) if "variance" in kern else None,
           "lengthscale_over_spacing": None, "log10_condition": None}
    if "lengthscale" in kern and "parts" not in kern:
        ell = np.broadcast_to(np.asarray(kern["lengthscale"], float), (d,))
        Z = X / ell
        if n <= COND_MAX_N:
            D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(D, np.inf)
            nn = D.min(1)
        else:
            idx = np.random.default_rng(0).choice(n, COND_MAX_N, replace=False)
            D = np.sqrt(((Z[idx][:, None, :] - Z[None, :, :]) ** 2).sum(-1))
            D[np.arange(len(idx)), idx] = np.inf
            nn = D.min(1)
        # duplicated inputs give a zero distance; spacing is then taken over the distinct pairs
        nn = nn[nn > 0]
        if len(nn):
            out["lengthscale_over_spacing"] = float(1.0 / np.median(nn))
    if n <= COND_MAX_N:
        K = _stationary_K(kern, X)
        if K is not None:
            ev = np.linalg.eigvalsh(K + noise * np.eye(n))
            out["log10_condition"] = float(np.log10(ev[-1] / max(ev[0], np.finfo(float).tiny)))
    if generator:
        out["generator"] = generator
    return out


def generate(out):
    import jsonschema
    from gp_scuderia import gpy_adapter
    schema = json.loads((root / "schemas" / "model.schema.json").read_text())
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    for bench in benchmarks():
        for i, case in enumerate(bench["cases"]):
            doc, Xnew = make_model(bench, case, i)
            jsonschema.validate(doc, schema)
            doc["x-descriptors"] = descriptors(doc, {"benchmark": bench["name"], "case": case["id"],
                                                     "seed": [bench["seed"], i]})
            ref = gpy_adapter.reference(doc, Xnew)
            ref.update(oracle="GPy", tolerance=bench["tolerance"])
            doc["x-reference"] = ref
            p = out / ("%s__%s.yaml" % (bench["name"], case["id"]))
            dump(doc, p); to_json_twin(p)
            print("generated", p.stem)


def run(fw, data, mode, json_out):
    framework = F.load_frameworks()[fw]
    rn = framework["runners"][0]
    data = pathlib.Path(data).resolve()
    known = {}
    for b in benchmarks():
        for f, cases in (b.get("known") or {}).items():
            for cid, why in cases.items():
                known[(f, "%s__%s" % (b["name"], cid))] = why
    names = sorted(p.stem for p in data.glob("*.yaml"))
    docs = {n: load(data / (n + ".yaml")) for n in names}
    runnable = [n for n in names if not F.missing(framework["_caps"], docs[n])]
    extra = {}
    t0 = time.time()
    raw = launch(fw, rn, [{"name": n, "golden": str(data / (n + ".json"))} for n in runnable], extra) if runnable else {}
    results, bad = [], 0
    for n in names:
        ref = docs[n]["x-reference"]; tol = ref["tolerance"]
        rec = {"benchmark": n.split("__")[0], "case": n.split("__")[1], "name": n}
        why = F.missing(framework["_caps"], docs[n])
        if why:
            rec.update(status="skipped", reason=str(why), ok=True)
        else:
            g = raw[n]
            if "refused" in g:
                rec.update(status="refused", reason=g["refused"])
            elif "error" in g:
                rec.update(status="error", reason=g["error"])
            else:
                dev = {"log_likelihood": abs(g["log_likelihood"] - ref["log_likelihood"]),
                       "mean": float(np.abs(np.array(g["mean"]) - ref["mean"]).max()),
                       "var": float(np.abs(np.array(g["var"]) - ref["var"]).max())}
                rec.update(deviation=dev, tolerance={k: tol[k] for k in dev},
                           status="match" if all(dev[k] <= tol[k] for k in dev) else "mismatch")
            k = known.get((fw, n))
            if k and rec["status"] != "match":
                rec.update(status="known", reason=k, ok=True)
            elif k:
                rec.update(ok=False, reason="known failure now matches; remove it from the benchmark")
            else:
                rec["ok"] = rec["status"] == "match"
        bad += not rec["ok"]
        results.append(rec)
        dev = rec.get("deviation")
        print("%-5s %-34s %-8s %s" % ("ok" if rec["ok"] else "FAIL", n, rec["status"],
              ("max dev/tol = %.2g" % max(dev[k] / rec["tolerance"][k] for k in dev)) if dev else rec.get("reason", "")[:80]))
    doc = {"benchmark_run": True, "framework": None, "benchmark_framework": fw, "mode": mode,
           "python": platform.python_version(), "versions": {**versions(rn), **extra},
           "seconds": round(time.time() - t0, 2), "results": results}
    del doc["framework"]
    if json_out:
        pathlib.Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(json_out).write_text(json.dumps(doc, indent=2) + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("framework"); r.add_argument("--data", required=True)
    r.add_argument("--mode", default="pinned"); r.add_argument("--json")
    a = ap.parse_args()
    generate(a.out) if a.cmd == "generate" else run(a.framework, a.data, a.mode, a.json)
