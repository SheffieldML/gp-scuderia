"""Accuracy benchmarks (CIP-0003): seeded models, oracle references, every framework.

    python scripts/benchmarks.py generate --out DIR          # needs GPy (the oracle)
    python scripts/benchmarks.py run FRAMEWORK --data DIR [--mode M] [--json out.json]

`generate` expands benchmarks/*.yaml into wire models (YAML and JSON twin) with an x-reference
computed by the oracle. `run` sends the cases a framework can run (capabilities from
frameworks/<name>.yaml, features derived from the model as in compare.py) through the runner
contract and compares with the references. A case outside the framework's capabilities is
`skipped`. A benchmark may list `known: {framework: {case: reason}}` (a reason may instead be `{why: ..., fixed_by: feature}`,
which stops applying once the oracle GPy reports that feature in x-reference.oracle_features); a mismatch recorded there is
`known` and does not fail the build, but a known case that starts to match is reported as an
unexpected pass so the entry gets removed.
"""
import argparse, copy, itertools, json, os, pathlib, platform, sys, time
import numpy as np, yaml

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python")); sys.path.insert(0, str(root / "scripts"))
from gp_scuderia import load, dump, to_json_twin
import frameworks as F
from harness import launch, versions, THREAD_VARS


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


_AXIS_ID = {"n": "n%d", "input_dim": "d%d", "inducing": "M%d"}


def latent_gate(framework, kind):
    """Reasons a framework cannot run a latent-variable benchmark of this kind (empty: it can).

    The collapsed bound is the least a GP-LVM framework must provide, so every latent kind needs `bound` as well as
    its own operation. Without this, a framework that declares only `gradient` (for sparse regression) would be sent
    a Bayesian GP-LVM document it cannot read.
    """
    ops = framework.get("operations", [])
    need = ["bound"] if LATENT_KINDS[kind] == "bound" else ["bound", LATENT_KINDS[kind]]
    return ["the framework file declares no %s operation" % o for o in need if o not in ops]


def expand_grid(b):
    """Expand `grid` (axis: values) into cases. `tiers` lists, per tier, the axis values it keeps;
    the `scheduled` tier is always everything. Each case records the tiers it belongs to."""
    if "grid" not in b:
        return b
    axes = list(b["grid"]); tiers = b.get("tiers", {}); cases = []
    for vals in itertools.product(*[b["grid"][a] for a in axes]):
        pt = dict(zip(axes, vals))
        cid = "_".join(_AXIS_ID[a] % pt[a] if a in _AXIS_ID else str(pt[a]) for a in axes)
        member = ["scheduled"] + [t for t, sel in tiers.items() if all(pt[a] in v for a, v in sel.items())]
        cases.append({"id": cid, **pt, "tiers": member})
    return {**b, "cases": cases}


def benchmarks():
    return [expand_grid(yaml.safe_load(p.read_text())) for p in sorted((root / "benchmarks").glob("*.yaml"))]


def make_model(bench, case, index):
    """A wire model for one case. The seed depends on benchmark seed and case index only."""
    c = deep_merge(bench["base"], {k: v for k, v in case.items() if k != "id"})
    shared = bench.get("shared_data", False)   # every case sees the same data and nested inducing sets
    rng = np.random.default_rng([bench["seed"], 0 if shared else index])
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
    if c.get("likelihood") == "bernoulli":
        Y = (Y + 0.3 * rng.standard_normal((n, 1)) > 0).astype(float)
    doc = {"schema_version": "0.1", "type": "models.gp_regression", "input_dim": d, "output_dim": 1,
           "approximation": c["approximation"], "kernel": c["kernel"],
           "likelihood": {"type": "likelihoods.gaussian", "variance": c.get("noise", 0.1)},
           "data": {"X": X.tolist(), "Y": Y.tolist()}}
    if c.get("likelihood") == "bernoulli":
        doc.update(type="models.gp", inference="laplace",
                   likelihood={"type": "likelihoods.bernoulli", "link": "probit"})
    if shared:
        perm = rng.permutation(n)               # drawn for every case, so the prefixes are nested
        Xnew = rng.uniform(-3, 3, size=(5, d)).tolist()
        if c.get("inducing"):
            doc["inducing_inputs"] = X[perm[:c["inducing"]]].tolist()
        return doc, Xnew
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
    noise = doc["likelihood"].get("variance")      # none for non-Gaussian likelihoods
    out = {"n": n, "input_dim": d,
           "signal_to_noise": float(kern["variance"] / noise) if "variance" in kern and noise else None,
           "lengthscale_over_spacing": None, "log10_condition": None}
    if "lengthscale" in kern and "parts" not in kern:
        ell = np.broadcast_to(np.asarray(kern["lengthscale"], float), (d,))
        Z = X / ell
        from scipy.spatial.distance import cdist
        idx = np.arange(n) if n <= COND_MAX_N else np.sort(np.random.default_rng(0).choice(n, COND_MAX_N, replace=False))
        nn = np.empty(len(idx))
        for lo in range(0, len(idx), 500):     # chunked: n x n x d would not fit in memory
            sel = idx[lo:lo + 500]
            D = cdist(Z[sel], Z)
            D[np.arange(len(sel)), sel] = np.inf
            nn[lo:lo + 500] = D.min(1)
        # duplicated inputs give a zero distance; spacing is then taken over the distinct pairs
        nn = nn[nn > 0]
        if len(nn):
            out["lengthscale_over_spacing"] = float(1.0 / np.median(nn))
    if n <= COND_MAX_N and noise:
        K = _stationary_K(kern, X)
        if K is not None:
            ev = np.linalg.eigvalsh(K + noise * np.eye(n))
            out["log10_condition"] = float(np.log10(ev[-1] / max(ev[0], np.finfo(float).tiny)))
    if generator:
        out["generator"] = generator
    return out


def check_properties(bench, ll):
    """Properties of a family of cases (CIP-0005), from the log likelihoods a framework returned.

    ll maps case id to log likelihood, with None for a case the framework did not run. Asserted:
    each DTCVar value is a lower bound on the exact value, and DTCVar does not decrease as the
    nested inducing set grows. Reported only: the FITC minus exact gap and both curves in M, which
    describe approximation quality but are not guaranteed by theory.
    Returns (records, table); a record is {property, status, ok, reason}.
    """
    p = bench["properties"]; exact = ll.get(p["exact"])
    # roundoff allowance: absolute, plus relative to the size of the log likelihood (GPy's own
    # DTCVar exceeds the exact value by 4e-6 at M=512, n=2000, where |LL| is about 2000)
    slack = p.get("slack", 1e-6) + p.get("rel_slack", 1e-8) * abs(exact if exact is not None else 0.0)
    sizes = p["sizes"]
    dv = {m: ll.get("dtcvar_%d" % m) for m in sizes}
    ft = {m: ll.get("fitc_%d" % m) for m in sizes}
    recs = []

    def rec(name, status, reason=""):
        recs.append({"benchmark": bench["name"], "case": name, "name": "%s__%s" % (bench["name"], name),
                     "status": status, "ok": status in ("match", "skipped"), "reason": reason})
    have = [m for m in sizes if dv[m] is not None]
    if exact is None or not have:
        rec("dtcvar_lower_bound", "skipped", "needs the exact case and at least one DTCVar case")
    else:
        viol = [(m, dv[m] - exact) for m in have if dv[m] > exact + slack]
        rec("dtcvar_lower_bound", "mismatch" if viol else "match",
            "DTCVar exceeds the exact log likelihood at M=%s by %s" % tuple(map(str, (viol[0][0], "%.3g" % viol[0][1]))) if viol else "")
    if len(have) < 2:
        rec("dtcvar_monotone_in_M", "skipped", "needs at least two DTCVar cases")
    else:
        viol = [(a, b, dv[a] - dv[b]) for a, b in zip(have, have[1:]) if dv[b] < dv[a] - slack]
        rec("dtcvar_monotone_in_M", "mismatch" if viol else "match",
            "DTCVar falls from M=%d to M=%d by %.3g" % viol[0] if viol else "")
    table = {"exact": exact, "rows": [{"M": m, "dtcvar": dv[m], "fitc": ft[m],
             "fitc_gap": (ft[m] - exact) if ft[m] is not None and exact is not None else None} for m in sizes]}
    return recs, table


def oracle_gradient(doc, Xnew):
    """GPy's analytic gradient, cross-checked against central differences of GPy's own log
    likelihood. The generator refuses to emit a reference the two disagree on (relative 1e-4,
    with an absolute floor of 1e-6 for gradients near zero)."""
    from gp_scuderia import gpy_adapter
    doc = dict(doc); doc["x-reference"] = {"Xnew": Xnew}
    a, fd = gpy_adapter.gradient(doc), gpy_adapter.gradient_fd(doc)
    worst = 0.0
    for k in a:
        x, y = np.atleast_1d(a[k]), np.atleast_1d(fd[k])
        worst = max(worst, float(np.max(np.abs(x - y) / (1e-6 + np.abs(y)))))
    if worst > 1e-4:
        raise SystemExit("oracle gradient disagrees with finite differences (max rel %.2g) for %s" % (worst, doc.get("x-descriptors")))
    return {k: (np.asarray(v).tolist() if np.ndim(v) else float(v)) for k, v in a.items()}, worst


def gradient_deviation(g, ref, tol):
    """Worst |a - r| / (atol + rtol |r|) over every wire parameter; 1.0 is the tolerance. Missing
    or extra keys, or a different vector length, make the deviation infinite."""
    if set(g) != set(ref):
        return float("inf"), "parameter keys differ: %s" % sorted(set(g) ^ set(ref))
    worst = 0.0
    for k, r in ref.items():
        a, r = np.atleast_1d(np.asarray(g[k], float)), np.atleast_1d(np.asarray(r, float))
        if a.shape != r.shape:
            return float("inf"), "%s has shape %s, expected %s" % (k, a.shape, r.shape)
        worst = max(worst, float(np.max(np.abs(a - r) / (tol["atol"] + tol["rtol"] * np.abs(r)))))
    return worst, None


LATENT_KINDS = {"latent_bound": "bound", "latent_collapse": "bound", "psi": "psi", "latent_gradient": "gradient", "latent_timing": "bound"}   # kind -> runner operation


def make_latent_model(bench, case, index):
    """A wire Bayesian GP-LVM at fixed q(X), Z and parameters (CIP-0006). The seed depends on the
    benchmark seed and case index only. `s_value` fixes S (the collapse limit); `s_range` sets its range."""
    c = deep_merge(bench["base"], {k: v for k, v in case.items() if k != "id"})
    rng = np.random.default_rng([bench["seed"], index])
    n, q, d, m = c["n"], c["latent_dim"], c["output_dim"], c["inducing"]
    mu = rng.standard_normal((n, q))
    lo, hi = c.get("s_range", [0.1, 0.6])
    S = rng.uniform(lo, hi, (n, q))
    if "s_value" in c:
        S = np.full((n, q), float(c["s_value"]))
    Z = rng.standard_normal((m, q))
    Y = np.sin(mu @ rng.standard_normal((q, d))) + 0.1 * rng.standard_normal((n, d))
    return {"schema_version": "0.1", "type": "models.bayesian_gplvm", "approximation": "dtcvar",
            "inducing_inputs": Z.tolist(), "kernel": c["kernel"],
            "likelihood": {"type": "likelihoods.gaussian", "variance": c["noise"]},
            "latent": {"dim": q, "mean": mu.tolist(), "variance": S.tolist(), "prior": {"type": "standard_normal"}},
            "data": {"Y": Y.tolist()}}


def latent_descriptors(doc, generator):
    """Meta-features of a latent case. Gauge-free: sizes and signal-to-noise only."""
    k = doc["kernel"]
    return {"n": len(doc["latent"]["mean"]), "latent_dim": doc["latent"]["dim"],
            "output_dim": len(doc["data"]["Y"][0]), "inducing": len(doc["inducing_inputs"]),
            "signal_to_noise": float(k["variance"] / doc["likelihood"]["variance"]), "generator": generator}


def _kernel_fn(kern):
    """k(X, Z) for the kernels with a quadrature oracle: rbf (ARD allowed) and linear."""
    t = kern["type"].split(".")[-1]
    v = kern["variance"]
    if t == "rbf":
        ell = np.asarray(kern["lengthscale"], float)
        return lambda A, B: v * np.exp(-0.5 * (((A[..., :, None, :] - B[None, :, :]) / ell) ** 2).sum(-1))
    if t == "linear":
        return lambda A, B: v * np.einsum("...ik,jk->...ij", A, B)
    raise SystemExit("no quadrature oracle for kernel %s" % kern["type"])


def gauss_hermite_psi(doc, nodes=48):
    """psi0, psi1 and psi2 as expectations of the kernel under q(X), by tensor Gauss-Hermite
    quadrature. Independent of every framework: it uses only the kernel and q(X)."""
    mu, S = np.asarray(doc["latent"]["mean"], float), np.asarray(doc["latent"]["variance"], float)
    Z = np.asarray(doc["inducing_inputs"], float)
    n, q = mu.shape
    if q > 3:
        raise SystemExit("Gauss-Hermite oracle limited to latent dimension 3")
    t, w = np.polynomial.hermite.hermgauss(nodes)
    w = w / np.sqrt(np.pi)
    grid = np.array(list(itertools.product(range(nodes), repeat=q)))
    T, W = t[grid], np.prod(w[grid], axis=1)            # (G, q), (G,)
    k = _kernel_fn(doc["kernel"])
    psi0, psi1, psi2 = np.zeros(n), np.zeros((n, len(Z))), np.zeros((len(Z), len(Z)))
    for i in range(n):
        X = mu[i] + np.sqrt(2 * S[i]) * T                 # (G, q)
        Kxz = k(X, Z)                                      # (G, m)
        psi1[i] = W @ Kxz
        psi2 += (Kxz * W[:, None]).T @ Kxz
        if doc["kernel"]["type"].endswith("rbf"):
            psi0[i] = doc["kernel"]["variance"]
        else:
            psi0[i] = W @ (doc["kernel"]["variance"] * (X ** 2).sum(1))
    return {"psi0": psi0, "psi1": psi1, "psi2": psi2}


def latent_reference(bench, doc):
    """Oracle reference for a latent case, refused if its independent cross-check fails."""
    from gp_scuderia import gpy_adapter
    kind = bench["kind"]
    if kind in ("latent_bound", "latent_timing"):
        ref = gpy_adapter.bound(doc); ref["oracle"] = "GPy"
    elif kind == "latent_collapse":
        mu = doc["latent"]["mean"]
        sparse = {"schema_version": "0.1", "type": "models.sparse_gp_regression", "input_dim": doc["latent"]["dim"],
                  "output_dim": len(doc["data"]["Y"][0]), "approximation": "dtcvar",
                  "inducing_inputs": doc["inducing_inputs"], "kernel": doc["kernel"], "likelihood": doc["likelihood"],
                  "data": {"X": mu, "Y": doc["data"]["Y"]}}
        gpy_adapter.KUU_JITTER = 1e-6   # the latent path uses 1e-6 on every GPy release
        try:
            ll = gpy_adapter.reference(sparse, [[0.0] * doc["latent"]["dim"]])["log_likelihood"]
        finally:
            gpy_adapter.KUU_JITTER = None
        own = gpy_adapter.bound(doc)["bound_without_kl"]
        if abs(own - ll) > 1e-6:
            raise SystemExit("GPy's latent bound %.10g differs from its sparse GP %.10g in the collapse limit" % (own, ll))
        ref = {"bound_without_kl": ll, "oracle": "GPy sparse GP (dtcvar) at inputs mu; no psi statistic involved"}
    elif kind == "latent_gradient":
        a, fd = gpy_adapter.gradient(doc), gpy_adapter.gradient_fd(doc)
        worst, why = gradient_deviation(a, fd, bench["tolerance"]["gradient"])   # 1.0 is the benchmark tolerance
        if why or worst > 0.1:
            raise SystemExit("GPy's latent gradient disagrees with finite differences (%.2g of the tolerance; %s) for %s"
                             % (worst, why, doc.get("x-descriptors")))
        ref = {"gradient": a, "gradient_fd_deviation_over_tolerance": worst,
               "oracle": "GPy analytic gradient of bound_without_kl - kl, checked against central differences"}
    else:
        gh, gp = gauss_hermite_psi(doc), gpy_adapter.psi(doc)
        worst = max(float(np.max(np.abs(gh[k] - np.asarray(gp[k])))) for k in gh)
        if worst > 1e-7:
            raise SystemExit("GPy psi statistics differ from Gauss-Hermite by %.3g" % worst)
        ref = {k: v.tolist() for k, v in gh.items()}
        ref.update(oracle="Gauss-Hermite quadrature, 48 nodes per dimension (cross-checked against GPy to 1e-7)",
                   oracle_gpy_maxabs=worst)
    ref["tolerance"] = bench["tolerance"]
    return ref


def latent_deviation(kind, g, ref):
    """{quantity: worst absolute deviation} between a framework's result and the reference."""
    if kind == "psi":
        p = g["psi"]
        return {k: float(np.max(np.abs(np.asarray(p[k], float) - np.asarray(ref[k], float)))) for k in ("psi0", "psi1", "psi2")}
    return {k: abs(g["bound"][k] - ref[k]) for k in ("bound_without_kl", "kl") if k in ref}


def generate(out, tier=None):
    import jsonschema
    from gp_scuderia import gpy_adapter
    schema = json.loads((root / "schemas" / "model.schema.json").read_text())
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    for bench in benchmarks():
        for i, case in enumerate(bench["cases"]):
            if tier and "tiers" in case and tier not in case["tiers"]:
                continue
            if bench.get("kind") in LATENT_KINDS:
                doc = make_latent_model(bench, case, i)
                jsonschema.validate(doc, schema)
                doc["x-descriptors"] = latent_descriptors(doc, {"benchmark": bench["name"], "case": case["id"],
                                                                "seed": [bench["seed"], i]})
                doc["x-benchmark-kind"] = bench["kind"]
                doc["x-reference"] = latent_reference(bench, doc)
                p = out / ("%s__%s.yaml" % (bench["name"], case["id"]))
                dump(doc, p); to_json_twin(p)
                print("generated", p.stem)
                continue
            doc, Xnew = make_model(bench, case, i)
            jsonschema.validate(doc, schema)
            doc["x-descriptors"] = descriptors(doc, {"benchmark": bench["name"], "case": case["id"],
                                                     "seed": [bench["seed"], i]})
            doc["x-benchmark-kind"] = bench.get("kind", "accuracy")
            if "tiers" in case:
                doc["x-tiers"] = case["tiers"]
            ref = gpy_adapter.reference(doc, Xnew)
            ref.update(oracle="GPy", oracle_features=gpy_adapter.FEATURES, tolerance=bench["tolerance"])
            if bench.get("kind") == "gradient":
                ref["gradient"], ref["gradient_fd_maxrel"] = oracle_gradient(doc, Xnew)
            doc["x-reference"] = ref
            p = out / ("%s__%s.yaml" % (bench["name"], case["id"]))
            dump(doc, p); to_json_twin(p)
            print("generated", p.stem)


def hardware():
    """What the timings ran on. numpy_blas is the BLAS of this interpreter, not necessarily of the runner."""
    blas = None
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:
        pass
    return {"machine": platform.machine(), "processor": platform.processor(), "system": platform.platform(),
            "logical_cpus": os.cpu_count(), "numpy_blas": blas, "dtype": "float64"}


def timing(fw, data, tier, threads, json_out):
    """Time every timing case the framework can run. Never fails the build on speed; a case that
    errors is recorded as an error, and the exit code is 0 unless the runner could not start."""
    framework = F.load_frameworks()[fw]
    rn = framework["runners"][0]
    data = pathlib.Path(data).resolve()
    TIMED = ("timing", "latent_timing")
    specs = {b["name"]: b for b in benchmarks() if b.get("kind") in TIMED}
    docs = {p.stem: load(p) for p in sorted(data.glob("*.yaml"))}
    names = [n for n, d in docs.items() if d.get("x-benchmark-kind") in TIMED and tier in d.get("x-tiers", ["scheduled"])
             or (d.get("x-benchmark-kind") in TIMED and tier == "scheduled")]

    def why_not(n):
        why = F.missing(framework["_caps"], docs[n])
        kind = docs[n]["x-benchmark-kind"]
        if not why and kind in LATENT_KINDS:
            why = latent_gate(framework, kind)
        return why
    runnable = [n for n in names if not why_not(n)]
    extra, t0 = {}, time.time()
    tasks = [dict({"name": n, "golden": str(data / (n + ".json")), "timing": specs[n.split("__")[0]]["timing"]},
                  **({"operations": ["bound"]} if docs[n]["x-benchmark-kind"] == "latent_timing" else {})) for n in runnable]
    raw = launch(fw, rn, tasks, extra, threads=1 if threads == "1" else None) if tasks else {}
    results = []
    for n in names:
        d = docs[n]; ref = d["x-reference"]; bname, cid = n.split("__")
        latent = d["x-benchmark-kind"] == "latent_timing"
        rec = {"benchmark": bname, "case": cid, "name": n, "n": d["x-descriptors"]["n"],
               "input_dim": d["latent"]["dim"] if latent else d["input_dim"], "approximation": d["approximation"],
               "inducing": len(d["inducing_inputs"]) if "inducing_inputs" in d else None}
        why = why_not(n)
        if why:
            rec.update(status="skipped", reason=str(why))
        else:
            g = raw[n]
            if "refused" in g:
                rec.update(status="refused", reason=g["refused"])
            elif "error" in g:
                rec.update(status="error", reason=g["error"][:200])
            else:
                tol = ref["tolerance"]
                if latent:
                    dev = float(abs(g["bound"]["bound_without_kl"] - ref["bound_without_kl"]))
                    sane_tol = tol["bound_without_kl"]
                else:
                    dev = float(abs(g["log_likelihood"] - ref["log_likelihood"]))
                    sane_tol = tol["log_likelihood"]
                rec.update(status="timed", **g["timing"], ll_deviation=dev, sane=dev <= sane_tol)
        results.append(rec)
        print("%-32s %-8s %s" % (n, rec["status"], ("median %.4gs cold %.4gs%s" % (
              rec["seconds_median"], rec["seconds_cold"], "" if rec["sane"] else "  LL DISAGREES"))
              if rec["status"] == "timed" else rec.get("reason", "")[:70]))
    doc = {"timing_run": True, "benchmark_framework": fw, "tier": tier, "threads": threads,
           "thread_env": THREAD_VARS if threads == "1" else None, "hardware": hardware(),
           "python": platform.python_version(), "versions": {**versions(rn), **extra},
           "seconds": round(time.time() - t0, 2), "results": results}
    if json_out:
        pathlib.Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(json_out).write_text(json.dumps(doc, indent=2) + "\n")


def run(fw, data, mode, json_out):
    framework = F.load_frameworks()[fw]
    rn = framework["runners"][0]
    data = pathlib.Path(data).resolve()
    names = sorted(p.stem for p in data.glob("*.yaml"))
    docs = {n: load(data / (n + ".yaml")) for n in names}
    oracle_features = set()      # what the GPy that wrote these goldens could be told (same for every golden)
    for d in docs.values():
        oracle_features |= set(d.get("x-reference", {}).get("oracle_features", []))
    known = {}
    for b in benchmarks():
        for f, cases in (b.get("known") or {}).items():
            for cid, why in cases.items():
                if isinstance(why, dict):        # {why: ..., fixed_by: feature}: moot once the oracle has the feature
                    if why.get("fixed_by") in oracle_features:
                        continue
                    why = why["why"]
                known[(f, "%s__%s" % (b["name"], cid))] = why
    names = [n for n in names if docs[n].get("x-benchmark-kind") not in ("timing", "latent_timing")]
    def gradient_kind(n):
        return docs[n].get("x-benchmark-kind") == "gradient"

    def skip_reason(n):
        why = F.missing(framework["_caps"], docs[n])
        if not why and gradient_kind(n) and "gradient" not in framework.get("operations", []):
            why = ["the framework file declares no gradient operation"]
        kind = docs[n].get("x-benchmark-kind")
        if not why and kind in LATENT_KINDS:
            why = latent_gate(framework, kind)
        return why
    runnable = [n for n in names if not skip_reason(n)]
    extra = {}
    t0 = time.time()
    raw = launch(fw, rn, [dict({"name": n, "golden": str(data / (n + ".json"))},
                               **({"operations": ["gradient"]} if gradient_kind(n) else
                                  {"operations": [LATENT_KINDS[docs[n]["x-benchmark-kind"]]]} if docs[n].get("x-benchmark-kind") in LATENT_KINDS else {})) for n in runnable], extra) if runnable else {}
    results, bad = [], 0
    for n in names:
        ref = docs[n]["x-reference"]; tol = ref["tolerance"]
        rec = {"benchmark": n.split("__")[0], "case": n.split("__")[1], "name": n,
               "kind": docs[n].get("x-benchmark-kind", "accuracy")}
        why = skip_reason(n)
        if why:
            rec.update(status="skipped", reason=str(why), ok=True)
        else:
            g = raw[n]
            if "refused" in g:
                rec.update(status="refused", reason=g["refused"])
            elif "error" in g:
                rec.update(status="error", reason=g["error"])
            elif docs[n].get("x-benchmark-kind") == "latent_gradient":
                worst, why_not = gradient_deviation(g["gradient"], ref["gradient"], tol["gradient"])
                rec.update(deviation={"gradient": worst}, tolerance={"gradient": 1.0},
                           status="match" if worst <= 1.0 else "mismatch")
                if why_not:
                    rec["detail"] = why_not
            elif docs[n].get("x-benchmark-kind") in LATENT_KINDS:
                dev = latent_deviation(docs[n]["x-benchmark-kind"], g, ref)
                tl = {k: tol.get(k, tol.get("psi")) for k in dev}
                rec.update(deviation=dev, tolerance=tl, status="match" if all(dev[k] <= tl[k] for k in dev) else "mismatch")
            elif gradient_kind(n):
                worst, why_not = gradient_deviation(g["gradient"], ref["gradient"], tol["gradient"])
                rec.update(deviation={"gradient": worst}, tolerance={"gradient": 1.0},
                           status="match" if worst <= 1.0 else "mismatch")
                if why_not:
                    rec["detail"] = why_not
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
    sparse = {}
    for b in benchmarks():
        if "properties" not in b:
            continue
        lls = {}
        for r_ in results:
            if r_["benchmark"] == b["name"]:
                g = raw.get(r_["name"])
                lls[r_["case"]] = g["log_likelihood"] if g and "log_likelihood" in g else None
        recs, table = check_properties(b, lls)
        for r_ in recs:
            k = known.get((fw, r_["name"]))
            if k and r_["status"] == "mismatch":
                r_.update(status="known", reason=k, ok=True)
            elif k and r_["status"] == "match" and k.startswith("intermittent:"):
                r_.update(reason="known intermittent issue, did not occur on this run")
            elif k and r_["status"] == "match":
                r_.update(ok=False, reason="known failure now holds; remove it from the benchmark")
            bad += not r_["ok"]
            print("%-5s %-34s %-8s %s" % ("ok" if r_["ok"] else "FAIL", r_["name"], r_["status"], r_["reason"][:80]))
        results.extend(recs); sparse[b["name"]] = table
    doc = {"benchmark_run": True, "framework": None, "benchmark_framework": fw, "mode": mode,
           "python": platform.python_version(), "versions": {**versions(rn), **extra},
           "seconds": round(time.time() - t0, 2), "results": results, "sparse": sparse}
    del doc["framework"]
    if json_out:
        pathlib.Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(json_out).write_text(json.dumps(doc, indent=2) + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--out", required=True); g.add_argument("--tier")
    t = sub.add_parser("time"); t.add_argument("framework"); t.add_argument("--data", required=True)
    t.add_argument("--tier", default="pr", choices=["pr", "scheduled"])
    t.add_argument("--threads", default="all", choices=["1", "all"]); t.add_argument("--json")
    r = sub.add_parser("run"); r.add_argument("framework"); r.add_argument("--data", required=True)
    r.add_argument("--mode", default="pinned"); r.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.out, a.tier)
    elif a.cmd == "time":
        timing(a.framework, a.data, a.tier, a.threads, a.json)
    else:
        run(a.framework, a.data, a.mode, a.json)
