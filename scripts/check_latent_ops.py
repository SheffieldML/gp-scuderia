"""Check the latent-variable runner operations bound and psi (CIP-0006), with GPy only.

1. Contract /3 task and result files validate; /1 and /2 files still do.
2. GPy's psi statistics agree with Monte Carlo expectations of the kernel under q(X)
   (a check that does not use GPy's own psi code), and with K(mu, Z) as S -> 0.
3. bound_without_kl at S -> 0 equals the sparse dtcvar log likelihood at inputs mu.
4. An adapter without the operation yields a refusal, and bound/psi must be requested alone.
"""
import json, pathlib, sys
import numpy as np, yaml, jsonschema

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import run, gpy_adapter as A

task_s = json.loads((root / "schemas/task.schema.json").read_text())
res_s = json.loads((root / "schemas/result.schema.json").read_text())
path = root / "examples/latent/bgplvm_rbf_ard.yaml"
doc = yaml.safe_load(path.read_text())
bad = 0


def check(label, ok):
    global bad
    print(("ok   " if ok else "FAIL ") + label)
    bad += not ok


for v, ops in ((1, None), (2, ["gradient"]), (3, ["bound"]), (3, ["psi"])):
    t = {"name": "a", "golden": "a.json"}
    if ops:
        t["operations"] = ops
    jsonschema.validate({"schema": "gp-scuderia-task/%d" % v, "tasks": [t]}, task_s)
check("task files /1 /2 /3 validate", True)

out = {}
for op in ("bound", "psi"):
    out[op] = run.run_tasks(A.reference, [{"name": "g", "golden": str(path), "operations": [op]}],
                            A.gradient, A.bound, A.psi)["g"]
    jsonschema.validate({"schema": "gp-scuderia-result/3", "results": {"g": out[op]}}, res_s)
check("results validate against /3", True)
for op in out:
    assert op in out[op], out[op].get("error")
b, p = out["bound"]["bound"], out["psi"]["psi"]

rng = np.random.default_rng(0)
mu, S, Z = (np.array(x, float) for x in (doc["latent"]["mean"], doc["latent"]["variance"], doc["inducing_inputs"]))
var, ls = doc["kernel"]["variance"], np.array(doc["kernel"]["lengthscale"], float)


def k(a, b_):
    return var * np.exp(-0.5 * (((a[:, None, :] - b_[None]) / ls) ** 2).sum(-1))


n, m, N = len(mu), len(Z), 200000
ps1, ps2 = np.zeros((n, m)), np.zeros((m, m))
for i in range(n):
    x = mu[i] + np.sqrt(S[i]) * rng.standard_normal((N, mu.shape[1]))
    kz = k(x, Z)
    ps1[i] = kz.mean(0)
    ps2 += kz.T @ kz / N
check("psi0 equals the kernel variance per point", np.allclose(p["psi0"], var))
check("psi1 matches Monte Carlo (%.1e)" % np.abs(np.array(p["psi1"]) - ps1).max(), np.allclose(p["psi1"], ps1, rtol=5e-3, atol=5e-3))
check("psi2 matches Monte Carlo (%.1e)" % np.abs(np.array(p["psi2"]) - ps2).max(), np.allclose(p["psi2"], ps2, rtol=5e-3, atol=2e-2))

d0 = json.loads(json.dumps(doc))
d0["latent"]["variance"] = np.full_like(S, 1e-10).tolist()
check("psi1 -> K(mu, Z) as S -> 0", np.allclose(A.psi(d0)["psi1"], k(mu, Z), atol=1e-6))

sparse = {"schema_version": "0.1", "type": "models.gp_regression", "approximation": "dtcvar", "input_dim": mu.shape[1],
          "inducing_inputs": doc["inducing_inputs"], "kernel": doc["kernel"], "likelihood": doc["likelihood"],
          "data": {"X": mu.tolist(), "Y": doc["data"]["Y"]}}
b0 = A.bound(d0)["bound_without_kl"]
A.KUU_JITTER = 1e-6   # the latent path uses 1e-6 on every GPy release
ll = A.reference(sparse, mu.tolist())["log_likelihood"]
A.KUU_JITTER = None
check("bound_without_kl at S -> 0 equals dtcvar log likelihood (%.2e)" % abs(b0 - ll), abs(b0 - ll) < 1e-5)
check("kl is reported separately", abs(b["kl"] - 0.5 * np.sum(mu ** 2 + S - np.log(S) - 1)) < 1e-12)

ref, tol = doc["x-reference"], doc["x-reference"]["tolerance"]
check("golden bound_without_kl reproduces within %.0e" % tol["bound_without_kl"], abs(b["bound_without_kl"] - ref["bound_without_kl"]) <= tol["bound_without_kl"])
check("golden kl reproduces within %.0e" % tol["kl"], abs(b["kl"] - ref["kl"]) <= tol["kl"])
check("golden psi statistics reproduce within %.0e" % tol["psi"],
      all(np.allclose(p[k_], ref[k_], atol=tol["psi"], rtol=0) for k_ in ("psi0", "psi1", "psi2")))
check("golden is a JSON twin of the YAML", json.loads((root / "examples/latent/bgplvm_rbf_ard.json").read_text()) == doc)
d1 = json.loads(json.dumps(doc)); d1["kernel"]["variance"] *= 1.01   # a perturbed model must break the match
check("perturbing the kernel variance moves the bound beyond tolerance",
      abs(A.bound(d1)["bound_without_kl"] - ref["bound_without_kl"]) > tol["bound_without_kl"])

r = run.run_tasks(A.reference, [{"name": "g", "golden": str(path), "operations": ["bound"]}])["g"]
check("no bound operation -> refusal", "refused" in r)
r = run.run_tasks(A.reference, [{"name": "g", "golden": str(path), "operations": ["bound", "psi"]}], None, A.bound, A.psi)["g"]
check("bound and psi together -> refusal", "refused" in r)

# The benchmark comparison must fail on a wrong answer, or a passing run proves nothing.
sys.path.insert(0, str(root / "scripts"))
import benchmarks as B
bad_psi = {"psi": {k_: (np.asarray(v_) * 1.001).tolist() for k_, v_ in p.items()}}
dev = B.latent_deviation("psi", bad_psi, ref)
check("psi comparison flags a 0.1% error", all(dev[k_] > tol["psi"] for k_ in ("psi1", "psi2")))
dev = B.latent_deviation("latent_bound", {"bound": {"bound_without_kl": b["bound_without_kl"] + 1e-2, "kl": b["kl"]}}, ref)
check("bound comparison flags an error of 1e-2", dev["bound_without_kl"] > tol["bound_without_kl"] and dev["kl"] == 0)
# Latent gradient: analytic against central differences, and the log-space mistake is caught.
g = A.gradient(doc); fd = A.gradient_fd(doc)
gtol = {"atol": 1e-4, "rtol": 1e-3}
dv, _ = B.gradient_deviation(g, fd, gtol)
check("latent gradient agrees with central differences (%.2g of tolerance)" % dv, dv < 0.1)
check("gradient keys include latent.mean, latent.variance, inducing_inputs",
      {"latent.mean", "latent.variance", "inducing_inputs"} <= set(g))
S_ = np.array(doc["latent"]["variance"])
wrong = dict(g); wrong["latent.variance"] = (np.array(g["latent.variance"]) * S_).tolist()   # dL/dlog S
dv, _ = B.gradient_deviation(wrong, g, gtol)
check("gradient comparison flags dL/dlog S in place of dL/dS", dv > 1)
r = run.run_tasks(A.reference, [{"name": "g", "golden": str(path), "operations": ["gradient"]}], A.gradient)["g"]
check("runner returns the latent gradient", "gradient" in r and "latent.mean" in r["gradient"])

# a framework that declares only `gradient` (sparse regression) must not be sent latent documents
sys.path.insert(0, str(root / "scripts"))
import benchmarks as B
ok_all = ("latent_gradient", "psi", "latent_bound", "latent_timing")
gated = (all(B.latent_gate({"operations": ["gradient"]}, k) for k in ok_all)
         and not any(B.latent_gate({"operations": ["gradient", "bound", "psi"]}, k) for k in ok_all)
         and B.latent_gate({"operations": ["bound"]}, "psi") == ["the framework file declares no psi operation"])
check("latent benchmarks are gated on `bound` plus their own operation", gated)

sys.exit(1 if bad else 0)

