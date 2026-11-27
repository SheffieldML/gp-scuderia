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
path = root / "tests/valid/bgplvm_rbf_ard.yaml"
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
check("psi1 matches Monte Carlo (%.1e)" % np.abs(np.array(p["psi1"]) - ps1).max(), np.allclose(p["psi1"], ps1, atol=5e-3))
check("psi2 matches Monte Carlo (%.1e)" % np.abs(np.array(p["psi2"]) - ps2).max(), np.allclose(p["psi2"], ps2, atol=2e-2))

d0 = json.loads(json.dumps(doc))
d0["latent"]["variance"] = np.full_like(S, 1e-10).tolist()
check("psi1 -> K(mu, Z) as S -> 0", np.allclose(A.psi(d0)["psi1"], k(mu, Z), atol=1e-6))

sparse = {"schema_version": "0.1", "type": "models.gp_regression", "approximation": "dtcvar", "input_dim": mu.shape[1],
          "inducing_inputs": doc["inducing_inputs"], "kernel": doc["kernel"], "likelihood": doc["likelihood"],
          "data": {"X": mu.tolist(), "Y": doc["data"]["Y"]}}
b0 = A.bound(d0)["bound_without_kl"]
ll = A.reference(sparse, mu.tolist())["log_likelihood"]
check("bound_without_kl at S -> 0 equals dtcvar log likelihood (%.2e)" % abs(b0 - ll), abs(b0 - ll) < 1e-5)
check("kl is reported separately", abs(b["kl"] - 0.5 * np.sum(mu ** 2 + S - np.log(S) - 1)) < 1e-12)

r = run.run_tasks(A.reference, [{"name": "g", "golden": str(path), "operations": ["bound"]}])["g"]
check("no bound operation -> refusal", "refused" in r)
r = run.run_tasks(A.reference, [{"name": "g", "golden": str(path), "operations": ["bound", "psi"]}], None, A.bound, A.psi)["g"]
check("bound and psi together -> refusal", "refused" in r)
sys.exit(1 if bad else 0)
