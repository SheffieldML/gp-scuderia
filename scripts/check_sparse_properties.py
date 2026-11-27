"""Check that the sparse property checks can fail (CIP-0005).

Numpy implementations of the exact, DTCVar and trace-dropped DTC log likelihoods on a small
problem, fed through check_properties. The correct DTCVar must pass; DTC without its trace term,
the bug the lower-bound check exists to catch, must fail; a DTCVar that falls as the inducing set
grows must fail the monotonicity check.
"""
import pathlib, sys
import numpy as np, yaml

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root / "python"))
from benchmarks import check_properties, _stationary_K

bench = yaml.safe_load((root / "benchmarks" / "sparse_properties.yaml").read_text())
bench["properties"]["sizes"] = [2, 4, 8, 16]
rng = np.random.default_rng(0)
n, s2 = 60, 0.3
X = rng.uniform(-3, 3, (n, 1)); Y = np.sin(X) + 0.3 * rng.standard_normal((n, 1))
kern = {"type": "kern.rbf", "variance": 1.0, "lengthscale": 1.0}
perm = rng.permutation(n)


def gauss(C):
    L = np.linalg.cholesky(C); a = np.linalg.solve(L, Y)
    return -0.5 * (a ** 2).sum() - np.log(np.diag(L)).sum() - 0.5 * n * np.log(2 * np.pi)


def sparse(m, trace):
    Z = X[perm[:m]]
    K = _stationary_K(kern, np.vstack([Z, X])); Kuu, Kuf = K[:m, :m] + 1e-8 * np.eye(m), K[:m, m:]
    Q = Kuf.T @ np.linalg.solve(Kuu, Kuf)
    ll = gauss(Q + s2 * np.eye(n))
    return ll - 0.5 * (n - np.trace(Q)) / s2 if trace else ll


exact = gauss(_stationary_K(kern, X) + s2 * np.eye(n))
sizes = bench["properties"]["sizes"]
good = {"exact": exact, **{"dtcvar_%d" % m: sparse(m, True) for m in sizes}}
dtc = {"exact": exact, **{"dtcvar_%d" % m: sparse(m, False) for m in sizes}}
fall = dict(good); fall["dtcvar_16"] = good["dtcvar_8"] - 1.0

bad = 0
def expect(label, lls, bound, mono):
    global bad
    recs, _ = check_properties(bench, lls)
    got = {r["case"]: r["status"] for r in recs}
    ok = got["dtcvar_lower_bound"] == bound and got["dtcvar_monotone_in_M"] == mono
    print(("ok   " if ok else "FAIL ") + label, got); bad += not ok

expect("correct DTCVar passes both", good, "match", "match")
expect("DTC with the trace term dropped fails the bound", dtc, "mismatch", "match" if all(dtc["dtcvar_%d" % b] >= dtc["dtcvar_%d" % a] for a, b in zip(sizes, sizes[1:])) else "mismatch")
expect("DTCVar that falls with M fails monotonicity", fall, "match", "mismatch")
expect("missing exact case is skipped, not passed", {k: v for k, v in good.items() if k != "exact"}, "skipped", "match")
sys.exit(1 if bad else 0)
