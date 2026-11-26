"""Independent check of the Laplace goldens: a from-scratch numpy implementation.

Rasmussen and Williams (2006) Alg 3.1 (mode), eq 3.32 (log marginal), Alg 3.2 (prediction).
No GPy code is used, so agreement with x-reference is evidence that the golden is the
Laplace approximation and not an artefact of one library. RBF/Matern kernels only.
Run: .venv-gpy/bin/python scripts/check_laplace_independent.py
"""
import pathlib, sys
import numpy as np
from scipy.special import gammaln
from scipy.stats import norm
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import load


def kernel(node, A, B):
    t = node["type"]
    if t == "kern.add":
        return sum(kernel(p, A, B) for p in node["parts"])
    if t == "kern.prod":
        out = 1.0
        for p in node["parts"]:
            out = out * kernel(p, A, B)
        return out
    ls = np.asarray(node["lengthscale"], float) * np.ones(A.shape[1])
    r = np.sqrt((((A[:, None, :] - B[None, :, :]) / ls) ** 2).sum(-1))
    v = node["variance"]
    return {"kern.rbf": lambda: v * np.exp(-0.5 * r ** 2),
            "kern.matern32": lambda: v * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r),
            "kern.matern52": lambda: v * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r)}[t]()


def bernoulli(f, y):
    z = (2 * y - 1) * f
    lp = norm.logcdf(z); r = np.exp(norm.logpdf(z) - lp)
    return lp, (2 * y - 1) * r, -(r ** 2 + z * r)


def poisson(f, y):
    return y * f - np.exp(f) - gammaln(y + 1), y - np.exp(f), -np.exp(f)


def laplace(doc, Xnew):
    X = np.array(doc["data"]["X"], float); y = np.array(doc["data"]["Y"], float).ravel()
    lik = {"likelihoods.bernoulli": bernoulli, "likelihoods.poisson": poisson}[doc["likelihood"]["type"]]
    K = kernel(doc["kernel"], X, X); n = len(y); f = np.zeros(n)
    for _ in range(500):
        _, g, h = lik(f, y); W = -h; sW = np.sqrt(W)
        L = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None])
        b = W * f + g
        a = b - sW * np.linalg.solve(L.T, np.linalg.solve(L, sW * (K @ b)))
        fn = K @ a
        done = np.abs(fn - f).max() < 1e-14
        f = fn
        if done:
            break
    lp, g, h = lik(f, y); W = -h; sW = np.sqrt(W)
    L = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None])
    ll = lp.sum() - 0.5 * f @ np.linalg.solve(K, f) - np.log(np.diag(L)).sum()
    Ks = kernel(doc["kernel"], Xnew, X)
    mu = Ks @ g
    v = np.linalg.solve(L, sW[:, None] * Ks.T)
    s2 = np.diag(kernel(doc["kernel"], Xnew, Xnew)) - (v ** 2).sum(0)
    if doc["likelihood"]["type"] == "likelihoods.bernoulli":
        p = norm.cdf(mu / np.sqrt(1 + s2)); mean, var = p, p * (1 - p)
    else:
        mean = np.exp(mu + s2 / 2); var = mean + mean ** 2 * np.expm1(s2)
    return ll, mu, s2, mean, var


bad = 0
for p in sorted((root / "examples").glob("laplace_*.yaml")):
    doc = load(p); ref = doc["x-reference"]; tol = ref["tolerance"]
    ll, mu, s2, mean, var = laplace(doc, np.array(ref["Xnew"], float))
    d = [abs(ll - ref["log_likelihood"]), np.abs(mu - ref["latent_mean"]).max(),
         np.abs(s2 - ref["latent_var"]).max(), np.abs(mean - ref["mean"]).max(),
         np.abs(var - ref["var"]).max()]
    ok = d[0] <= tol["log_likelihood"] and max(d[1], d[3]) <= tol["mean"] and max(d[2], d[4]) <= tol["var"]
    print(("ok   " if ok else "FAIL ") + "%s: independent numpy Laplace matches (dLL=%.2g dmu=%.2g ds2=%.2g dmean=%.2g dvar=%.2g)"
          % ((p.stem,) + tuple(d)))
    bad += (not ok)
sys.exit(bad > 0)
