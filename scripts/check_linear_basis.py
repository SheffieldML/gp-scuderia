#!/usr/bin/env python3
"""Property checks for the mean.linear_basis reference (CIP-0009). Needs numpy only.

Each relation is checked against a different computation than the one the
reference uses, and a deliberately broken reference must fail.
"""
import pathlib, sys
import numpy as np
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import linear_basis as lb

rng = np.random.default_rng(3)
n, m, d = 9, 4, 2
X, Xs = rng.normal(size=(n, d)), rng.normal(size=(m, d))
y = rng.normal(size=n)
noise = 0.15
powers = [[0, 0], [1, 0], [0, 1], [1, 1]]
F, Fs = lb.polynomial_basis(X, powers), lb.polynomial_basis(Xs, powers)
p = F.shape[1]


def rbf(A, B, var=1.3, ell=0.9):
    return var * np.exp(-0.5 * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / ell ** 2)


K, Ks, kss = rbf(X, X), rbf(X, Xs), np.full(m, 1.3)
args = (K, noise, y, F)
tail = (Fs, Ks, kss)
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (" " + detail if detail else ""))
    if not ok:
        fails.append(name)


def gp_direct(Kfull, mean_n, mean_s):
    """Zero-mean-GP prediction written without the reference's pieces."""
    Ky = Kfull + noise * np.eye(n)
    Ki = np.linalg.inv(Ky)
    mu = mean_s + Ks.T @ Ki @ (y - mean_n)
    var = kss - np.einsum("im,ij,jm->m", Ks, Ki, Ks)
    sign, logdet = np.linalg.slogdet(Ky)
    r = y - mean_n
    return -0.5 * r @ Ki @ r - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi), mu, var


def run_all(ref):
    # fixed: plain GP with mean F b
    b = rng.normal(size=p)
    r = ref(*args, "fixed", *tail, value=b)
    ll, mu, var = gp_direct(K, F @ b, Fs @ b)
    check("fixed matches a GP with mean F b",
          abs(r["log_likelihood"] - ll) < 1e-10 and np.allclose(r["mean"], mu) and np.allclose(r["var"], var))

    # gaussian_prior: GP with kernel K + F S0 F' and mean F m0 (cross-covariance gains Fs S0 F')
    m0 = rng.normal(size=p)
    B = rng.normal(size=(p, p)); S0 = B @ B.T + 0.5 * np.eye(p)
    g = ref(*args, "gaussian_prior", *tail, prior_mean=m0, prior_cov=S0)
    Kfull = K + F @ S0 @ F.T
    Ky = Kfull + noise * np.eye(n)
    Ksf = Ks + F @ S0 @ Fs.T
    Ki = np.linalg.inv(Ky)
    mu = Fs @ m0 + Ksf.T @ Ki @ (y - F @ m0)
    var = kss + np.einsum("mp,pq,mq->m", Fs, S0, Fs) - np.einsum("im,ij,jm->m", Ksf, Ki, Ksf)
    ll, _, _ = gp_direct(Kfull, F @ m0, Fs @ m0)
    check("gaussian_prior matches the GP with kernel K + F S0 F'",
          abs(g["log_likelihood"] - ll) < 1e-9 and np.allclose(g["mean"], mu) and np.allclose(g["var"], var),
          "dmean=%.1e dvar=%.1e" % (np.abs(g["mean"] - mu).max(), np.abs(g["var"] - var).max()))

    # profiled: beta-hat maximises the likelihood over beta, and the gradient vanishes there
    pr = ref(*args, "profiled", *tail)
    Kyi = np.linalg.inv(K + noise * np.eye(n))
    grad = F.T @ Kyi @ (y - F @ pr["beta"])
    check("profiled beta is a stationary point of the likelihood", np.abs(grad).max() < 1e-9)
    worse = [ref(*args, "fixed", *tail, value=pr["beta"] + 0.05 * rng.normal(size=p))["log_likelihood"]
             for _ in range(20)]
    check("profiled log likelihood is the maximum over beta", all(w < pr["log_likelihood"] for w in worse))

    # flat_prior: same predictions, log likelihood lower by 0.5 log|A|
    fl = ref(*args, "flat_prior", *tail)
    A = F.T @ Kyi @ F
    check("profiled and flat_prior predict identically",
          np.allclose(pr["mean"], fl["mean"]) and np.allclose(pr["var"], fl["var"]))
    check("flat_prior = profiled - 0.5 log|A|",
          abs(fl["log_likelihood"] - (pr["log_likelihood"] - 0.5 * np.linalg.slogdet(A)[1])) < 1e-10)

    # flat_prior is the large-variance limit of gaussian_prior once -(p/2) log v is restored
    for v in (1e4, 1e6):
        gv = ref(*args, "gaussian_prior", *tail, prior_mean=np.zeros(p), prior_cov=v * np.eye(p))
        err = abs(gv["log_likelihood"] + 0.5 * p * np.log(v) - fl["log_likelihood"])
        check("gaussian_prior(v=%g) -> flat_prior as v grows" % v, err < 20.0 / v * np.trace(np.linalg.inv(A)) + 1e-6,
              "err=%.2e" % err)
    big = ref(*args, "gaussian_prior", *tail, prior_mean=np.zeros(p), prior_cov=1e8 * np.eye(p))
    check("gaussian_prior with huge variance predicts like flat_prior",
          np.allclose(big["mean"], fl["mean"], atol=1e-5) and np.allclose(big["var"], fl["var"], atol=1e-5))

    # universal kriging term: profiled variance exceeds the zero-uncertainty (fixed) variance
    check("estimated beta inflates predictive variance", bool(np.all(pr["var"] > fixed_var(ref, pr["beta"]) + 1e-12)))


def fixed_var(ref, beta):
    return ref(*args, "fixed", *tail, value=beta)["var"]


def broken(*a, **k):
    """Reference that forgets the universal kriging variance term."""
    out = lb.reference(*a, **k)
    if a[4] != "fixed" and a[4] != "gaussian_prior":
        out = dict(out, var=fixed_var(lb.reference, out["beta"]))
    return out


print("== reference")
run_all(lb.reference)
real_fails = list(fails)
fails.clear()
print("== deliberately broken reference (must fail)")
golden_fail = False
# The committed goldens reproduce their own x-reference.
from gp_scuderia import load
for gpath in sorted((pathlib.Path(__file__).resolve().parents[1] / "examples").glob("ftc_linear_*.yaml")):
    doc = load(gpath); ref = doc["x-reference"]; r = lb.reference_from_doc(doc, ref["Xnew"])
    ok = (abs(r["log_likelihood"] - ref["log_likelihood"]) < 1e-9
          and np.abs(np.array(r["mean"]) - ref["mean"]).max() < 1e-9
          and np.abs(np.array(r["var"]) - ref["var"]).max() < 1e-9)
    print(("ok   " if ok else "FAIL ") + gpath.stem + " reproduces its x-reference")
    if not ok: golden_fail = True

import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    run_all(broken)
print(buf.getvalue(), end="")
caught = bool(fails)
print("ok   broken reference is caught" if caught else "FAIL broken reference passed the checks")
sys.exit(0 if not real_fails and caught and not golden_fail else 1)

