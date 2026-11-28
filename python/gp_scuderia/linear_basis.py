"""Framework-independent reference for mean.linear_basis (CIP-0009).

Single output, exact GP. With K_y = K + noise I and A = F' K_y^-1 F, the four
beta modes give

  fixed           log N(y; F b,    K_y)
  profiled        log N(y; F bhat, K_y)                       bhat = A^-1 F' K_y^-1 y
  flat_prior      profiled - 0.5 log|A|                        (-(p/2) log v dropped)
  gaussian_prior  log N(y; F m0,   K_y + F S0 F')

Predictions are for the latent function f(x*). Add the noise variance for
observations, as the wire runner contract says.
"""
import numpy as np


def polynomial_basis(X, powers):
    """Columns prod_d X[:, d] ** p_d, one per multi-index in `powers`."""
    X = np.atleast_2d(np.asarray(X, float))
    P = np.asarray(powers, int)
    if P.shape[1] != X.shape[1]:
        raise ValueError("powers width %d != input_dim %d" % (P.shape[1], X.shape[1]))
    return np.prod(X[:, None, :] ** P[None, :, :], axis=2)


def _logdet_chol(L):
    return 2.0 * np.log(np.diag(L)).sum()


def _gauss_logpdf(r, L):
    """log N(r; 0, L L') for a residual vector r."""
    z = np.linalg.solve(L, r)
    return -0.5 * (z @ z) - 0.5 * _logdet_chol(L) - 0.5 * len(r) * np.log(2 * np.pi)


def reference(K, noise, y, F, mode, Fs, Ks, kss_diag, value=None, prior_mean=None, prior_cov=None):
    """Log likelihood and latent predictive moments.

    K: n x n kernel, noise: variance, y: n, F: n x p, Fs: m x p basis at test inputs,
    Ks: n x m cross-covariance, kss_diag: m prior variances at test inputs.
    """
    y = np.asarray(y, float).ravel()
    n, p = F.shape
    Ky = K + noise * np.eye(n)
    L = np.linalg.cholesky(Ky)
    solve = lambda B: np.linalg.solve(L.T, np.linalg.solve(L, B))
    KiF, Kiy = solve(F), solve(y)
    A = F.T @ KiF
    # Extra predictive covariance factor: r = Fs' - F' K_y^-1 Ks, one column per test point.
    R = Fs.T - F.T @ solve(Ks)
    if mode == "fixed":
        beta, post_cov, ll = np.asarray(value, float), None, _gauss_logpdf(y - F @ value, L)
    elif mode in ("profiled", "flat_prior"):
        beta = np.linalg.solve(A, F.T @ Kiy)
        ll = _gauss_logpdf(y - F @ beta, L)
        post_cov = np.linalg.inv(A)
        if mode == "flat_prior":
            ll -= 0.5 * np.linalg.slogdet(A)[1]
    elif mode == "gaussian_prior":
        m0, S0 = np.asarray(prior_mean, float), np.asarray(prior_cov, float)
        ll = _gauss_logpdf(y - F @ m0, np.linalg.cholesky(Ky + F @ S0 @ F.T))
        post_cov = np.linalg.inv(A + np.linalg.inv(S0))
        beta = post_cov @ (F.T @ Kiy + np.linalg.solve(S0, m0))
    else:
        raise ValueError("unknown beta mode %r" % mode)
    mean = Fs @ beta + Ks.T @ (Kiy - KiF @ beta)
    var = kss_diag - np.einsum("im,im->m", Ks, solve(Ks))
    if post_cov is not None:
        var = var + np.einsum("im,ij,jm->m", R, post_cov, R)
    return {"log_likelihood": float(ll), "mean": mean, "var": var, "beta": beta}


def _rbf(A, B, variance, lengthscale):
    sq = ((np.asarray(A)[:, None, :] - np.asarray(B)[None, :, :]) ** 2).sum(-1)
    return variance * np.exp(-0.5 * sq / lengthscale ** 2)


def reference_from_doc(doc, Xnew):
    """Reference for a wire document: exact GP, one non-ARD kern.rbf, Gaussian likelihood.

    Returns the wire runner quantities: log likelihood, predictive mean, and the
    predictive variance of y (latent variance plus the noise variance).
    """
    k, lik, mean = doc["kernel"], doc["likelihood"], doc["mean"]
    if (k["type"] != "kern.rbf" or k.get("ARD") or lik["type"] != "likelihoods.gaussian"
            or doc.get("approximation", "ftc") != "ftc" or doc["output_dim"] != 1
            or mean["type"] != "mean.linear_basis"):
        raise NotImplementedError("reference_from_doc covers exact single-output rbf regression only")
    X = np.asarray(doc["data"]["X"], float)
    Y = np.asarray(doc["data"]["Y"], float).ravel()
    Xs = np.asarray(Xnew, float)
    powers = mean["basis"]["powers"]
    beta = mean["beta"]
    r = reference(_rbf(X, X, k["variance"], k["lengthscale"]), lik["variance"], Y,
                  polynomial_basis(X, powers), beta["mode"], polynomial_basis(Xs, powers),
                  _rbf(X, Xs, k["variance"], k["lengthscale"]), np.full(len(Xs), k["variance"]),
                  value=beta.get("value"), prior_mean=beta.get("prior_mean"), prior_cov=beta.get("prior_cov"))
    return {"log_likelihood": r["log_likelihood"], "Xnew": Xs.tolist(),
            "mean": r["mean"].tolist(), "var": (r["var"] + lik["variance"]).tolist(), "beta": r["beta"].tolist()}
