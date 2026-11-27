"""GPJax adapter: wire document -> GPJax conjugate posterior, plus reference computation.

Needs float64 (set on import). GPJax adds Prior(jitter) to the Gram diagonal, so the
jitter is 1e-6 for the sparse bound (the GPmat convention, inside the golden
tolerances) and 0 for ftc, where 1e-6 would exceed the 1e-7 mean tolerance. `ftc` uses conjugate_mll; `dtcvar` uses collapsed_elbo.
FITC and coregionalisation are refused.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import gpjax as gpx
from ._common import (UnsupportedByFramework, check_single_output_ftc_or,
                      reference_from_transformed, transformed_Y)

JITTER = 1e-6
_LEAF = {"kern.rbf": gpx.kernels.RBF, "kern.matern32": gpx.kernels.Matern32,
         "kern.matern52": gpx.kernels.Matern52}


def _kern(node, d):
    t = node["type"]
    if t in ("kern.add", "kern.prod"):
        parts = [_kern(p, d) for p in node["parts"]]
        return (gpx.kernels.SumKernel if t == "kern.add" else gpx.kernels.ProductKernel)(kernels=parts)
    if t == "kern.coregionalize":
        raise UnsupportedByFramework("kern.coregionalize: not realised by the GPJax adapter")
    dims = node.get("active_dims", list(range(d)))
    if t in _LEAF:
        ls = node["lengthscale"]
        ls = jnp.array(ls if isinstance(ls, list) else [ls], dtype=jnp.float64)
        if len(dims) > 1 and ls.size == 1:
            ls = jnp.repeat(ls, len(dims))
        return _LEAF[t](active_dims=dims, lengthscale=ls, variance=jnp.asarray(node["variance"], dtype=jnp.float64))
    if t == "kern.white":
        return gpx.kernels.White(active_dims=dims, variance=jnp.asarray(node["variance"], dtype=jnp.float64))
    if t == "kern.bias":
        return gpx.kernels.Constant(active_dims=dims, constant=jnp.asarray(node["variance"], dtype=jnp.float64))
    if t == "kern.linear":
        return gpx.kernels.Linear(active_dims=dims, variance=jnp.asarray(node["variance"], dtype=jnp.float64))
    raise UnsupportedByFramework("GPJax adapter: %s" % t)


def from_wire(doc):
    approx = check_single_output_ftc_or(doc, ("ftc", "dtcvar"))
    X = jnp.array(doc["data"]["X"], dtype=jnp.float64)
    y = jnp.array(transformed_Y(doc), dtype=jnp.float64).reshape(-1, 1)
    D = gpx.Dataset(X=X, y=y)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(),
                          kernel=_kern(doc["kernel"], doc["input_dim"]), jitter=JITTER if approx == "dtcvar" else 0.0)
    lik = gpx.likelihoods.Gaussian(obs_stddev=jnp.sqrt(jnp.asarray(doc["likelihood"]["variance"], dtype=jnp.float64)))
    post = prior * lik
    if approx == "dtcvar":
        Z = jnp.array(doc["inducing_inputs"], dtype=jnp.float64)
        q = gpx.variational_families.CollapsedVariationalGaussian(model=post, inducing_inputs=Z)
        return post, D, q
    return post, D, None


def reference(doc, Xnew):
    post, D, q = from_wire(doc)
    Xn = jnp.array(Xnew, dtype=jnp.float64)
    if q is None:
        ll = gpx.objectives.conjugate_mll(post, D)
        dist = post.predict(Xn, D)
    else:
        ll = gpx.objectives.collapsed_elbo(q, D)
        dist = q.predict(Xn, D)
    # predict() returns the latent distribution; add observation noise for y
    noise = float(doc["likelihood"]["variance"])
    mean = np.asarray(dist.mean)
    var = np.asarray(jnp.diag(dist.covariance_matrix)) + noise
    return reference_from_transformed(doc, float(ll), mean, var, np.asarray(Xn))


def gradient(doc):
    """dLL/d(wire parameter) by jax.grad through from_wire itself.

    The wire values are passed as traced arguments, so the derivative includes GPJax's own
    parametrisation (the unconstrained bijection, and obs_stddev = sqrt(variance)) with no
    hand-written chain rule to get wrong.
    """
    from .params import wire_paths, node_at
    wire_paths(doc)
    theta = {p: jnp.asarray(v, dtype=jnp.float64) for p, v in wire_paths(doc)}

    def objective(theta):
        d = __import__("copy").deepcopy(doc)
        for p, v in theta.items():
            node, key = node_at(d, p)
            node[key] = list(v) if v.ndim else v
        post, D, q = from_wire(d)
        return gpx.objectives.conjugate_mll(post, D) if q is None else gpx.objectives.collapsed_elbo(q, D)

    g = jax.grad(objective)(theta)
    return {p: (np.asarray(v).tolist() if np.ndim(v) else float(v)) for p, v in g.items()}
