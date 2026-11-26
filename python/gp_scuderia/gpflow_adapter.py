"""GPflow adapter: wire document -> GPflow model, plus reference computation.

GPflow kernels carry their own variance, so the mapping is direct. GPR realises
`ftc`, SGPR realises `dtcvar` (the collapsed Titsias bound). GPflow has no FITC
or DTC, and its coregionalisation uses a switched likelihood that GPR cannot
marginalise, so those raise UnsupportedByFramework.
"""
import numpy as np
import tensorflow as tf
import gpflow
from ._common import (UnsupportedByFramework, check_single_output_ftc_or,
                      reference_from_transformed, transformed_Y)

_LEAF = {
    "kern.rbf": gpflow.kernels.SquaredExponential,
    "kern.matern32": gpflow.kernels.Matern32,
    "kern.matern52": gpflow.kernels.Matern52,
}


def _kern(node, d):
    t = node["type"]
    if t in ("kern.add", "kern.prod"):
        parts = [_kern(p, d) for p in node["parts"]]
        return (gpflow.kernels.Sum if t == "kern.add" else gpflow.kernels.Product)(parts)
    kw = {}
    if "active_dims" in node:
        kw["active_dims"] = list(node["active_dims"])
    if t in _LEAF:
        ls = node["lengthscale"]
        kw["lengthscales"] = ls if isinstance(ls, list) else float(ls)
        return _LEAF[t](variance=node["variance"], **kw)
    if t == "kern.white":
        return gpflow.kernels.White(variance=node["variance"], **kw)
    if t == "kern.bias":
        return gpflow.kernels.Constant(variance=node["variance"], **kw)
    if t == "kern.linear":
        return gpflow.kernels.Linear(variance=node["variance"], **kw)
    raise UnsupportedByFramework("GPflow adapter: %s" % t)


def from_wire(doc):
    if "active_dims" in str(doc["kernel"]) and "coregionalize" in str(doc["kernel"]):
        raise UnsupportedByFramework("kern.coregionalize with mixed noise is not realised by GPflow GPR")
    approx = check_single_output_ftc_or(doc, ("ftc", "dtcvar"))
    d = doc["input_dim"]
    data = (tf.constant(np.array(doc["data"]["X"], float)), tf.constant(transformed_Y(doc)))
    kern = _kern(doc["kernel"], d)
    noise = doc["likelihood"]["variance"]
    if approx == "ftc":
        return gpflow.models.GPR(data, kern, noise_variance=noise)
    Z = np.array(doc["inducing_inputs"], float)
    return gpflow.models.SGPR(data, kern, inducing_variable=Z, noise_variance=noise)


def reference(doc, Xnew):
    m = from_wire(doc)
    Xn = np.array(Xnew, float)
    mean, var = m.predict_y(tf.constant(Xn))
    ll = m.log_marginal_likelihood() if hasattr(m, "log_marginal_likelihood") else m.elbo()
    if isinstance(m, gpflow.models.SGPR):
        ll = m.elbo()
    return reference_from_transformed(doc, ll.numpy(), mean.numpy(), var.numpy(), Xn)
