"""Wire parameter paths, shared by the gradient implementations (contract /2).

A gradient result is keyed by the path of each parameter in the wire document, so the
same key means the same quantity in every framework:

    kernel.variance                 a single leaf kernel
    kernel.parts.0.lengthscale      leaf 0 of a compound kernel (add or prod)
    likelihood.variance             Gaussian noise variance

Values are derivatives of the log likelihood (of the target as the model sees it, after any
mean or normalizer shift, which does not depend on these parameters) with respect to the wire
value itself - positive, untransformed. A vector lengthscale (ARD) gives a vector.
Parameters held fixed by the wire format (inducing inputs of sparse regression) are not included.

A Bayesian GP-LVM (CIP-0006) adds the variational parameters, as matrices:

    latent.mean                     n x q, the means of q(X)
    latent.variance                 n x q, the variances of q(X), positive
    inducing_inputs                 m x q

Its gradient is that of the full objective, bound_without_kl - kl, which is what training
maximises, with respect to the wire values themselves (not log S). Gradients with respect
to latent coordinates are compared in the gauge of the document: the benchmark fixes q(X),
so there is no rotation or sign freedom to remove.
"""
LEAF_PARAMS = {
    "kern.rbf": ("variance", "lengthscale"),
    "kern.matern32": ("variance", "lengthscale"),
    "kern.matern52": ("variance", "lengthscale"),
    "kern.exponential": ("variance", "lengthscale"),
    "kern.bias": ("variance",),
    "kern.linear": ("variance",),
    "kern.white": ("variance",),
}


def kernel_leaves(node, path="kernel"):
    """Yield (path, leaf node) for the leaves of a (possibly compound) kernel node."""
    if "parts" in node:
        for i, p in enumerate(node["parts"]):
            yield from kernel_leaves(p, "%s.parts.%d" % (path, i))
    else:
        yield path, node


def wire_paths(doc):
    """Ordered list of (path, wire value) for every differentiable parameter in doc.

    Raises UnsupportedByFramework for models the gradient contract does not cover:
    non-Gaussian likelihoods and kernels outside LEAF_PARAMS.
    """
    from ._common import UnsupportedByFramework
    if doc["likelihood"]["type"] != "likelihoods.gaussian":
        raise UnsupportedByFramework("gradient: %s is outside the gradient contract" % doc["likelihood"]["type"])
    out = []
    for path, leaf in kernel_leaves(doc["kernel"]):
        if leaf["type"] not in LEAF_PARAMS:
            raise UnsupportedByFramework("gradient: %s is outside the gradient contract" % leaf["type"])
        for name in LEAF_PARAMS.get(leaf["type"], ()):
            if name in leaf:
                out.append(("%s.%s" % (path, name), leaf[name]))
    out.append(("likelihood.variance", doc["likelihood"]["variance"]))
    if doc.get("type") == "models.bayesian_gplvm":
        out += [("latent.mean", doc["latent"]["mean"]), ("latent.variance", doc["latent"]["variance"]),
                ("inducing_inputs", doc["inducing_inputs"])]
    return out


def node_at(doc, path):
    """(container, key) for a wire parameter path, so container[key] is its value."""
    parts = path.split(".")
    node = doc
    for p in parts[:-1]:
        node = node[int(p)] if p.isdigit() else node[p]
    return node, parts[-1]


def perturbed(doc, path, delta, index=None):
    """Deep copy of doc with the wire parameter at path (element index if a vector) moved by delta."""
    import copy
    d = copy.deepcopy(doc)
    node, key = node_at(d, path)
    if index is None:
        node[key] = node[key] + delta
    elif isinstance(index, tuple):
        node[key][index[0]][index[1]] += delta
    else:
        node[key][index] = node[key][index] + delta
    return d
