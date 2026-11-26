"""Pieces shared by the framework adapters that have no GPy-style normalizer."""
import numpy as np


class UnsupportedByFramework(NotImplementedError):
    """The wire document describes something this framework cannot realise exactly."""


def target_transform(doc):
    """(shift, scale) such that the model is fitted to (y - shift) / scale.

    mean.constant c is equivalent to fitting y - c; normalizer.center and
    normalizer.standardize fit (y - mean) / std. The two are exclusive in 0.1.
    """
    if "mean" in doc and "normalizer" in doc:
        raise UnsupportedByFramework("mean and normalizer together are not in schema 0.1")
    if "normalizer" in doc:
        n = doc["normalizer"]
        return float(n["mean"]), float(n.get("std", 1.0))
    if "mean" in doc:
        return float(doc["mean"]["value"]), 1.0
    return 0.0, 1.0


def reference_from_transformed(doc, ll, mean, var, Xnew):
    """Map log p((y - shift)/scale) and its predictions back to the original scale."""
    shift, scale = target_transform(doc)
    n = np.asarray(doc["data"]["Y"], float).size
    ll = float(ll)
    if "normalizer" in doc:
        ll -= n * np.log(scale)          # Jacobian of the target transform
    return {
        "log_likelihood": ll,
        "Xnew": np.asarray(Xnew, float).tolist(),
        "mean": (np.asarray(mean, float).ravel() * scale + shift).tolist(),
        "var": (np.asarray(var, float).ravel() * scale ** 2).tolist(),
    }


def check_single_output_ftc_or(doc, allowed):
    approx = doc.get("approximation", "ftc")
    if approx not in allowed:
        raise UnsupportedByFramework("approximation '%s' is not realised by this framework" % approx)
    if doc["likelihood"]["type"] != "likelihoods.gaussian":
        raise UnsupportedByFramework(doc["likelihood"]["type"])
    return approx


def transformed_Y(doc):
    shift, scale = target_transform(doc)
    return (np.asarray(doc["data"]["Y"], float) - shift) / scale
