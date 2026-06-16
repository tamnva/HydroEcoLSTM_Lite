
import numpy as np


"""Evaluation metrics used for model performance reporting.

Currently this module provides the Nash–Sutcliffe efficiency (NSE)
implementation used for hydrological model evaluation.
"""


def nse(sim, obs, skip=0):
    """Compute the Nash–Sutcliffe efficiency between simulation and observations.

    The function ignores NaN pairs and can optionally skip an initial
    portion of the arrays using `skip`.

    Parameters
    ----------
    sim : array-like
        Simulated values.
    obs : array-like
        Observed values.
    skip : int, optional
        Number of leading samples to discard from both arrays, by default 0.

    Returns
    -------
    float
        NSE value in [-inf, 1] or `np.nan` if computation is not possible
        (e.g. zero variance in observations or no valid pairs).
    """

    sim = np.asarray(sim)[skip:]
    obs = np.asarray(obs)[skip:]

    # Keep only pairs where both values are not NaN
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    sim = sim[mask]
    obs = obs[mask]

    # If there are no valid observations after masking, return NaN
    if obs.size == 0:
        return np.nan

    # Compute denominator; protect against zero-variance or NaN
    denominator = ((obs - obs.mean()) ** 2).sum()

    if not np.isfinite(denominator) or denominator == 0:
        return np.nan

    numerator = ((obs - sim) ** 2).sum()

    return 1 - numerator / denominator




