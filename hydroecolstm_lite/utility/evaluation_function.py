
import numpy as np

def nse(sim, obs, skip = 0):
    
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




