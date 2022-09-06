import numpy as np
import pandas as pd
from scipy.stats import random_correlation
from tqdm import tqdm

# ---------------------------------------------------------------------
# Config
n = 100  # number of samples
k = 50  # number of securities
t = 252 * 3  # number of observations
rng = np.random.default_rng(seed=0)

# ---------------------------------------------------------------------
# Simulation
samples = {}
for i in tqdm(range(n)):
    # Random Correlation
    rnd_exp = 1 / rng.exponential(size=k, scale=1.75)
    eig_val = np.divide(rnd_exp, rnd_exp.sum()) * k
    eig_val.sort()
    eig_val = eig_val[::-1]
    Rho = random_correlation.rvs(eig_val, random_state=rng)

    # Difsussion process
    L = np.linalg.cholesky(Rho)
    s = [.1 / np.sqrt(252)] * k
    S = np.diag(s)
    dWt = rng.standard_normal(size=(t, k))

    # Jumps process
    jump_location = rng.poisson(lam=.15, size=(t, k))
    jump_magnitude = rng.normal(loc=-.05 / 252, scale=.075 / np.sqrt(252), size=(t, k))
    dJt = np.multiply(jump_location, jump_magnitude).T

    r_t = S @ L @ dWt.T + dJt

    pd.DataFrame(r_t.T).add(1).cumprod().plot()

    samples[i] = pd.DataFrame(r_t.T)

pd.to_pickle(samples, 'simulation_samples.pkl')
