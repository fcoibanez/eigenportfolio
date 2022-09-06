import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import riskfolio as rp
import cvxpy as cp
from sklearn import preprocessing
from scipy.linalg import svd
from scipy.stats import random_correlation
from sklearn.cluster import AgglomerativeClustering

# ---------------------------------------------------------------------
# Simulations parameters
n_iter = 10000  # number of samples
k = 50  # number of securities
t = 252 * 3  # number of observations
rng = np.random.default_rng(seed=0)
n_rebalances = 12
rebalance_dts = [-21 * n_iter for n_iter in range(n_rebalances, 0, -1)]
window = 252 * 2

# Model paramters
thresh = .55
lmb = 1E6

# Results
flds = ['Sharpe', 'Omega', 'Sortino', 'Vol', 'MaxDD', 'TurnOver']
summary_mdl = pd.DataFrame(index=range(n_iter), columns=flds)
summary_ew = pd.DataFrame(index=range(n_iter), columns=flds)
summary_invol = pd.DataFrame(index=range(n_iter), columns=flds)
summary_erc = pd.DataFrame(index=range(n_iter), columns=flds)

# ---------------------------------------------------------------------
# Synthetic Backtest
for sim in tqdm(tqdm(range(n_iter))):
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

    rt = S @ L @ dWt.T + dJt
    rt = pd.DataFrame(rt.T)  # Daily simulated returns
    rt_oos = rt.iloc[rebalance_dts[0]:]  # OOS performance

    model_wt = pd.DataFrame(index=rt_oos.index, columns=rt.columns)  # Daily drifted weights
    model_wt_tgt = pd.DataFrame(index=range(n_rebalances), columns=rt.columns)  # Target weights
    ew_wt = pd.DataFrame(index=rt_oos.index, columns=rt.columns)  # Daily drifted weights
    ew_wt_tgt = pd.DataFrame(index=range(n_rebalances), columns=rt.columns)  # Target weights
    invol_wt = pd.DataFrame(index=rt_oos.index, columns=rt.columns)  # Daily drifted weights
    invol_wt_tgt = pd.DataFrame(index=range(n_rebalances), columns=rt.columns)  # Target weights
    erc_wt = pd.DataFrame(index=rt_oos.index, columns=rt.columns)  # Daily drifted weights
    erc_wt_tgt = pd.DataFrame(index=range(n_rebalances), columns=rt.columns)  # Target weights

    for i in range(n_rebalances):
        dt_loc = rt.index[rebalance_dts[i]]  # Rebalance 'date'
        sample = rt.iloc[dt_loc - window:dt_loc].copy()

        # ---------------------------------------------------------------------
        # Model
        scaler = preprocessing.StandardScaler().fit(sample.values)
        sample_z = scaler.transform(sample.values)

        D = np.diag(np.sqrt(scaler.var_))
        U, s, Vt = svd(
            sample_z,
            full_matrices=False,
            compute_uv=True,
            lapack_driver='gesdd'
        )
        S = np.diag(s)
        V = Vt.T
        n = V.shape[0]
        L = V @ S

        # Agglomerative Clustering
        clustering = AgglomerativeClustering(
            affinity='cosine',
            linkage='average',
            distance_threshold=thresh,
            n_clusters=None
        )
        clustering.fit(L)
        c = clustering.n_clusters_

        # Cluster mapping matrix C
        clusters = pd.Series(clustering.labels_, index=sample.columns)
        cluster_map = pd.DataFrame(0, index=sample.columns, columns=range(c))

        for x in cluster_map.columns:
            members = clusters[clusters == x].index
            cluster_map.loc[members, x] = 1

        C = cluster_map.values

        # Optimization
        P = C.T @ V @ S.T @ S @ V.T @ C
        P = P + 2 * lmb * np.eye(c)
        q = -C.T @ V @ S @ np.ones((n, 1))

        # Constraints
        lb = 0
        A = np.ones((1, n)) @ np.linalg.inv(D) @ C
        b = np.array(1)
        G = np.concatenate([-np.linalg.inv(D) @ C])
        h = np.array([-lb] * n)

        # Allocating
        x = cp.Variable(c)
        prob = cp.Problem(cp.Minimize((1/2) * cp.quad_form(x, P) + q.T @ x), [G @ x <= h, A @ x == b])
        prob.solve()
        w_star = np.linalg.inv(D) @ C @ x.value.reshape(-1, 1)
        w_star[np.isclose(w_star, 0, atol=1E-11)] = 0
        model_wt_tgt.loc[i] = w_star.flatten()

        # Weights drifting
        rt_post_rebalance = rt_oos.copy().loc[dt_loc + 1:dt_loc + 20]
        gross_rt = rt_post_rebalance.add(1).cumsum()[::-1]
        gross_rt.index = rt_post_rebalance.index
        upr = gross_rt.mul(w_star.flatten())  # Securities part
        lwr = upr.sum(axis=1)  # Portfolio part
        drifted_wts = upr.div(lwr, axis=0)

        model_wt.loc[dt_loc] = w_star.flatten()
        model_wt.loc[dt_loc + 1:dt_loc + 20] = drifted_wts.values

        # ---------------------------------------------------------------------
        # Equal-weighted
        w_star = np.array([1 / n] * n)
        ew_wt_tgt.loc[i] = w_star

        # Weights drifting
        rt_post_rebalance = rt_oos.copy().loc[dt_loc + 1:dt_loc + 20]
        gross_rt = rt_post_rebalance.add(1).cumsum()[::-1]
        gross_rt.index = rt_post_rebalance.index
        upr = gross_rt.mul(w_star)  # Securities part
        lwr = upr.sum(axis=1)  # Portfolio part
        drifted_wts = upr.div(lwr, axis=0)

        ew_wt.loc[dt_loc] = w_star
        ew_wt.loc[dt_loc + 1:dt_loc + 20] = drifted_wts.values

        # ---------------------------------------------------------------------
        # Inverse-vol
        vol = sample.std() * np.sqrt(252)
        w_star = np.divide(vol, vol.sum())
        invol_wt_tgt.loc[i] = w_star

        # Weights drifting
        rt_post_rebalance = rt_oos.copy().loc[dt_loc + 1:dt_loc + 20]
        gross_rt = rt_post_rebalance.add(1).cumsum()[::-1]
        gross_rt.index = rt_post_rebalance.index
        upr = gross_rt.mul(w_star)  # Securities part
        lwr = upr.sum(axis=1)  # Portfolio part
        drifted_wts = upr.div(lwr, axis=0)

        invol_wt.loc[dt_loc] = w_star
        invol_wt.loc[dt_loc + 1:dt_loc + 20] = drifted_wts.values

        # ---------------------------------------------------------------------
        # Equal-risk contribution
        port = rp.Portfolio(returns=sample)
        port.assets_stats(method_mu='hist', method_cov='hist')
        w_star = port.rp_optimization(
            model='Classic',
            rm='MV',
            rf=0,
            b=None,
            hist=True
        )
        erc_wt_tgt.loc[i] = w_star.squeeze()

        # Weights drifting
        rt_post_rebalance = rt_oos.copy().loc[dt_loc + 1:dt_loc + 20]
        gross_rt = rt_post_rebalance.add(1).cumsum()[::-1]
        gross_rt.index = rt_post_rebalance.index
        upr = gross_rt.mul(w_star.squeeze())  # Securities part
        lwr = upr.sum(axis=1)  # Portfolio part
        drifted_wts = upr.div(lwr, axis=0)

        erc_wt.loc[dt_loc] = w_star.squeeze()
        erc_wt.loc[dt_loc + 1:dt_loc + 20] = drifted_wts.values

    # Model results
    wt_iter = model_wt.copy()
    wt_tgt_iter = model_wt_tgt.copy()
    rt_iter = wt_iter.shift(1).mul(rt_oos).dropna(how='all').sum(axis=1)
    nav_iter = wt_iter.shift(1).mul(rt_oos).sum(axis=1).add(1).cumprod().mul(100)
    dd = nav_iter.div(nav_iter.cummax()).sub(1)

    summary_mdl.loc[sim, 'Sharpe'] = rt_iter.mean() / rt_iter.std() * np.sqrt(252)
    summary_mdl.loc[sim, 'Omega'] = rp.LPM(rt_iter, p=1)
    summary_mdl.loc[sim, 'Sortino'] = rp.LPM(rt_iter, p=2)
    summary_mdl.loc[sim, 'Vol'] = rt_iter.std() * np.sqrt(252)
    summary_mdl.loc[sim, 'TurnOver'] = wt_tgt_iter.diff().abs().sum().sum() / 2
    summary_mdl.loc[sim, 'MaxDD'] = -dd.min()

    # Equally-weighted results
    wt_iter = ew_wt.copy()
    wt_tgt_iter = ew_wt_tgt.copy()
    rt_iter = wt_iter.shift(1).mul(rt_oos).dropna(how='all').sum(axis=1)
    nav_iter = wt_iter.shift(1).mul(rt_oos).sum(axis=1).add(1).cumprod().mul(100)
    dd = nav_iter.div(nav_iter.cummax()).sub(1)

    summary_ew.loc[sim, 'Sharpe'] = rt_iter.mean() / rt_iter.std() * np.sqrt(252)
    summary_ew.loc[sim, 'Omega'] = rp.LPM(rt_iter, p=1)
    summary_ew.loc[sim, 'Sortino'] = rp.LPM(rt_iter, p=2)
    summary_ew.loc[sim, 'Vol'] = rt_iter.std() * np.sqrt(252)
    summary_ew.loc[sim, 'TurnOver'] = wt_tgt_iter.diff().abs().sum().sum() / 2
    summary_ew.loc[sim, 'MaxDD'] = -dd.min()

    # Inverse-vol results
    wt_iter = invol_wt.copy()
    wt_tgt_iter = invol_wt_tgt.copy()
    rt_iter = wt_iter.shift(1).mul(rt_oos).dropna(how='all').sum(axis=1)
    nav_iter = wt_iter.shift(1).mul(rt_oos).sum(axis=1).add(1).cumprod().mul(100)
    dd = nav_iter.div(nav_iter.cummax()).sub(1)

    summary_invol.loc[sim, 'Sharpe'] = rt_iter.mean() / rt_iter.std() * np.sqrt(252)
    summary_invol.loc[sim, 'Omega'] = rp.LPM(rt_iter, p=1)
    summary_invol.loc[sim, 'Sortino'] = rp.LPM(rt_iter, p=2)
    summary_invol.loc[sim, 'Vol'] = rt_iter.std() * np.sqrt(252)
    summary_invol.loc[sim, 'TurnOver'] = wt_tgt_iter.diff().abs().sum().sum() / 2
    summary_invol.loc[sim, 'MaxDD'] = -dd.min()

    # Equal risk contribution results
    wt_iter = erc_wt.copy()
    wt_tgt_iter = erc_wt_tgt.copy()
    rt_iter = wt_iter.shift(1).mul(rt_oos).dropna(how='all').sum(axis=1)
    nav_iter = wt_iter.shift(1).mul(rt_oos).sum(axis=1).add(1).cumprod().mul(100)
    dd = nav_iter.div(nav_iter.cummax()).sub(1)

    summary_erc.loc[sim, 'Sharpe'] = rt_iter.mean() / rt_iter.std() * np.sqrt(252)
    summary_erc.loc[sim, 'Omega'] = rp.LPM(rt_iter, p=1)
    summary_erc.loc[sim, 'Sortino'] = rp.LPM(rt_iter, p=2)
    summary_erc.loc[sim, 'Vol'] = rt_iter.std() * np.sqrt(252)
    summary_erc.loc[sim, 'TurnOver'] = wt_tgt_iter.diff().abs().sum().sum() / 2
    summary_erc.loc[sim, 'MaxDD'] = -dd.min()

print('hola')
