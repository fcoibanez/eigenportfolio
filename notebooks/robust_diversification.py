"""
Objective:
- Backtest the idea and see if the results make sense

Functions:
- Factor weights
- ENB (with truncation option)

Class:
- Receives:
    - Sample (even panel for now)
- Calculates:
    - Normalized data
    - Robust PCA
    - Optimal Hard Truncation Threshold
    - Weights (min_(w) -ENB + lambda(w.T @ w)) -> first without regularization
        If we are truncating, should we also control the outside of truncation part?
    - ENB value
"""
import pandas as pd
import numpy as np
from rpca import RobustPCA
import matplotlib.pyplot as plt
from scipy.linalg import svd
from optht import optht
from scipy.optimize import minimize
from sklearn import preprocessing


def effective_bets(weights, singular_values_matrix, eigen_vector_matrix, vol, k=None):
    w = weights.reshape(-1, 1)
    eigen_wts = eigen_vector_matrix.T @ np.diag(vol) @ w
    p = (np.diag(eigen_wts.flatten()) @ singular_values_matrix.T @ singular_values_matrix @ eigen_wts).flatten()
    if k is not None:
        p = p[:k]
    p_norm = np.divide(p, p.sum())
    eta = np.exp(-np.sum(np.multiply(p_norm, np.log(p_norm))))
    return eta


def objective_diversify(weights, singular_values_matrix, eigen_vector_matrix, vol, k=None):
    enb = effective_bets(weights, singular_values_matrix, eigen_vector_matrix, vol, k)
    return -enb


# noinspection PyTupleAssignmentBalance
class RobustDiversification:
    def __init__(self, sample):
        self._raw = sample.copy()
        self._t, self._n = self._raw.shape
        self._scaler = preprocessing.StandardScaler().fit(self._raw)
        self.M = self._scaler.transform(self._raw)  # Original matrix
        self.L = self.M.copy()  # Low rank clean-data matrix
        self.S = np.zeros(self.M.shape)  # Sparse corrupt-data matrix
        self.k = self._n  # No truncation
        self.weights = None
        self.volatility = self._raw.std().values

        # Initial SVD
        u, s, vt = svd(
            self._raw,
            full_matrices=False,
            compute_uv=True,
            lapack_driver='gesdd'
        )
        s = np.diag(s)
        v = vt.T
        self.singular_values = s
        self.eigen_vectors = v
        self.eta = effective_bets(np.ones(self._n), self.singular_values, self.eigen_vectors, self.volatility)

    def robustify(self, hp, max_iter=1E6):
        rob = RobustPCA(lmb=hp, max_iter=int(max_iter))
        self.L, self.S = rob.fit(self.M)

    def compress(self, sigma=None):
        # Hard thresholding Truncated SVD
        k = optht(self.L, sv=np.diag(self.singular_values), sigma=sigma)
        self.k = k
        self.eigen_vectors[:, k:] = 0
        self.singular_values[k:, k:] = 0

    def allocate(self, objective_func=objective_diversify, initial_guess=None, constraints=(), tol=1E-12,
                 solver='SLSQP', maxiter=1E9):
        if initial_guess is None:
            x0 = np.array([1 / self._n] * self._n)
        else:
            x0 = initial_guess

        opti = minimize(
            fun=objective_func,
            x0=x0,
            args=(self.singular_values, self.eigen_vectors, self.volatility, self.k),
            constraints=constraints,
            method=solver,
            options={'maxiter': maxiter, 'ftol': tol}
        )
        res = pd.Series(opti.x, index=self._raw.columns)
        self.weights = res
        self.eta = effective_bets(opti.x, self.singular_values, self.eigen_vectors, self.volatility, self.k)


if __name__ == '__main__':
    raw = pd.read_pickle('etf_er.pkl')
    data = raw.dropna()  # Working with even panel for now
    lmb = 4 / np.sqrt(max(data.shape[0], data.shape[1]))

    cons0 = (
        {'type': 'ineq', 'fun': lambda x: np.sum(x) - 1}
    )

    cons1 = (
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda x: np.sum(x) - 1}
    )

    model = RobustDiversification(sample=data)
    model.robustify(hp=lmb)
    model.compress()
    model.allocate(constraints=cons0)
    model.eta







