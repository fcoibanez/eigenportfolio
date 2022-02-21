import numpy as np
import itertools
import pandas as pd


class PortfolioRebalance:
    def __init__(self, sample, volatility_target=None):
        self.sample = sample
        self.cov = self.risk_model()
        self.vol = np.sqrt(np.diag(self.cov))
        self.mu = self.expected_returns()
        self.weights = None
        self._n = self.sample.shape[-1]
        self._names = list(sample.columns)
        self.vol_target = volatility_target

    def risk_model(self):
        x = self.sample.dropna()  # Only overlapping observations
        sigma = x.cov().values
        return sigma

    def expected_returns(self):
        x = self.sample.dropna()  # Only overlapping observations
        mu = x.mean().values.reshape(-1, 1)
        return mu

    def scale_weights(self, weights):
        current_vol = np.sqrt((weights.T @ self.cov @ weights).item())
        scaling_factor = self.vol_target / current_vol
        scaled_weights = weights * scaling_factor
        return scaled_weights

    def get_spectral(self, criteria=None, cutoff=None):
        S = np.diag(np.sqrt(np.diag(self.cov)))
        R = np.linalg.inv(S) @ self.cov @ np.linalg.inv(S)
        Lambda, V = np.linalg.eig(R)
        order = Lambda.argsort()[::-1]
        L = np.diag(Lambda[order])
        V = V[:, order]

        # Eigen-values/vectors truncation
        if cutoff is not None:
            lambdas = np.diag(L).copy()
            cumulative_variance = np.cumsum(lambdas / lambdas.sum())
            cutoff_position = (cumulative_variance < cutoff).sum() + 1
            L_truncated = np.diag(lambdas[:cutoff_position])
            V_truncated = V[:, :cutoff_position]
            L, V = L_truncated, V_truncated

        combinations = [np.reshape(np.array(i), (L.shape[0], 1)) for i in itertools.product([1, -1], repeat=L.shape[0])]
        J = np.concatenate(combinations, axis=1)
        W = np.linalg.inv(S) @ V @ np.power(np.linalg.inv(L), 1/2) @ J
        W = np.divide(W, W.sum(axis=0))

        if criteria == 'RiskContribution':
            risk_contributions = pd.DataFrame(index=self._names, columns=range(W.shape[1]))
            for i in range(J.shape[1]):
                w = W[:, i].reshape(-1, 1)
                risk_contributions[i] = ((np.diag(w.flatten()) @ self.cov @ w) / (w.T @ self.cov @ w)).flatten()
            portfolio_number = risk_contributions.pow(2).sum().argmin()
        elif criteria == 'MinVol':
            portfolio_vol = pd.Series(np.sqrt(np.diag(W.T @ self.cov @ W)), index=range(W.shape[1]))
            portfolio_number = np.argmin(portfolio_vol)
        else:
            portfolio_number = 0

        wts = W[:, portfolio_number]
        if self.vol_target is not None:
            wts = self.scale_weights(wts)
        wts = pd.Series(wts, index=self._names)
        self.weights = wts

    def get_maxsharpe(self):
        wts = self.mu.T @ np.linalg.inv(self.cov)
        wts /= wts.sum()
        if self.vol_target is not None:
            wts = self.scale_weights(wts.reshape(-1, 1))
        wts = pd.Series(wts.flatten(), index=self._names)
        self.weights = wts

    def get_equalweight(self):
        wts = pd.Series(1 / self._n, index=self._names)
        if self.vol_target is not None:
            wts = self.scale_weights(wts.values.reshape(-1, 1))
        self.weights = pd.Series(wts.flatten(), index=self._names)

    def get_inversevol(self):
        wts = 1 / self.vol
        wts /= wts.sum()
        if self.vol_target is not None:
            wts = self.scale_weights(wts.reshape(-1, 1))
        wts = pd.Series(wts.flatten(), index=self._names)
        self.weights = wts

        
def effective_bets(weights, cov):
    Sigma = cov
    S = np.diag(np.sqrt(np.diag(Sigma)))
    Rho = np.linalg.inv(S) @ Sigma @ np.linalg.inv(S)

    # We diagonalize Rho into its eigen-vector and eigen-value matrices
    Lambda, V = np.linalg.eig(Rho)
    order = Lambda.argsort()[::-1]
    Lambda = np.diag(Lambda[order])
    V = V[:, order]

    p = np.multiply(V.T @ S @ weights, Lambda @ V.T @ S @ weights) / (weights.T @ cov @ weights)
    eta = np.exp(-np.sum(np.multiply(p, np.log(p))))
    return eta
