# -*- coding: utf-8 -*-
"""Régression GPR dans l'espace latent (Partie 2).

La GPR prédit les codes latents z à partir de (mu, sigma^2). Avec plusieurs
sorties, scikit-learn traite les d colonnes de Z comme d processus gaussiens
qui PARTAGENT les mêmes hyperparamètres (normalize_y=True standardise chaque colonne).

Choix de modélisation :
  - entrée (mu, log sigma^2) : les codes varient très vite quand sigma^2 -> 0,
    le logarithme étale cette zone ;
  - noyaux anisotropes (une longueur caractéristique par entrée) quand
    scikit-learn le permet, avec des bornes qui empêchent leur effondrement.
"""
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ConstantKernel as C, WhiteKernel,
)

_LS_BOUNDS = (1e-2, 1e2)
_NOISE = dict(noise_level=1e-3, noise_level_bounds=(1e-10, 1e0))
KERNELS = {
    "RBF":    C(1.0) * RBF([1.0, 1.0], length_scale_bounds=_LS_BOUNDS)
              + WhiteKernel(**_NOISE),
    "Matern": C(1.0) * Matern([1.0, 1.0], length_scale_bounds=_LS_BOUNDS, nu=2.5)
              + WhiteKernel(**_NOISE),
    "RQ":     C(1.0) * RationalQuadratic(1.0, 1.0, length_scale_bounds=_LS_BOUNDS)
              + WhiteKernel(**_NOISE),
}


def log_variance_features(X):
    """(mu, sigma^2) -> (mu, log sigma^2) : entrées effectivement vues par la GPR."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.column_stack([X[:, 0], np.log(X[:, 1])])


class LogVarianceGPR(GaussianProcessRegressor):
    """GaussianProcessRegressor qui travaille sur (mu, log sigma^2).

    On lui donne toujours (mu, sigma^2) brut : la transformation est faite dans
    fit() et predict(), donc gpr.predict(X) s'utilise normalement en Partie 3.
    """

    def fit(self, X, y):
        return super().fit(log_variance_features(X), y)

    def predict(self, X, return_std=False, return_cov=False):
        return super().predict(log_variance_features(X), return_std=return_std,
                               return_cov=return_cov)


def fit_gpr(X_train, Z_train, kernel_name="RBF", n_restarts=3, alpha=1e-5,
            log_variance=True, kernel=None, seed=0):
    """Entraîne une GPR multi-sorties X -> Z.

    Parameters
    ----------
    X_train : (n, 2) paramètres (mu, sigma^2)
    Z_train : (n, d) codes latents
    kernel_name : clé de KERNELS (ignorée si `kernel` est fourni)
    n_restarts : redémarrages de l'optimisation de la log-vraisemblance marginale
    alpha : valeur ajoutée à la diagonale de la matrice de covariance
    log_variance : True -> la GPR travaille sur (mu, log sigma^2)
    kernel : noyau à utiliser à la place de KERNELS[kernel_name] (tests de Q2.2)
    seed : graine des redémarrages

    Returns
    -------
    gpr : GPR ajustée, avec l'attribut fit_time_ (s)
    """
    kern = clone(kernel if kernel is not None else KERNELS[kernel_name])
    cls = LogVarianceGPR if log_variance else GaussianProcessRegressor
    gpr = cls(kernel=kern, alpha=alpha, n_restarts_optimizer=n_restarts,
              normalize_y=True, random_state=seed)
    t0 = time.perf_counter()
    gpr.fit(X_train, Z_train)
    gpr.fit_time_ = time.perf_counter() - t0
    return gpr


def latent_rmse(gpr, X_test, Z_test):
    """RMSE moyenne des prédictions latentes sur le test, sqrt(mean((z-z_hat)^2))."""
    Z_test = np.asarray(Z_test, dtype=float).reshape(len(X_test), -1)
    Z_hat = np.asarray(gpr.predict(X_test)).reshape(Z_test.shape)
    return float(np.sqrt(np.mean((Z_test - Z_hat) ** 2)))


def latent_rmse_relative(gpr, X_test, Z_test):
    """RMSE latente / écart-type des codes : sans unité, comparable entre réducteurs
    (0 = parfait, 1 = aussi mauvais que prédire le code moyen)."""
    Z_test = np.asarray(Z_test, dtype=float).reshape(len(X_test), -1)
    return latent_rmse(gpr, X_test, Z_test) / float(np.sqrt(np.mean(Z_test.var(axis=0))))


def decoded_curve_errors(gpr, decoder, X_test, y_test):
    """EQM de chaque courbe de test après GPR + décodage, array (n_test,)."""
    y_hat = np.asarray(decoder(gpr.predict(X_test)))
    return np.mean((np.asarray(y_test) - y_hat) ** 2, axis=1)


def decoded_curve_mse(gpr, decoder, X_test, y_test):
    """EQM moyenne de courbe après décodage des prédictions GPR."""
    return float(decoded_curve_errors(gpr, decoder, X_test, y_test).mean())


def paired_comparison(err_a, err_b, n_boot=5000, seed=0):
    """Compare deux modèles courbe par courbe (mêmes courbes de test).

    Returns
    -------
    dict : ratio (EQM A / EQM B), ratio_ic95 (bootstrap), part_a_meilleur,
           p_wilcoxon (petite p-valeur -> écart significatif)
    """
    err_a, err_b = np.asarray(err_a), np.asarray(err_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err_a), size=(n_boot, len(err_a)))
    ratios = err_a[idx].mean(axis=1) / err_b[idx].mean(axis=1)
    return {
        'ratio': float(err_a.mean() / err_b.mean()),
        'ratio_ic95': (float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))),
        'part_a_meilleur': float(np.mean(err_a < err_b)),
        'p_wilcoxon': float(wilcoxon(err_a, err_b).pvalue),
    }


def evaluate_gpr(reducer, X_train, y_train, X_test, y_test, kernel_name="RBF",
                 n_restarts=0, alpha=1e-5, **kwargs):
    """Construit et évalue le méta-modèle (mu, sigma^2) -> GPR -> z -> décodeur -> courbe.

    Returns
    -------
    gpr : GPR ajustée
    record : dict des métriques (prêt pour un tableau)
    errors : array (n_test,), EQM décodée de chaque courbe de test
    """
    Z_train, Z_test = reducer.transform(y_train), reducer.transform(y_test)
    gpr = fit_gpr(X_train, Z_train, kernel_name=kernel_name, n_restarts=n_restarts,
                  alpha=alpha, **kwargs)
    errors = decoded_curve_errors(gpr, reducer.decoder, X_test, y_test)
    y_rec = np.asarray(reducer.inverse_transform(Z_test))
    record = {
        'RMSE latente': latent_rmse(gpr, X_test, Z_test),
        'RMSE latente relative': latent_rmse_relative(gpr, X_test, Z_test),
        'EQM reconstruction': float(np.mean((np.asarray(y_test) - y_rec) ** 2)),
        'EQM décodée': float(errors.mean()),
        'temps GPR (s)': gpr.fit_time_,
    }
    return gpr, record, errors