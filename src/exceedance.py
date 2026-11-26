# -*- coding: utf-8 -*-

"""Probabilité de dépassement de seuil — Partie 3.

Deux estimateurs de E(x, tau) = P(phi(x; mu, sigma^2) >= tau) :
  1. la référence exacte par intégration numérique (FOURNIE) ;
  2. l'estimateur Monte Carlo à travers le méta-modèle (TODO).

Modèle d'entrée (cf. sujet, section « Loi d'entrée ») :
    mu     ~ Uniforme[mu_min, mu_max]        (typiquement [-4, 4])
    sigma^2 ~ Gamma(k = shape, theta = scale) (typiquement k=2, theta=2)
    mu et sigma^2 indépendants.

Référence exacte (cf. équation (3) du sujet) :
    E(x, tau) = int_0^{v_tau} |[x-c(v), x+c(v)] inter [mu_min, mu_max]|
                              / (mu_max - mu_min)
                              * gamma(v; k, theta)  dv
    avec c(v) = sqrt(2 v ln(1/(tau sqrt(2 pi v))))
         v_tau = 1 / (2 pi tau^2)
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma


# ---------------------------------------------------------------------------
# Référence exacte (FOURNIE)
# ---------------------------------------------------------------------------
def make_exact_exceedance(tau, mu_min, mu_max, k_shape, theta_scale):
    """Construit la fonction x -> E(x, tau) par intégration numérique 1-D.

    Parameters
    ----------
    tau : seuil (> 0)
    mu_min, mu_max : bornes de la loi uniforme sur mu
    k_shape, theta_scale : paramètres shape / scale de la loi Gamma
                           sur sigma^2 (attention : c'est une VARIANCE)

    Returns
    -------
    exact_exceedance : callable x -> E(x, tau) (scalaire)
    """
    v_tau = 1.0 / (2.0 * np.pi * tau ** 2)

    def exact_exceedance(x):
        def integrand(v):
            # clip : évite log(0) en v=0 et log(arg > 1) en v = v_tau
            v = np.clip(v, 1e-300, v_tau * (1.0 - 1e-12))
            c = np.sqrt(2.0 * v
                        * np.log(1.0 / (tau * np.sqrt(2.0 * np.pi * v))))
            lo, hi = max(x - c, mu_min), min(x + c, mu_max)
            overlap = max(hi - lo, 0.0) / (mu_max - mu_min)
            return overlap * gamma.pdf(v, a=k_shape, scale=theta_scale)

        return quad(integrand, 0.0, v_tau,
                    epsabs=1e-12, epsrel=1e-10, limit=200)[0]

    return exact_exceedance


# ---------------------------------------------------------------------------
# Monte Carlo à travers le méta-modèle 
# ---------------------------------------------------------------------------
def mc_exceedance(gpr, decoder, tau, n_samples, mu_min, mu_max,
                  k_shape, theta_scale, seed=42, batch_size=2000):
    """Estime p_hat(x, tau) = (1/L) sum_l 1{ y_hat_l(x) >= tau } par MC.

    Returns
    -------
    p_hat : array (M,), profil de probabilité de dépassement
    X_mc : array (L, 2), paramètres tirés (mu, sigma^2), pour le contrôle de Q3.2
    """
    rng = np.random.default_rng(seed)
    # 1. tirage des L paramètres sous la loi d'entrée
    mu = rng.uniform(mu_min, mu_max, size=n_samples)
    sigma2 = rng.gamma(shape=k_shape, scale=theta_scale, size=n_samples)
    # 2. tableau des entrées
    X = np.column_stack([mu, sigma2])
    # 3 à 5. GPR -> codes, décodeur -> courbes, comptage (par paquets, pour la mémoire)
    count = 0
    for start in range(0, n_samples, batch_size):
        Z = gpr.predict(X[start:start + batch_size])
        Y_hat = np.asarray(decoder(Z))
        count = count + (Y_hat >= tau).sum(axis=0)
    p_hat = count / n_samples
    # 6.
    return p_hat, X


def mc_exceedance_true_simulator(x_grid, tau, X_mc, gaussian_pdf, batch_size=2000):
    """Contrôle : probabilité empirique sur le VRAI simulateur, avec les MÊMES
    tirages X_mc que mc_exceedance (isole l'erreur propre au méta-modèle, Q3.2).

    Returns
    -------
    p_true : array (M,)
    """
    X_mc = np.asarray(X_mc, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    count = 0
    for start in range(0, len(X_mc), batch_size):
        mu = X_mc[start:start + batch_size, 0][:, None]        # (b, 1)
        sigma2 = X_mc[start:start + batch_size, 1][:, None]    # (b, 1)
        Y_true = gaussian_pdf(x_grid[None, :], mu, sigma2)     # (b, M) : b courbes d'un coup
        count = count + (Y_true >= tau).sum(axis=0)
    return count / len(X_mc)


def area_error(p_hat, p_ref, x_grid):
    """Erreur en aire int |p_hat - p_ref| dx et erreur max.

    Returns
    -------
    l1 : float, erreur L1 (intégrée sur x_grid)
    linf : float, erreur max_{x in x_grid} |p_hat(x) - p_ref(x)|
    """
    diff = np.abs(np.asarray(p_hat) - np.asarray(p_ref))
    trapz = getattr(np, 'trapezoid', None) or np.trapz      # np.trapz si numpy < 2.0
    l1 = float(trapz(diff, x_grid))
    linf = float(diff.max())
    return l1, linf


def exceedance_metrics(p_hat, p_ref, x_grid):
    """Erreur L1, erreur max, position de l'erreur max et corrélation, dans un dict."""
    l1, linf = area_error(p_hat, p_ref, x_grid)
    diff = np.abs(np.asarray(p_hat) - np.asarray(p_ref))
    return {'erreur L1 (aire)': l1,
            'erreur max': linf,
            'x de l\'erreur max': float(np.asarray(x_grid)[diff.argmax()]),
            'corrélation': float(np.corrcoef(p_hat, p_ref)[0, 1])}
    
    
def mc_error_study(gpr, decoder, x_grid, p_exact, tau, L_values, n_repeats,
                   mu_min, mu_max, k_shape, theta_scale, gaussian_pdf, seed0=0):
    """Q3.2 : pour chaque L et chaque répétition (graine différente), erreurs en aire
    'bruit MC' (vrai simulateur vs E), 'erreur méta-modèle' (méta-modèle vs vrai
    simulateur, mêmes tirages) et 'erreur totale' (méta-modèle vs E)."""
    records = []
    for L in L_values:
        for r in range(n_repeats):
            p_hat, X_mc = mc_exceedance(gpr, decoder, tau, L, mu_min, mu_max,
                                        k_shape, theta_scale, seed=seed0 + r)
            p_true = mc_exceedance_true_simulator(x_grid, tau, X_mc, gaussian_pdf)
            records.append({'L': L, 'répétition': r,
                            'bruit MC': area_error(p_true, p_exact, x_grid)[0],
                            'erreur méta-modèle': area_error(p_hat, p_true, x_grid)[0],
                            'erreur totale': area_error(p_hat, p_exact, x_grid)[0]})
    return records