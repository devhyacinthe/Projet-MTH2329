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
# Monte Carlo à travers le méta-modèle (TODO Partie 3)
# ---------------------------------------------------------------------------
def mc_exceedance(gpr, decoder, tau, n_samples, mu_min, mu_max,
                  k_shape, theta_scale, seed=42):
    """Estime p_hat(x, tau) = (1/L) sum_l 1{ y_hat_l(x) >= tau } par MC.

    Parameters
    ----------
    gpr : GPR ajusté ; gpr.predict(X) -> codes latents z (L, d)
    decoder : appelable ; decoder(z) -> courbes (L, M)
              (adapter selon votre backend PCA / AE)
    tau : seuil (> 0)
    n_samples : nombre de tirages L
    mu_min, mu_max, k_shape, theta_scale : loi d'entrée (cf. sujet)
    seed : graine pour la reproductibilité

    Returns
    -------
    p_hat : array (M,), profil de probabilité de dépassement
    X_mc : array (L, 2), paramètres tirés (mu, sigma^2)
           -> à conserver pour le contrôle « vrai simulateur » de Q3.2
    """
    rng = np.random.default_rng(seed)
    # TODO 3.1 :
    #   1. tirer L paramètres (mu, sigma^2) sous la loi d'entrée :
    #         mu      = rng.uniform(mu_min, mu_max, size=n_samples)
    #         sigma2  = rng.gamma(shape=k_shape, scale=theta_scale,
    #                             size=n_samples)
    #   2. construire X = np.column_stack([mu, sigma2])
    #   3. prédire les codes latents : Z = gpr.predict(X)         # (L, d)
    #   4. décoder : Y_hat = decoder(Z)                            # (L, M)
    #   5. moyenner les indicateurs : p_hat = (Y_hat >= tau).mean(axis=0)
    #   6. renvoyer (p_hat, X)
    raise NotImplementedError


def mc_exceedance_true_simulator(x_grid, tau, X_mc, gaussian_pdf):
    """Contrôle : probabilité empirique sur le VRAI simulateur.

    Utilise les MÊMES tirages X_mc que `mc_exceedance` afin d'isoler le
    bruit de Monte Carlo (commun aux deux estimateurs) de l'erreur
    propre au méta-modèle (question Q3.2).

    Parameters
    ----------
    x_grid : array (M,), abscisses sur lesquelles évaluer
    tau : seuil
    X_mc : array (L, 2), tirages (mu, sigma^2) issus de `mc_exceedance`
    gaussian_pdf : callable (x_grid, mu, sigma2) -> courbe (M,)
                   (fonction `gaussian_pdf` fournie dans src/data.py)

    Returns
    -------
    p_true : array (M,), probabilité empirique sur le vrai simulateur
    """
    # TODO 3.2 :
    #   Y_true = np.stack([gaussian_pdf(x_grid, mu, s2) for mu, s2 in X_mc])
    #   p_true = (Y_true >= tau).mean(axis=0)
    #   return p_true
    raise NotImplementedError


def area_error(p_hat, p_ref, x_grid):
    """Erreur en aire int |p_hat - p_ref| dx et erreur max.

    Returns
    -------
    l1 : float, erreur L1 (intégrée sur x_grid)
    linf : float, erreur max_{x in x_grid} |p_hat(x) - p_ref(x)|
    """
    # TODO 3.1 :
    #   diff = np.abs(p_hat - p_ref)
    #   l1   = np.trapezoid(diff, x_grid)       # np.trapz si numpy < 2.0
    #   linf = diff.max()
    #   return l1, linf
    raise NotImplementedError