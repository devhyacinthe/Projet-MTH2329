# -*- coding: utf-8 -*-

"""Génération des données du projet MTH2329.

Contient la densité gaussienne discrétisée et la génération du jeu de
courbes aléatoires. Ce module est FOURNI complet : lisez-le, il fixe les
conventions utilisées dans tout le projet.
"""
import numpy as np


def gaussian_pdf(x_grid, mu, var):
    """Densité gaussienne de moyenne mu et de variance var, évaluée sur x_grid.

    Parameters
    ----------
    x_grid : array-like, shape (M,)
    mu : float
    var : float (variance sigma^2 > 0)

    Returns
    -------
    array, shape (M,)
    """
    return (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-(x_grid - mu) ** 2 / (2.0 * var))


def make_dataset(N=2000, M=800, mu_min=-4.0, mu_max=4.0, k_shape=2.0,
                 theta_scale=2.0, seed=42):
    """Génère N courbes gaussiennes aléatoires.

    Paramètres d'entrée (loi fixée du projet) :
        mu  ~ U[mu_min, mu_max]
        sigma^2 ~ Gamma(k_shape, theta_scale)   (paramétrisation shape-scale)

    Returns
    -------
    x_grid : array (M,), grille uniforme de [-10, 10]
    y : array (N, M), les courbes
    X : array (N, 2), les paramètres (mu, sigma^2)
    """
    rng = np.random.default_rng(seed)
    mu_values = rng.uniform(mu_min, mu_max, N)
    var_values = rng.gamma(k_shape, theta_scale, N)
    x_grid = np.linspace(-10.0, 10.0, M)
    y = np.array([gaussian_pdf(x_grid, mu, var)
                  for mu, var in zip(mu_values, var_values)])
    X = np.column_stack([mu_values, var_values])
    return x_grid, y, X
