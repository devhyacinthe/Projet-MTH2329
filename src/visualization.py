# -*- coding: utf-8 -*-

"""Fonctions de visualisation du projet MTH2329.

Toutes les figures du notebook sont construites ici, pour que le notebook ne
contienne que l'orchestration (appels de fonctions) et les réponses.

Convention commune :
    - chaque fonction renvoie l'objet matplotlib créé (Figure ou Axes) ;
    - aucune fonction n'appelle plt.show() : c'est le notebook qui affiche ;
    - les fonctions qui dessinent sur un seul graphique acceptent un argument
      `ax` optionnel, pour pouvoir composer plusieurs panneaux.
"""

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Numérotation des figures
# ---------------------------------------------------------------------------
# légende -> numéro ; rempli dans l'ordre où les figures sont créées
_FIGURE_NUMBERS = {}


def add_caption(fig, text, label=None):
    """Écrit « Figure n – text » sous la figure et renvoie n.

    Les numéros sont attribués dans l'ordre d'exécution du notebook. Réexécuter
    une cellule redonne le même numéro à la même figure (repérée par `label`,
    par défaut le texte de la légende). Après « Restart & Run All », les
    figures sont donc numérotées 1, 2, 3... dans l'ordre du notebook.

    Parameters
    ----------
    fig : Figure ou Axes matplotlib (pour un Axes, sa figure est utilisée)
    text : légende affichée après « Figure n – »
    label : identifiant de la figure (optionnel, par défaut `text`)

    Returns
    -------
    n : int, numéro de la figure (utile pour y faire référence)
    """
    fig = getattr(fig, 'figure', fig)          # accepte aussi un Axes
    key = label if label is not None else text
    if key not in _FIGURE_NUMBERS:
        _FIGURE_NUMBERS[key] = len(_FIGURE_NUMBERS) + 1
    n = _FIGURE_NUMBERS[key]

    # on place la légende sous tout ce qui est déjà dessiné (axes, titres, étiquettes)
    renderer = fig.canvas.get_renderer()
    bottom = fig.get_tightbbox(renderer).y0           # en pouces
    width_chars = int(fig.get_figwidth() * 11)        # ~11 caractères par pouce
    caption = textwrap.fill(f'Figure {n} – {text}', width=width_chars)
    fig.text(0.5, (bottom - 0.15) / fig.get_figheight(), caption,
             ha='center', va='top', fontsize=11, style='italic', color='0.15')
    return n


def reset_figure_numbers():
    """Remet la numérotation des figures à zéro (la prochaine sera la Figure 1)."""
    _FIGURE_NUMBERS.clear()
    
    

# ---------------------------------------------------------------------------
# Partie 0: Exploration des données
# ---------------------------------------------------------------------------
def plot_curve_points(x_grid, curve, mu=None, var=None, name=None,
                      zoom=(0.0, 0.5), label_every=4):
    """Dessine UNE courbe point par point, pour montrer qu'elle n'est qu'un
    vecteur de M valeurs phi(x_j), j = 0, ..., M-1.

    Haut : les M points de la courbe (sans trait entre eux).
    Bas  : agrandissement de la bande `zoom`, où chaque point est relié à
           l'axe par un trait vertical et numéroté par son indice j.

    Parameters
    ----------
    x_grid : array (M,), grille des abscisses
    curve : array (M,), la courbe (une ligne de y)
    mu, var : floats ou None, paramètres de la courbe (affichés dans le titre)
    name : str ou None, nom de la courbe (ex. « courbe n°505 du train »)
    zoom : (x_min, x_max), bande agrandie dans le panneau du bas
    label_every : on affiche l'indice j d'un point sur `label_every`

    Returns
    -------
    fig : Figure matplotlib
    """
    M = len(curve)
    dx = x_grid[1] - x_grid[0]
    z_lo, z_hi = zoom
    sel = np.where((x_grid >= z_lo) & (x_grid <= z_hi))[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                   gridspec_kw={'hspace': 0.35})

    # --- haut : les M points ------------------------------------------------
    ax1.plot(x_grid, curve, 'o', ms=2.5, color='C0')
    title = fr'Une courbe = {M} points $(x_j,\ \phi(x_j))$'
    if name is not None:
        title += f'  —  {name}'
    if mu is not None and var is not None:
        title += fr' : $\mu = {mu:.2f}$, $\sigma^2 = {var:.2f}$'
    ax1.set_title(title, loc='left')
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\phi(x_j; \mu, \sigma^2)$')
    ax1.set_xlim(x_grid[0] - 0.5, x_grid[-1] + 0.5)
    ax1.set_ylim(-0.03 * curve.max() / 0.28, curve.max() * 1.15)

    y_note = 0.2 * curve.max()
    ax1.annotate(fr'$j = 0$ : $x_0 = {x_grid[0]:g}$', xy=(x_grid[0], curve[0]),
                 xytext=(x_grid[0] + 0.2, y_note),
                 arrowprops=dict(arrowstyle='->'), fontsize=11)
    ax1.annotate(fr'$j = {M - 1}$ : $x_{{{M - 1}}} = {x_grid[-1]:g}$',
                 xy=(x_grid[-1], curve[-1]), xytext=(x_grid[-1] - 3.4, y_note),
                 arrowprops=dict(arrowstyle='->'), fontsize=11)
    ax1.text(x_grid[0] + 3.8, 0.72 * curve.max(),
             f'Les {M} points sont si serrés\n(un tous les {dx:.3f})\nqu\'on pense qu\'ils forment une ligne',
             fontsize=11, ha='center', color='C0')
    ax1.axvspan(z_lo, z_hi, color='C1', alpha=0.25)
    ax1.text((z_lo + z_hi) / 2, -0.022 * curve.max() / 0.28, 'zoom',
             ha='center', color='C1', fontsize=10)

    # --- bas : zoom, chaque point est une valeur posée en x_j ---------------
    ax2.vlines(x_grid[sel], 0, curve[sel], color='C0', lw=1, alpha=0.6)
    ax2.plot(x_grid[sel], curve[sel], 'o', ms=6, color='C0')
    for j in sel[::label_every]:
        ax2.text(x_grid[j], curve[j] + 0.015 * curve[sel].max(), f'j={j}',
                 ha='center', fontsize=9, color='C3')
    y_dx = 0.08 * curve[sel].max()
    ax2.annotate('', xy=(x_grid[sel[0]], y_dx), xytext=(x_grid[sel[1]], y_dx),
                 arrowprops=dict(arrowstyle='<->', color='k'))
    ax2.text(x_grid[sel[0]] + dx / 2, y_dx * 1.3,
             fr'$\Delta x = {x_grid[-1] - x_grid[0]:g}/{M - 1} \approx {dx:.4f}$',
             ha='left', fontsize=11, bbox=dict(fc='white', ec='none', pad=1.5))
    ax2.set_xlim(z_lo - 0.01, z_hi + 0.01)
    ax2.set_ylim(0, curve[sel].max() * 1.12)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\phi(x_j)$')
    ax2.set_title(fr'Zoom sur $x \in [{z_lo:g}, {z_hi:g}]$ : {len(sel)} des {M} points, '
                  fr'chaque point est la valeur de $\phi$ en '
                  fr'$x_j = {x_grid[0]:g} + j\,\Delta x$', loc='left')
    return fig

# ---------------------------------------------------------------------------
# Partie 1: Réduction de dimension
# ---------------------------------------------------------------------------
def mse_table(d_values, mse_train, mse_test, cum_var=None):
    """Tableau des EQM de reconstruction (train et test) pour chaque d.

    Parameters
    ----------
    d_values : liste des dimensions latentes d
    mse_train, mse_test : listes des EQM de reconstruction (même ordre que d_values)
    cum_var : array ou None, variance expliquée cumulée de la PCA (élément k-1
              = k composantes) ; si fourni, ajoute la colonne correspondante

    Returns
    -------
    Styler : tableau pandas formaté pour affichage dans le notebook
    """
    table = pd.DataFrame({'EQM train': mse_train, 'EQM test': mse_test},
                         index=pd.Index(d_values, name='d'))
    # rapport > 1 : la reconstruction est moins bonne sur des courbes non vues
    table['rapport test / train'] = table['EQM test'] / table['EQM train']
    formats = {'EQM train': '{:.3e}', 'EQM test': '{:.3e}',
               'rapport test / train': '{:.3g}'}
    if cum_var is not None:
        table.insert(0, 'variance expliquée cumulée',
                     [cum_var[d - 1] for d in d_values])
        formats['variance expliquée cumulée'] = '{:.2%}'
    return table.style.format(formats)


def plot_mse_vs_d(mse_curves, cum_var=None, threshold=0.99, ax=None):
    """EQM de reconstruction en fonction de d (échelle semi-log), avec en option la
    variance expliquée cumulée de la PCA sur un second axe.

    Parameters
    ----------
    mse_curves : dict  étiquette -> (liste des d, liste des EQM)
                 ex. {'PCA (test)': (ds, eqm_test), 'PCA (train)': (ds, eqm_train)}
                 Une étiquette contenant « train » est tracée en pointillés.
    cum_var : array ou None, variance expliquée cumulée (élément k-1 = k
              composantes), tracée sur l'axe de droite
    threshold : seuil de variance à repérer (ligne horizontale + valeur de d)
    ax : Axes matplotlib ou None

    Returns
    -------
    ax : Axes de gauche (EQM)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    for k, (label, (ds, mse)) in enumerate(mse_curves.items()):
        ls = '--' if 'train' in label else '-'
        ax.semilogy(ds, mse, ls, marker='o', color=f'C{k}', label=label)
    ax.set_xlabel('dimension latente $d$')
    ax.set_ylabel('EQM de reconstruction (échelle log)')
    ax.grid(alpha=0.3, which='both')

    if cum_var is not None:
        ax2 = ax.twinx()
        n = np.arange(1, len(cum_var) + 1)
        ax2.plot(n, cum_var, color='0.4', lw=1.5, label='variance expliquée cumulée')
        d_thr = int(np.searchsorted(cum_var, threshold) + 1)
        ax2.axhline(threshold, color='0.4', ls=':', lw=1)
        ax2.axvline(d_thr, color='0.4', ls=':', lw=1)
        ax2.annotate(fr'{threshold:.0%} de variance : $d = {d_thr}$',
                     xy=(d_thr, threshold), xytext=(d_thr + 3, threshold - 0.18),
                     arrowprops=dict(arrowstyle='->', color='0.3'), color='0.3')
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel('variance expliquée cumulée', color='0.4')
        # une seule légende pour les deux axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='center right')
        ax.set_xlim(0, max(max(ds) for ds, _ in mse_curves.values()) + 2)
    else:
        ax.legend()
    ax.set_title('Erreur de reconstruction en fonction de la dimension latente')
    return ax


def plot_reconstructions(x_grid, y_true, recons, titles=None):
    """Compare des courbes et leurs reconstructions, un panneau par courbe.

    Parameters
    ----------
    x_grid : array (M,)
    y_true : array (k, M), les k courbes originales
    recons : dict  étiquette -> array (k, M), reconstructions à superposer
             ex. {'PCA d=2': y_hat2, 'PCA d=10': y_hat10}
    titles : liste de k titres ou None

    Returns
    -------
    fig : Figure matplotlib
    """
    y_true = np.atleast_2d(y_true)
    k = len(y_true)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4), squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.plot(x_grid, y_true[i], 'k', lw=2.2, label='courbe originale')
        for c, (label, y_hat) in enumerate(recons.items()):
            ax.plot(x_grid, np.atleast_2d(y_hat)[i], color=f'C{c + 1}', lw=1.6,
                    ls=['-', '--', '-.', ':'][c % 4], label=label)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xlabel(r'$x$')
        if titles is not None:
            ax.set_title(titles[i])
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(r'$\phi(x)$')
    axes[0][0].legend(fontsize=9)
    fig.tight_layout()
    return fig

def plot_pca_components(x_grid, pca_reducer, n_components=4):
    """Courbe moyenne et premières composantes principales d'une PCA ajustée.

    Montre les « courbes de base » que la PCA combine linéairement :
    y_hat = moyenne + z_1 * composante_1 + ... + z_d * composante_d.

    Parameters
    ----------
    x_grid : array (M,)
    pca_reducer : PCAReducer déjà ajusté (attribut .model = sklearn PCA)
    n_components : nombre de composantes à tracer

    Returns
    -------
    fig : Figure matplotlib
    """
    pca = pca_reducer.model
    n = min(n_components, pca.n_components_)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4),
                                   gridspec_kw={'width_ratios': [1, 2]})
    ax1.plot(x_grid, pca.mean_, 'k', lw=2)
    ax1.set_title('Courbe moyenne du train')
    ax1.set_xlabel(r'$x$')
    ax1.grid(alpha=0.3)
    for k in range(n):
        ax2.plot(x_grid, pca.components_[k], lw=1.6,
                 label=f'composante {k + 1} ({pca.explained_variance_ratio_[k]:.1%})')
    ax2.axhline(0, color='0.6', lw=0.8)
    ax2.set_title(f'Les {n} premières composantes principales (part de variance)')
    ax2.set_xlabel(r'$x$')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig




def compare_table(d_values, mse_pca, mse_ae, time_pca=None, time_ae=None):
    """Tableau de comparaison PCA / autoencodeur (EQM test) pour chaque d.

    Parameters
    ----------
    d_values : liste des dimensions latentes d
    mse_pca, mse_ae : EQM de reconstruction sur le test (même ordre que d_values)
    time_pca, time_ae : durées d'entraînement en secondes (optionnelles)

    Returns
    -------
    Styler pandas 
    """
    table = pd.DataFrame({'EQM PCA': mse_pca, 'EQM AE': mse_ae},
                         index=pd.Index(d_values, name='d'))
    # facteur > 1 : l'autoencodeur fait mieux que la PCA
    table['facteur PCA / AE'] = table['EQM PCA'] / table['EQM AE']
    formats = {'EQM PCA': '{:.3e}', 'EQM AE': '{:.3e}', 'facteur PCA / AE': '{:.1f}'}
    if time_pca is not None:
        table['temps PCA (s)'] = time_pca
        formats['temps PCA (s)'] = '{:.2f}'
    if time_ae is not None:
        table['temps AE (s)'] = time_ae
        formats['temps AE (s)'] = '{:.0f}'
    return table.style.format(formats)


def plot_training_history(histories, ax=None):
    """Courbes d'apprentissage (perte de validation) de plusieurs autoencodeurs.

    Parameters
    ----------
    histories : dict  d -> history_ (dict avec 'loss' et 'val_loss', par époque)
    ax : Axes matplotlib ou None

    Returns
    -------
    ax : Axes matplotlib
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    for k, (d, h) in enumerate(histories.items()):
        ax.semilogy(h['val_loss'], color=f'C{k}', lw=1.5, label=f'd = {d} (validation)')
        ax.semilogy(h['loss'], color=f'C{k}', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('époque')
    ax.set_ylabel('EQM (échelle log)')
    ax.set_title("Courbes d'apprentissage de l'autoencodeur "
                 "(trait plein : validation, pointillés : apprentissage)")
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=9)
    return ax


def plot_latent_space(Z, X, title='Espace latent'):
    """Codes latents 2-D colorés par mu (gauche) puis par sigma^2 (droite).

    Parameters
    ----------
    Z : array (n, 2), codes latents (ex. reducer.transform(y_train))
    X : array (n, 2), paramètres (mu, sigma^2) des mêmes courbes
    title : titre général

    Returns
    -------
    fig : Figure matplotlib
    """
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(Z[:, 0], Z[:, 1], c=X[:, 0], cmap='coolwarm', s=8)
    fig.colorbar(sc, ax=axes[0], label=r'$\mu$')
    axes[0].set_title(r'coloré par $\mu$')
    # sigma^2 est très asymétrique (loi Gamma) : échelle de couleur logarithmique
    sc = axes[1].scatter(Z[:, 0], Z[:, 1], c=X[:, 1], cmap='viridis', s=8,
                         norm=LogNorm(vmin=X[:, 1].min(), vmax=X[:, 1].max()))
    fig.colorbar(sc, ax=axes[1], label=r'$\sigma^2$ (échelle log)')
    axes[1].set_title(r'coloré par $\sigma^2$')
    for ax in axes:
        ax.set_xlabel(r'$z_1$')
        ax.set_ylabel(r'$z_2$')
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def models_table(records):
    """Tableau comparatif des méthodes de réduction de dimension.

    Parameters
    ----------
    records : liste de dicts, une ligne par méthode ; la clé 'méthode' sert
              d'index, les autres clés deviennent des colonnes. Les colonnes
              dont le nom commence par 'EQM' sont affichées en notation
              scientifique, celles qui commencent par 'temps' avec 1 décimale.
              Une valeur absente (None ou NaN) s'affiche « — ».

    Returns
    -------
    Styler pandas
    """
    table = pd.DataFrame(records).set_index('méthode')
    formats = {c: '{:.2e}' for c in table.columns if c.startswith('EQM')}
    formats.update({c: '{:.1f}' for c in table.columns if c.startswith('temps')})
    return table.style.format(formats, na_rep='—')




# ---------------------------------------------------------------------------
# Partie 2 — GPR dans l'espace latent
# ---------------------------------------------------------------------------
def results_table(records, index):
    """Tableau de résultats générique, formaté selon le nom des colonnes"""
    table = pd.DataFrame(records).set_index(index)
    formats = {}
    for c in table.columns:
        if c.startswith(('EQM', 'RMSE')):
            formats[c] = '{:.3e}'
        elif c.startswith('temps'):
            formats[c] = '{:.1f}'
        elif c.startswith('p-valeur'):
            formats[c] = '{:.1e}'
        elif c.startswith('part'):
            formats[c] = '{:.0%}'
        elif c.startswith(('ratio', 'facteur')):
            formats[c] = '{:.2f}'
    return table.style.format(formats, na_rep='—')


def plot_decoded_vs_d(results):
    """Q2.3 — à gauche : EQM après GPR + décodage (plein) et EQM de reconstruction
    seule (pointillés) ; à droite : RMSE latente relative, en fonction de d.

    results : dict 
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    for k, (name, r) in enumerate(results.items()):
        ax1.semilogy(r['d'], r['EQM décodée'], '-o', color=f'C{k}',
                     label=f'{name} : GPR + décodage')
        ax1.semilogy(r['d'], r['EQM reconstruction'], '--', marker='s', ms=4,
                     color=f'C{k}', alpha=0.6, label=f'{name} : reconstruction seule')
        ax2.plot(r['d'], r['RMSE latente relative'], '-o', color=f'C{k}', label=name)
    ax1.set_xlabel('dimension latente $d$')
    ax1.set_ylabel('EQM de courbe sur le test (échelle log)')
    ax1.set_title('Erreur du méta-modèle complet et erreur de reconstruction')
    ax1.grid(alpha=0.3, which='both')
    ax1.legend(fontsize=9)
    ax2.set_xlabel('dimension latente $d$')
    ax2.set_ylabel('RMSE latente / écart-type des codes')
    ax2.set_title("Erreur de la GPR dans l'espace latent (relative)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    return fig