# -*- coding: utf-8 -*-

"""Réducteurs de dimension — squelette à compléter (Partie 1).

Toutes les classes doivent respecter l'interface commune :

    reducer.fit(y_train)                 # apprentissage (sur le train seulement)
    z = reducer.transform(y)             # encodage  R^M -> R^d
    y_hat = reducer.inverse_transform(z) # décodage  R^d -> R^M

ainsi que l'attribut `latent_dim` (dimension d de l'espace latent).
"""

import os
import time

import numpy as np
from sklearn.decomposition import PCA, KernelPCA

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')   # masque les messages d'information de TensorFlow
import keras
from keras import layers



def _as_2d(a):
    """Renvoie (tableau 2-D, booléen « l'entrée était 1-D »).

    Permet de passer d'un vecteur 1-D à un tableau 2-D (une seule ligne) pour l'interface
    des réducteurs de dimension. Si l'entrée est déjà 2-D, on ne fait rien.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a[None, :], True
    return a, False



class PCAReducer:
    """PCA scikit-learn avec l'interface du projet.

    Attributs
    ---------
    latent_dim : d, dimension de l'espace latent
    model : objet sklearn.decomposition.PCA sous-jacent
    decoder : appelable z -> courbes (identique à inverse_transform), pour
              avoir la même interface que l'autoencodeur
    """

    def __init__(self, n_components=2):
        self.latent_dim = n_components
        self.model = PCA(n_components=n_components, svd_solver='full') #svd_solver='full' : SVD exacte et déterministe.

    def fit(self, y_train):
        """Ajuste la PCA sur les courbes d'entraînement y_train (n, M)."""
        y_train, _ = _as_2d(y_train)
        t0 = time.perf_counter()
        self.model.fit(y_train)
        self.fit_time_ = time.perf_counter() - t0   # durée de l'ajustement (s)
        return self
    
    def transform(self, y):
        """Encode des courbes (n, M) ou (M,) en codes latents (n, d) ou (d,)."""
        y, single = _as_2d(y)
        z = self.model.transform(y)
        return z[0] if single else z

    def inverse_transform(self, z):
        """Décode des codes latents (n, d) ou (d,) en courbes (n, M) ou (M,)."""
        z, single = _as_2d(z)
        y_hat = self.model.inverse_transform(z)
        return y_hat[0] if single else y_hat

    @property
    def decoder(self):
        """Décodeur seul, réutilisable : decoder(z) -> courbes."""
        return self.inverse_transform


def cumulative_explained_variance(y_train, n_max=100):
    """Ratio de variance expliquée cumulée des n premières composantes.

    Returns
    -------
    array (n_max,) : élément k-1 = part de variance portée par les k
                     premières composantes principales
    """
    pca = PCA(n_components=n_max, svd_solver='full').fit(y_train)
    return np.cumsum(pca.explained_variance_ratio_)

class KernelPCAReducer:
    """Kernel PCA (noyau RBF).

    Principe : on applique une PCA non pas aux courbes elles-mêmes, mais à
    leurs similarités k(y, y') = exp(-gamma ||y - y'||^2). Cela permet de
    capter une structure non linéaire.

    Contrairement à la PCA, il n'existe pas de décodage exact : scikit-learn
    APPREND le décodage (z -> courbe) par une régression ridge à noyau
    (fit_inverse_transform=True), régularisée par `alpha`.

    Paramètres
    ----------
    n_components : d, dimension latente
    gamma : largeur du noyau RBF (s'applique aux distances entre courbes)
    alpha : régularisation de la régression qui apprend le décodage
    """

    def __init__(self, n_components=2, gamma=0.01, alpha=1e-6):
        self.latent_dim = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.model = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma,
                               fit_inverse_transform=True, alpha=alpha,
                               eigen_solver='dense')

    def fit(self, y_train):
        y_train, _ = _as_2d(y_train)
        t0 = time.perf_counter()
        self.model.fit(y_train)
        self.fit_time_ = time.perf_counter() - t0
        return self

    def transform(self, y):
        y, single = _as_2d(y)
        z = self.model.transform(y)
        return z[0] if single else z

    def inverse_transform(self, z):
        z, single = _as_2d(z)
        y_hat = self.model.inverse_transform(z)
        return y_hat[0] if single else y_hat

    @property
    def decoder(self):
        return self.inverse_transform


def tune_kernel_pca(y_train, n_components, gammas=(0.005, 0.01, 0.02, 0.05),
                    alphas=(1e-7, 1e-6, 1e-5, 1e-4), val_fraction=0.2, seed=42):
    """Choisit (gamma, alpha) de la Kernel PCA sur une validation tirée du train.

    Le jeu de test n'est jamais utilisé : on découpe y_train en 80 % pour
    l'ajustement et 20 % pour mesurer l'EQM de reconstruction, puis on
    réajuste le meilleur modèle sur tout y_train.

    Returns
    -------
    best : KernelPCAReducer ajusté sur tout y_train avec les meilleurs (gamma, alpha)
    scores : dict (gamma, alpha) -> EQM de validation
    """
    y_train, _ = _as_2d(y_train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_train))
    n_val = int(round(val_fraction * len(y_train)))
    y_val, y_fit = y_train[perm[:n_val]], y_train[perm[n_val:]]
    scores = {}
    for g in gammas:
        for a in alphas:
            r = KernelPCAReducer(n_components, gamma=g, alpha=a).fit(y_fit)
            scores[(g, a)] = reconstruction_mse(r, y_val)
    g_best, a_best = min(scores, key=scores.get)
    best = KernelPCAReducer(n_components, gamma=g_best, alpha=a_best).fit(y_train)
    return best, scores



class KerasDecoder:
    """Décodeur seul de l'autoencodeur, utilisable sans l'encodeur.

    S'appelle comme une fonction : decoder(z) -> courbes (tableaux numpy).
    `decoder.predict(z)` fonctionne aussi. 

    Le réseau travaille sur des courbes divisées par `scale` (normalisation
    faite pendant l'entraînement) : on remultiplie ici par `scale` pour
    rendre des courbes dans leur unité d'origine.
    """

    def __init__(self, model, scale):
        self.model = model
        self.scale = scale

    def __call__(self, z):
        z, single = _as_2d(z)
        y_hat = self.model.predict(z, batch_size=4096, verbose=0) * self.scale
        y_hat = np.asarray(y_hat, dtype=float)
        return y_hat[0] if single else y_hat

    def predict(self, z):
        return self(z)


class AutoEncoderReducer:
    """Autoencodeur dense, implémenté avec TensorFlow / Keras.

    Architecture :  M -> 128 -> 64 -> d -> 64 -> 128 -> M
        encodeur : Dense(128, SiLU) -> Dense(64, SiLU) -> Dense(d)   (code linéaire)
        décodeur : Dense(64, SiLU) -> Dense(128, SiLU) -> Dense(M)   (sortie linéaire)
    Loss : MSE. Optimiseur : Adam(1e-3).

    Entraînement :
        - les courbes sont normalisées par leur écart-type (pour que la loss soit comparable à la PCA) ;
        - 20 % du train servent de validation (tirés au hasard, graine fixée) ;
        - early stopping sur la perte de validation (patience `patience`),
          avec retour aux meilleurs poids ;
        - le pas d'apprentissage est divisé par 2 quand la validation stagne
          (ReduceLROnPlateau), ce qui affine la fin de l'entraînement.

    Attributs après fit
    -------------------
    encoder, decoder_model : réseaux Keras de l'encodeur et du décodeur
    decoder : KerasDecoder, décodeur seul réutilisable (Parties 2 et 3)
    history_ : dict, pertes train / validation par époque (unités d'origine)
    fit_time_ : float, durée de l'entraînement (secondes)
    n_epochs_ : int, nombre d'époques effectuées (early stopping compris)
    """

    def __init__(self, nbx=800, latent_dim=2, backend="keras", epochs=300,
                 batch_size=64, seed=42, learning_rate=1e-3, patience=40, #Nombre d'époques sans amélioration avant arrêt anticipé,
                 val_fraction=0.2):
        if backend not in ("keras", "tensorflow"):
            raise ValueError("Seul le backend 'keras' (TensorFlow) est implémenté.")
        self.nbx = nbx
        self.latent_dim = latent_dim
        self.backend = backend
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.learning_rate = learning_rate
        self.patience = patience
        self.val_fraction = val_fraction
        self.decoder = None   

    def _build(self):
        """Construit l'encodeur et le décodeur (deux réseaux séparés)."""
        d, M = self.latent_dim, self.nbx
        self.encoder = keras.Sequential([
            keras.Input(shape=(M,)),
            layers.Dense(128, activation='silu'),
            layers.Dense(64, activation='silu'),
            layers.Dense(d),                      # code latent (pas d'activation)
        ], name='encoder')
        self.decoder_model = keras.Sequential([
            keras.Input(shape=(d,)),
            layers.Dense(64, activation='silu'),
            layers.Dense(128, activation='silu'),
            layers.Dense(M),                      # sortie linéaire
        ], name='decoder')
        # autoencodeur complet = encodeur puis décodeur (poids partagés)
        self.autoencoder = keras.Sequential([self.encoder, self.decoder_model],
                                            name='autoencoder')
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse')

    def fit(self, y_train):
        """Entraîne l'autoencodeur sur y_train (n, M) : 80 % apprentissage,
        20 % validation pour l'early stopping."""
        keras.utils.set_random_seed(self.seed)   # graines Python, numpy et TensorFlow
        y_train, _ = _as_2d(y_train)
        self.scale_ = float(y_train.std())
        y_scaled = y_train / self.scale_

        # découpage apprentissage / validation (aléatoire, graine fixée)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(y_scaled))
        n_val = int(round(self.val_fraction * len(y_scaled)))
        y_val, y_fit = y_scaled[perm[:n_val]], y_scaled[perm[n_val:]]

        self._build()
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=max(self.patience // 3, 1),
                                              min_lr=1e-5),
        ]
        t0 = time.perf_counter()
        hist = self.autoencoder.fit(y_fit, y_fit, validation_data=(y_val, y_val),
                                    epochs=self.epochs, batch_size=self.batch_size,
                                    shuffle=True, callbacks=callbacks, verbose=0)
        self.fit_time_ = time.perf_counter() - t0
        self.n_epochs_ = len(hist.history['loss'])
        # pertes ramenées dans l'unité des courbes (EQM comparables à la PCA)
        self.history_ = {k: np.array(v) * self.scale_ ** 2
                         for k, v in hist.history.items() if k in ('loss', 'val_loss')}
        self.decoder = KerasDecoder(self.decoder_model, self.scale_)
        return self

    def transform(self, y):
        """Codes latents (n, d) des courbes (n, M) ; (d,) pour une seule courbe."""
        y, single = _as_2d(y)
        z = self.encoder.predict(y / self.scale_, batch_size=4096, verbose=0)
        z = np.asarray(z, dtype=float)
        return z[0] if single else z

    def inverse_transform(self, z):
        """Reconstruction des courbes par le décodeur seul."""
        return self.decoder(z)



def reconstruction_mse(reducer, y):
    """EQM moyenne de reconstruction d'un jeu de courbes.

    Returns
    -------
    float : (1/(n*M)) * sum_i ||y_i - reducer.inverse_transform(transform(y_i))||^2
    """
    y, _ = _as_2d(y)
    y_hat = np.asarray(reducer.inverse_transform(reducer.transform(y)))
    return float(np.mean((y - y_hat) ** 2))
