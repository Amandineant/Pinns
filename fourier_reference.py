"""Solution indépendante du carré chaud par série de Fourier, t>0."""
import numpy as np
from parameters import (L_X, L_Y, ALPHA, T_AMBIANTE, T_OBJET,
                        OBJET_CENTRE_X, OBJET_CENTRE_Y, OBJET_TAILLE)
from initial_condition import masque_objet


def temperature_fourier(x, y, temps, n_modes=100, longueur=L_X, largeur=L_Y,
                         alpha=ALPHA, temperature_ambiante=T_AMBIANTE,
                         temperature_objet=T_OBJET, centre_x=OBJET_CENTRE_X,
                         centre_y=OBJET_CENTRE_Y, taille=OBJET_TAILLE):
    """Carte (len(y),len(x)), intégrales exactes du carré et murs Dirichlet.

    La chaleur 2D est séparable pour ce carré : produit de deux séries 1D.
    Pas d'évaluation à t=0 (discontinuité et convergence de Gibbs).
    """
    if not np.isfinite(temps) or temps <= 0:
        raise ValueError("La référence Fourier est évaluée pour t>0.")
    if not isinstance(n_modes, int) or n_modes <= 0 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Nombre de modes et diffusivité invalides.")
    masque_objet(np.array([0.]), np.array([0.]), centre_x, centre_y, taille, longueur, largeur)
    def serie(coordonnees, centre, dimension):
        n = np.arange(1, n_modes + 1, dtype=float)
        k = np.pi * n / dimension
        bas, haut = centre - taille / 2, centre + taille / 2
        coefficients = 2 * (np.cos(k * bas) - np.cos(k * haut)) / (np.pi * n)
        return (coefficients * np.exp(-alpha * k**2 * temps)) @ np.sin(k[:, None] * np.asarray(coordonnees)[None, :])
    return temperature_ambiante + (temperature_objet - temperature_ambiante) * np.outer(
        serie(y, centre_y, largeur), serie(x, centre_x, longueur))
