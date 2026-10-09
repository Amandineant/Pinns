"""Une définition commune du carré chaud pour le solveur et le PINN."""
import numpy as np
from parameters import (L_X, L_Y, T_AMBIANTE, T_OBJET,
                        OBJET_CENTRE_X, OBJET_CENTRE_Y, OBJET_TAILLE)


def masque_objet(x, y, centre_x=OBJET_CENTRE_X,
                 centre_y=OBJET_CENTRE_Y, taille=OBJET_TAILLE,
                 longueur=L_X, largeur=L_Y):
    """Accepte des tableaux NumPy ou des tenseurs PyTorch, en mètres.

    L'objet doit rester strictement à l'intérieur pour ne pas chauffer les murs.
    """
    if not all(np.isfinite(v) for v in (centre_x, centre_y, taille, longueur, largeur)):
        raise ValueError("Les paramètres géométriques doivent être finis.")
    demi = taille / 2
    if not (taille > 0 and demi < centre_x < longueur - demi
            and demi < centre_y < largeur - demi):
        raise ValueError("Le carré doit rester strictement à l'intérieur de la pièce.")
    return ((x >= centre_x - demi) & (x <= centre_x + demi)
            & (y >= centre_y - demi) & (y <= centre_y + demi))


def temperature_initiale(x, y, temperature_ambiante=T_AMBIANTE,
                         temperature_objet=T_OBJET, **geometrie):
    """Température en °C, calculée à partir de l'unique masque de l'objet."""
    return (temperature_ambiante + (temperature_objet - temperature_ambiante)
            * masque_objet(x, y, **geometrie))


def temperature_initiale_discrete(X, Y, temperature_ambiante=T_AMBIANTE,
                                  temperature_objet=T_OBJET,
                                  centre_x=OBJET_CENTRE_X, centre_y=OBJET_CENTRE_Y,
                                  taille=OBJET_TAILLE, longueur=L_X, largeur=L_Y):
    """Fractions de couverture pour un carré aligné avec la grille.

    Bords : 1/2 ; coins : 1/4, appliqués à l'écart de température.
    La condition ponctuelle du PINN reste définie par temperature_initiale.
    Une géométrie non alignée est refusée plutôt que d'utiliser ces fractions
    incorrectement ; la couverture réelle des cellules sera une extension.
    """
    masque_objet(X, Y, centre_x, centre_y, taille, longueur, largeur)
    limites_x = (centre_x - taille / 2, centre_x + taille / 2)
    limites_y = (centre_y - taille / 2, centre_y + taille / 2)
    x, y = X[0, :], Y[:, 0]
    for coordonnees, limites in ((x, limites_x), (y, limites_y)):
        if any(not np.any(np.isclose(coordonnees, bord, rtol=0, atol=1e-12))
               for bord in limites):
            raise ValueError("Le carré doit être aligné avec la grille pour la correction 1/2, 1/4.")
    def fraction(coordonnees, limites):
        bas, haut = limites
        bord = (np.isclose(coordonnees, bas, rtol=0, atol=1e-12)
                | np.isclose(coordonnees, haut, rtol=0, atol=1e-12))
        resultat = ((coordonnees > bas) & (coordonnees < haut)).astype(float)
        resultat[bord] = .5
        return resultat
    couverture = fraction(X, limites_x) * fraction(Y, limites_y)
    return temperature_ambiante + couverture * (temperature_objet - temperature_ambiante)


def creer_grille(n_x=51, n_y=51, longueur=L_X, largeur=L_Y):
    """Coordonnées physiques ; les cartes ont la forme (n_y, n_x)."""
    if (not isinstance(n_x, int) or not isinstance(n_y, int)
            or n_x < 3 or n_y < 3):
        raise ValueError("La grille doit avoir au moins 3 points par direction.")
    if not (np.isfinite(longueur) and np.isfinite(largeur)
            and longueur > 0 and largeur > 0):
        raise ValueError("Les dimensions doivent être positives et finies.")
    x = np.linspace(0, longueur, n_x)
    y = np.linspace(0, largeur, n_y)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y, X, Y = creer_grille()
    plt.imshow(temperature_initiale(X, Y), origin="lower",
               extent=[0, L_X, 0, L_Y], cmap="coolwarm",
               vmin=T_AMBIANTE, vmax=T_OBJET)
    plt.colorbar(label="Température (°C)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Température initiale à t = 0")
    plt.show()
