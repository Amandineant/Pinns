# NumPy sert à manipuler la grille de températures
import numpy as np

# Matplotlib sert à afficher la carte de température
import matplotlib.pyplot as plt

# On récupère les paramètres définis dans parameters.py
from parameters import (
    L_X,
    L_Y,
    T_AMBIANTE,
    T_OBJET,
    OBJET_CENTRE_X,
    OBJET_CENTRE_Y,
    OBJET_TAILLE,
)

# Nombre de points dans chaque direction
N_X = 51
N_Y = 51

# Coordonnées des points entre 0 et 1 mètre
x = np.linspace(0, L_X, N_X)
y = np.linspace(0, L_Y, N_Y)

# Création de la grille 2D
X, Y = np.meshgrid(x, y)

# Au départ, toute la pièce est à 20 °C
temperature = np.full((N_Y, N_X), T_AMBIANTE)

# Calcul de la moitié de la taille de l'objet
demi_taille = OBJET_TAILLE / 2

# Repérage des points situés dans l'objet carré
masque_objet = (
    (X >= OBJET_CENTRE_X - demi_taille)
    & (X <= OBJET_CENTRE_X + demi_taille)
    & (Y >= OBJET_CENTRE_Y - demi_taille)
    & (Y <= OBJET_CENTRE_Y + demi_taille)
)

# Les points appartenant à l'objet sont placés à 80 °C
temperature[masque_objet] = T_OBJET

# Affichage de la condition initiale
plt.imshow(
    temperature,
    origin="lower",
    extent=[0, L_X, 0, L_Y],
    cmap="coolwarm",
    vmin=T_AMBIANTE,
    vmax=T_OBJET,
)

plt.colorbar(label="Température (°C)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Température initiale à t = 0")
plt.show()