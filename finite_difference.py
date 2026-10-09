import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    L_X,
    L_Y,
    T_AMBIANTE,
    T_OBJET,
    ALPHA,
    T_MAX,
    OBJET_CENTRE_X,
    OBJET_CENTRE_Y,
    OBJET_TAILLE,
)

# Taille de la grille
N_X = 51
N_Y = 51

# Distance entre deux points voisins
DX = L_X / (N_X - 1)
DY = L_Y / (N_Y - 1)

# Une étape représente 4 secondes
DT = 4.0

# Création des coordonnées de la grille
x = np.linspace(0, L_X, N_X)
y = np.linspace(0, L_Y, N_Y)

# Toute la pièce commence à la température ambiante
temperature = np.full((N_Y, N_X), T_AMBIANTE)

# Placement de l'objet chaud au centre
demi_taille = OBJET_TAILLE / 2

for j in range(N_Y):
    for i in range(N_X):
        dans_objet_x = (
            OBJET_CENTRE_X - demi_taille
            <= x[i]
            <= OBJET_CENTRE_X + demi_taille
        )

        dans_objet_y = (
            OBJET_CENTRE_Y - demi_taille
            <= y[j]
            <= OBJET_CENTRE_Y + demi_taille
        )

        if dans_objet_x and dans_objet_y:
            temperature[j, i] = T_OBJET

# Conservation de la carte initiale pour pouvoir la comparer
temperature_initiale = temperature.copy()

# Nombre d'étapes nécessaires pour atteindre une heure
nombre_etapes = int(T_MAX / DT)

# Répétition du calcul de diffusion
for etape in range(nombre_etapes):

    # Nouvelle grille calculée à partir de l'instant précédent
    nouvelle_temperature = temperature.copy()

    # Calcul uniquement sur les points intérieurs
    for j in range(1, N_Y - 1):
        for i in range(1, N_X - 1):

            derivee_x = (
                temperature[j, i + 1]
                - 2 * temperature[j, i]
                + temperature[j, i - 1]
            ) / DX**2

            derivee_y = (
                temperature[j + 1, i]
                - 2 * temperature[j, i]
                + temperature[j - 1, i]
            ) / DY**2

            nouvelle_temperature[j, i] = (
                temperature[j, i]
                + ALPHA * DT * (derivee_x + derivee_y)
            )

    # La nouvelle grille devient la grille actuelle
    temperature = nouvelle_temperature

print("Nombre d'étapes calculées :", nombre_etapes)
print("Température minimale finale :", temperature.min(), "°C")
print("Température maximale finale :", temperature.max(), "°C")

# Vérification des quatre murs
murs_corrects = (
    np.allclose(temperature[0, :], T_AMBIANTE)
    and np.allclose(temperature[-1, :], T_AMBIANTE)
    and np.allclose(temperature[:, 0], T_AMBIANTE)
    and np.allclose(temperature[:, -1], T_AMBIANTE)
)

# Vérification que les températures restent physiquement acceptables
temperatures_correctes = (
    temperature.min() >= T_AMBIANTE
    and temperature.max() <= T_OBJET
)

print("Les murs restent à 20 °C :", murs_corrects)
print("Les températures restent entre 20 et 80 °C :", temperatures_correctes)


# Création d'une figure contenant deux cartes
figure, axes = plt.subplots(1, 2, figsize=(11, 4))

# Carte au début
image_initiale = axes[0].imshow(
    temperature_initiale,
    origin="lower",
    extent=[0, L_X, 0, L_Y],
    cmap="coolwarm",
    vmin=T_AMBIANTE,
    vmax=T_OBJET,
)

axes[0].set_title("Température à t = 0 s")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")

# Carte après une heure
image_finale = axes[1].imshow(
    temperature,
    origin="lower",
    extent=[0, L_X, 0, L_Y],
    cmap="coolwarm",
    vmin=T_AMBIANTE,
    vmax=T_OBJET,
)

axes[1].set_title("Température à t = 3600 s")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")

# Une barre de couleur commune aux deux cartes
figure.colorbar(
    image_finale,
    ax=axes,
    label="Température (°C)",
)

plt.show()