import torch
import matplotlib.pyplot as plt

from parameters import (
    OBJET_CENTRE_X,
    OBJET_CENTRE_Y,
    OBJET_TAILLE,
)

# Toujours générer les mêmes points pendant nos tests
torch.manual_seed(0)

# Nombre de points utilisés pour la condition initiale
N_IC = 1000

# Positions aléatoires dans la pièce
x_ic = torch.rand(N_IC, 1)
y_ic = torch.rand(N_IC, 1)

# Tous ces points correspondent à l'instant initial t = 0
t_ic = torch.zeros(N_IC, 1)

# Limites de l'objet carré
demi_taille = OBJET_TAILLE / 2

x_min = OBJET_CENTRE_X - demi_taille
x_max = OBJET_CENTRE_X + demi_taille
y_min = OBJET_CENTRE_Y - demi_taille
y_max = OBJET_CENTRE_Y + demi_taille

# Recherche des points situés dans l'objet
dans_objet = (
    (x_ic >= x_min)
    & (x_ic <= x_max)
    & (y_ic >= y_min)
    & (y_ic <= y_max)
)

# Température réduite :
# 1 dans l'objet chaud et 0 dans le reste de la pièce
u_ic = torch.where(
    dans_objet,
    torch.ones_like(x_ic),
    torch.zeros_like(x_ic),
)

# Affichage de quelques informations
print("Nombre total de points :", N_IC)
print("Forme de x_ic :", x_ic.shape)
print("Tous les temps valent 0 :", torch.all(t_ic == 0).item())
print("Points dans l'objet :", dans_objet.sum().item())

# Affichage des points
plt.scatter(
    x_ic[:, 0],
    y_ic[:, 0],
    c=u_ic[:, 0],
    cmap="coolwarm",
    vmin=0,
    vmax=1,
)

plt.colorbar(label="Température réduite")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Points de la condition initiale")
plt.show()