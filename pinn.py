"""Réseau de base et dérivées physiques ; aucun entraînement à ce stade."""
import torch
from torch import nn
from nondimensionalization import BETA_X, BETA_Y


class PINN(nn.Module):
    """Entrées (N,3) : X,Y,tau ; sortie (N,1) : température réduite u.

    Trois couches cachées de 32 neurones, tanh puis sortie linéaire.
    La sortie n'est pas bornée artificiellement ; les futures pertes guideront
    le réseau vers les conditions physiques.
    """
    def __init__(self):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1))

    def forward(self, points):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Les entrées doivent avoir la forme (N,3) : X,Y,tau.")
        return self.reseau(points)


def derivees_temperature(modele, points):
    """Calcule u, u_tau, u_XX et u_YY par autograd.

    points doit contenir X,Y,tau normalisés et requires_grad=True.
    create_graph=True conserve la possibilité de dériver les futures pertes
    par rapport aux paramètres du réseau, y compris via les dérivées secondes.
    Le modèle doit traiter chaque point indépendamment (comme notre MLP).
    """
    if not points.requires_grad:
        raise ValueError("Les points physiques doivent avoir requires_grad=True.")
    u = modele(points)
    premieres = torch.autograd.grad(
        u, points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = premieres[:, 0:1]
    u_y = premieres[:, 1:2]
    u_tau = premieres[:, 2:3]
    u_xx = torch.autograd.grad(
        u_x, points, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, points, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return {"u": u, "u_tau": u_tau, "u_xx": u_xx, "u_yy": u_yy}


def residu_physique(modele, points):
    """Résidu normalisé, pas encore une perte : u_tau-beta_x*u_XX-beta_y*u_YY.

    Les coefficients viennent du module commun : 0.072 pour les deux axes
    dans le scénario actuel. On ne dérive pas en mètres ou en secondes ici.
    """
    d = derivees_temperature(modele, points)
    return d["u_tau"] - BETA_X * d["u_xx"] - BETA_Y * d["u_yy"]


if __name__ == "__main__":
    from sampling import generer_points
    torch.manual_seed(0)
    modele = PINN()
    groupes = generer_points()
    print(modele)
    print("Paramètres entraînables :", sum(p.numel() for p in modele.parameters()))
    for nom, groupe in groupes.items():
        with torch.no_grad():
            sortie = modele(groupe["points"])
        print(nom, "entrée", tuple(groupe["points"].shape),
              "sortie", tuple(sortie.shape),
              "valeurs finies", bool(torch.isfinite(sortie).all()))
    points = groupes["RES"]["points"][:16].detach().requires_grad_(True)
    for nom, valeur in derivees_temperature(modele, points).items():
        print(nom, tuple(valeur.shape), "fini", bool(torch.isfinite(valeur).all()))
    residu = residu_physique(modele, points)
    print("Résidu", tuple(residu.shape), "fini", bool(torch.isfinite(residu).all()))
    print("Réseau non entraîné : les températures ne sont pas encore significatives.")
