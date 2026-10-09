"""Points IC, BC et RES en coordonnées normalisées (X, Y, tau)."""
import torch
from parameters import L_X, L_Y, OBJET_CENTRE_X, OBJET_CENTRE_Y, OBJET_TAILLE
from initial_condition import masque_objet


def generer_points(n_ic=1000, n_bc=1000, n_res=5000, seed=0, fraction_objet=0.25, **geometrie):
    """Retourne un dictionnaire de tenseurs float32 sur CPU.

    Chaque groupe contient 'points' de forme (N,3), colonnes X,Y,tau.
    IC et BC contiennent aussi 'valeurs' de forme (N,1), température réduite.
    IC garantit fraction_objet des points dans le carré (25 % par défaut).
    'poids' corrige ce tirage stratifié : la future perte IC doit utiliser
    mean(poids * (prediction - valeurs)**2) pour représenter l'aire du domaine.
    BC contient autant de points par mur ; n_bc doit être multiple de 4.
    RES contient les points intérieurs avec requires_grad=True pour autograd.
    Le générateur local évite de modifier la graine globale de PyTorch.
    """
    if any(not isinstance(n, int) or n <= 0 for n in (n_ic, n_bc, n_res)):
        raise ValueError("Les nombres de points doivent être des entiers positifs.")
    if n_bc % 4:
        raise ValueError("n_bc doit être multiple de 4 pour équilibrer les murs.")
    generateur = torch.Generator().manual_seed(seed)
    def tirage(n):
        # Évite les bornes exactes 0 et 1 pour les points intérieurs.
        epsilon = torch.finfo(torch.float32).eps
        return torch.rand(n, 3, generator=generateur).clamp(epsilon, 1 - epsilon)
    if n_ic < 2 or not 0 < fraction_objet < 1:
        raise ValueError("Il faut au moins 2 points IC et 0 < fraction_objet < 1.")
    centre_x = geometrie.get("centre_x", OBJET_CENTRE_X)
    centre_y = geometrie.get("centre_y", OBJET_CENTRE_Y)
    taille = geometrie.get("taille", OBJET_TAILLE)
    # Valide les paramètres avec le masque partagé avant de tirer les points.
    masque_objet(torch.zeros(1), torch.zeros(1), **geometrie)
    if geometrie.get("longueur", L_X) != L_X or geometrie.get("largeur", L_Y) != L_Y:
        raise ValueError("L'échantillonnage utilise les dimensions de parameters.py.")
    n_chaud = min(n_ic - 1, max(1, round(n_ic * fraction_objet)))
    n_froid = n_ic - n_chaud
    chaud = tirage(n_chaud)
    chaud[:, 0] = (centre_x - taille / 2 + chaud[:, 0] * taille) / L_X
    chaud[:, 1] = (centre_y - taille / 2 + chaud[:, 1] * taille) / L_Y
    # Tirage uniforme conditionné à être à l'extérieur du carré.
    morceaux = []
    restant = n_froid
    while restant:
        candidats = tirage(max(2 * restant, 32))
        dedans = masque_objet(candidats[:, 0] * L_X,
                             candidats[:, 1] * L_Y, **geometrie)
        retenus = candidats[~dedans][:restant]
        morceaux.append(retenus)
        restant -= len(retenus)
    froid = torch.cat(morceaux)
    ic = torch.cat((chaud, froid))
    ic[:, 2] = 0
    valeurs_ic = masque_objet(ic[:, 0:1] * L_X,
                             ic[:, 1:2] * L_Y, **geometrie).to(ic.dtype)
    fraction_aire = taille**2 / (L_X * L_Y)
    poids_ic = torch.cat((
        torch.full((n_chaud, 1), fraction_aire * n_ic / n_chaud),
        torch.full((n_froid, 1), (1 - fraction_aire) * n_ic / n_froid)))
    permutation = torch.randperm(n_ic, generator=generateur)
    ic, valeurs_ic, poids_ic = ic[permutation], valeurs_ic[permutation], poids_ic[permutation]
    bc = tirage(n_bc)
    par_mur = n_bc // 4
    bc[:par_mur, 0] = 0
    bc[par_mur:2 * par_mur, 0] = 1
    bc[2 * par_mur:3 * par_mur, 1] = 0
    bc[3 * par_mur:, 1] = 1
    res = tirage(n_res).requires_grad_(True)
    return {"IC": {"points": ic, "valeurs": valeurs_ic, "poids": poids_ic},
            "BC": {"points": bc, "valeurs": torch.zeros(n_bc, 1)},
            "RES": {"points": res}}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    groupes = generer_points()
    figure = plt.figure(figsize=(14, 4), layout="constrained")
    for index, (nom, groupe) in enumerate(groupes.items(), start=1):
        points = groupe["points"].detach().numpy()
        axe = figure.add_subplot(1, 3, index, projection="3d")
        couleurs = groupe["valeurs"].numpy().ravel() if nom == "IC" else None
        axe.scatter(points[:, 0], points[:, 1], points[:, 2], c=couleurs,
                    cmap="coolwarm" if couleurs is not None else None, s=3)
        axe.set(xlabel="X", ylabel="Y", zlabel="tau", title=f"{nom} : {len(points)} points")
        print(nom, groupe["points"].shape)
    print("Points IC dans l'objet :", int(groupes["IC"]["valeurs"].sum()))
    plt.show()
