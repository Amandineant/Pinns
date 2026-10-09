"""Pertes du scénario fixe : IC, BC, RES et combinaison pondérée.

Les poids par point IC corrigent le tirage spatial stratifié. Les coefficients
w_ic, w_bc, w_res règlent, séparément, l'équilibre entre les trois objectifs.
Aucun optimiseur ni entraînement dans ce module.
"""
import math
import torch
from pinn import residu_physique


def _verifier_points(points):
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("Les points doivent avoir la forme non vide (N,3).")


def _verifier_colonne(valeurs, points, nom):
    if valeurs.shape != (points.shape[0], 1):
        raise ValueError(f"{nom} doit avoir la forme (N,1), sans broadcasting.")


def loss_ic(modele, groupe):
    """mean(poids * (u_pred-u_cible)^2), scalaire différentiable.

    Avec a=aire_objet/aire_piece et q=n_objet/N : poids_objet=a/q,
    poids_exterieur=(1-a)/(1-q). Pour a=.04, q=.25 : .16 et 1.28.
    Cela estime a*mean(erreur² dans l'objet)+(1-a)*mean(erreur² dehors).
    """
    points, valeurs, poids = groupe["points"], groupe["valeurs"], groupe["poids"]
    _verifier_points(points)
    _verifier_colonne(valeurs, points, "valeurs IC")
    _verifier_colonne(poids, points, "poids IC")
    if not bool(torch.isfinite(poids).all()) or not bool((poids > 0).all()):
        raise ValueError("Les poids IC doivent être positifs et finis.")
    prediction = modele(points)
    _verifier_colonne(prediction, points, "prédiction IC")
    return (poids * (prediction - valeurs).square()).mean()


def loss_bc(modele, groupe):
    """mean((u_pred-u_mur)^2), avec u_mur=0 dans notre scénario."""
    points, valeurs = groupe["points"], groupe["valeurs"]
    _verifier_points(points)
    _verifier_colonne(valeurs, points, "valeurs BC")
    prediction = modele(points)
    _verifier_colonne(prediction, points, "prédiction BC")
    return (prediction - valeurs).square().mean()


def loss_res(modele, groupe):
    """mean(r²), r=u_tau-BETA_X*u_XX-BETA_Y*u_YY, entrées normalisées.

    Les points RES sont utilisés directement, sans detach ni changement de
    requires_grad, pour conserver les dérivées et la propagation des gradients.
    """
    points = groupe["points"]
    _verifier_points(points)
    if not points.requires_grad:
        raise ValueError("Les coordonnées RES doivent garder requires_grad=True.")
    return residu_physique(modele, points).square().mean()


def calculer_pertes(modele, groupes, w_ic=1., w_bc=1., w_res=1.):
    """Retourne les quatre tenseurs scalaires, sans détacher leur graphe.

    total=w_ic*IC+w_bc*BC+w_res*RES. Coefficients égaux à 1 par défaut.
    Des coefficients nuls permettent une ablation ; négatifs/non finis refusés.
    """
    if any(not math.isfinite(w) or w < 0 for w in (w_ic, w_bc, w_res)):
        raise ValueError("Les coefficients globaux doivent être positifs ou nuls et finis.")
    ic = loss_ic(modele, groupes["IC"])
    bc = loss_bc(modele, groupes["BC"])
    res = loss_res(modele, groupes["RES"])
    return {"loss_ic": ic, "loss_bc": bc, "loss_res": res,
            "loss_total": w_ic * ic + w_bc * bc + w_res * res}


if __name__ == "__main__":
    from pinn import PINN
    from sampling import generer_points
    torch.manual_seed(0)
    modele = PINN()
    groupes = generer_points()
    pertes = calculer_pertes(modele, groupes)
    for nom, valeur in pertes.items():
        print(f"{nom} = {valeur.detach().item():.8f}, forme = {tuple(valeur.shape)}")
    pertes["loss_total"].backward()
    gradients_valides = all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                           for p in modele.parameters())
    print("Gradients des paramètres présents et finis :", gradients_valides)
    print("RES requires_grad :", groupes["RES"]["points"].requires_grad)
    print("Aucun entraînement : backward calcule les gradients sans modifier les paramètres.")
