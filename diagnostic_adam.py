"""Court diagnostic Adam reproductible, pas un entraînement final."""
import argparse
import json
from pathlib import Path
import torch
from pinn import PINN
from sampling import generer_points
from losses import calculer_pertes


def diagnostiquer(etapes=300, sortie=Path("experiments/diagnostic_adam")):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    modele = PINN()
    groupes = generer_points(n_ic=1000, n_bc=1000, n_res=2000, seed=0)
    controle = generer_points(n_ic=2000, n_bc=1000, n_res=2000, seed=123)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)
    historique = []
    def mesurer():
        p = calculer_pertes(modele, controle)
        resultats = {nom: float(v.detach()) for nom, v in p.items()}
        with torch.no_grad():
            ic = controle["IC"]
            pred = modele(ic["points"])
            chaud = ic["valeurs"].bool()
            resultats["u_moyen_objet_t0"] = float(pred[chaud].mean())
            resultats["rmse_objet_t0"] = float((pred[chaud]-1).square().mean().sqrt())
            resultats["rmse_exterieur_t0"] = float(pred[~chaud].square().mean().sqrt())
            resultats["erreur_max_murs_u"] = float(modele(controle["BC"]["points"]).abs().max())
        return resultats
    avant = mesurer()
    for etape in range(etapes):
        optimiseur.zero_grad(set_to_none=True)
        groupes["RES"]["points"].grad = None
        pertes = calculer_pertes(modele, groupes)
        if not all(bool(torch.isfinite(v)) for v in pertes.values()):
            raise RuntimeError("Perte non finie : arrêt du diagnostic.")
        pertes["loss_total"].backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in modele.parameters()):
            raise RuntimeError("Gradient invalide : arrêt du diagnostic.")
        optimiseur.step()
        if etape == 0 or (etape+1) % 50 == 0:
            ligne = {"etape": etape+1, **{nom: float(v.detach()) for nom,v in pertes.items()}}
            historique.append(ligne)
            print(ligne, flush=True)
    apres = mesurer()
    rapport = {"configuration": {"etapes": etapes, "lr": 1e-3, "seed": 0,
               "seed_controle": 123, "n_ic": 1000, "n_bc": 1000, "n_res": 2000,
               "w_ic": 1, "w_bc": 1, "w_res": 1, "scenario": "carre central fixe"},
               "controle_avant": avant, "controle_apres": apres, "historique": historique}
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie/"resultats.json").write_text(json.dumps(rapport, indent=2)+"\n")
    torch.save({"state_dict": modele.state_dict(), "configuration": rapport["configuration"]},
               sortie/"modele.pt")
    print(json.dumps({"avant": avant, "apres": apres}, indent=2))
    return rapport


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--etapes", type=int, default=300)
    args = parseur.parse_args()
    if args.etapes <= 0:
        parseur.error("--etapes doit être positif")
    diagnostiquer(args.etapes)
