"""Valeurs connues, scalaires et gradients des pertes, sans entraînement."""
import unittest
import torch
from losses import loss_ic, loss_bc, loss_res, calculer_pertes
from pinn import PINN
from sampling import generer_points
from nondimensionalization import BETA_X, BETA_Y


class TestLosses(unittest.TestCase):
    def test_scalaires_et_combinaison(self):
        modele = PINN()
        groupes = generer_points(n_ic=40, n_bc=40, n_res=40)
        pertes = calculer_pertes(modele, groupes)
        for valeur in pertes.values():
            self.assertEqual(valeur.ndim, 0)
            self.assertEqual(valeur.numel(), 1)
            self.assertTrue(bool(torch.isfinite(valeur)))
            self.assertGreaterEqual(float(valeur.detach()), 0)
        torch.testing.assert_close(pertes["loss_total"],
                                   pertes["loss_ic"]+pertes["loss_bc"]+pertes["loss_res"])
        p = calculer_pertes(modele, groupes, w_ic=2, w_bc=3, w_res=4)
        torch.testing.assert_close(p["loss_total"],
                                   2*p["loss_ic"]+3*p["loss_bc"]+4*p["loss_res"])

    def test_ic_ponderation_exacte(self):
        groupes = generer_points()
        ic = groupes["IC"]
        chaud = ic["valeurs"].bool()
        torch.testing.assert_close(ic["poids"][chaud], torch.full((250,), .16))
        torch.testing.assert_close(ic["poids"][~chaud], torch.full((750,), 1.28))
        # Prédiction nulle : erreur 1 dans le carré, 0 ailleurs => perte .04.
        zero = lambda p: p[:, 0:1] * 0
        self.assertAlmostEqual(float(loss_ic(zero, ic)), .04, places=6)
        # Prédiction 1 : erreur 0 dans le carré, 1 ailleurs => perte .96.
        un = lambda p: p[:, 0:1] * 0 + 1
        self.assertAlmostEqual(float(loss_ic(un, ic)), .96, places=6)

    def test_bc_valeur_connue(self):
        bc = generer_points(n_ic=20, n_bc=20, n_res=20)["BC"]
        constant = lambda p: p[:, 0:1] * 0 + 2
        self.assertEqual(float(loss_bc(constant, bc)), 4.)

    def test_res_valeur_connue_et_solution_exacte(self):
        res = generer_points(n_ic=20, n_bc=20, n_res=20)["RES"]
        def polynome(p):
            return p[:, 0:1]**2 + 3*p[:, 1:2]**2 + 5*p[:, 2:3]**2
        attendu = (10*res["points"][:, 2:3]-2*BETA_X-6*BETA_Y).square().mean()
        torch.testing.assert_close(loss_res(polynome, res), attendu)
        def solution(p):
            return (torch.sin(torch.pi*p[:, 0:1])*torch.sin(torch.pi*p[:, 1:2])
                    *torch.exp(-torch.pi**2*(BETA_X+BETA_Y)*p[:, 2:3]))
        # float64 pour une tolérance précise sur une solution analytique.
        points = res["points"].detach().double().requires_grad_(True)
        self.assertLess(float(loss_res(solution, {"points": points}).detach()), 1e-24)

    def test_backward_et_res_preserves(self):
        modele = PINN()
        groupes = generer_points(n_ic=40, n_bc=40, n_res=40)
        res = groupes["RES"]["points"]
        points_avant = res.detach().clone()
        parametres_avant = [p.detach().clone() for p in modele.parameters()]
        pertes = calculer_pertes(modele, groupes)
        self.assertIs(res, groupes["RES"]["points"])
        self.assertTrue(res.requires_grad)
        pertes["loss_total"].backward()
        for parametre, avant in zip(modele.parameters(), parametres_avant):
            self.assertIsNotNone(parametre.grad)
            self.assertTrue(bool(torch.isfinite(parametre.grad).all()))
            torch.testing.assert_close(parametre.detach(), avant, rtol=0, atol=0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in modele.parameters()), 0)
        self.assertTrue(res.requires_grad)
        self.assertIs(res, groupes["RES"]["points"])
        torch.testing.assert_close(res.detach(), points_avant, rtol=0, atol=0)

    def test_arguments_invalides(self):
        modele = PINN()
        groupes = generer_points(n_ic=20, n_bc=20, n_res=20)
        with self.assertRaises(ValueError):
            loss_res(modele, {"points": groupes["RES"]["points"].detach()})
        # Une cible (N,) risquerait un broadcasting en (N,N) : refus explicite.
        ic = dict(groupes["IC"])
        ic["valeurs"] = ic["valeurs"].squeeze(1)
        with self.assertRaises(ValueError):
            loss_ic(modele, ic)
        for poids in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calculer_pertes(modele, groupes, w_ic=poids)


if __name__ == "__main__":
    unittest.main()
