"""Architecture et dérivées vérifiées avant les pertes et l'entraînement."""
import unittest
import torch
from torch import nn
from pinn import PINN, derivees_temperature, residu_physique
from nondimensionalization import BETA_X, BETA_Y
from sampling import generer_points


class TestPINN(unittest.TestCase):
    def test_architecture_et_sorties(self):
        modele = PINN()
        couches = [c for c in modele.reseau if isinstance(c, nn.Linear)]
        self.assertEqual([(c.in_features, c.out_features) for c in couches],
                         [(3, 32), (32, 32), (32, 32), (32, 1)])
        self.assertEqual(sum(isinstance(c, nn.Tanh) for c in modele.reseau), 3)
        for groupe in generer_points(n_ic=40, n_bc=40, n_res=40).values():
            sortie = modele(groupe["points"])
            self.assertEqual(tuple(sortie.shape), (40, 1))
            self.assertTrue(bool(torch.isfinite(sortie).all()))
        with self.assertRaises(ValueError):
            modele(torch.zeros(5, 2))

    def test_derivees_exactes(self):
        # Fonction connue : u=X²+3Y²+5tau² ; permet de vérifier les axes.
        def fonction(p):
            return p[:, 0:1]**2 + 3*p[:, 1:2]**2 + 5*p[:, 2:3]**2
        points = torch.tensor([[.2, .4, .6], [.7, .3, .1]],
                              dtype=torch.float64, requires_grad=True)
        d = derivees_temperature(fonction, points)
        torch.testing.assert_close(d["u_tau"], 10*points[:, 2:3], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(d["u_xx"], torch.full((2, 1), 2., dtype=torch.float64))
        torch.testing.assert_close(d["u_yy"], torch.full((2, 1), 6., dtype=torch.float64))
        attendu = 10*points[:, 2:3] - 2*BETA_X - 6*BETA_Y
        torch.testing.assert_close(residu_physique(fonction, points), attendu)

    def test_solution_exacte_equation_normalisee(self):
        def solution(p):
            return (torch.sin(torch.pi*p[:, 0:1]) * torch.sin(torch.pi*p[:, 1:2])
                    * torch.exp(-torch.pi**2*(BETA_X+BETA_Y)*p[:, 2:3]))
        points = torch.tensor([[.2, .4, .6], [.7, .3, .1]],
                              dtype=torch.float64, requires_grad=True)
        residu = residu_physique(solution, points)
        self.assertLess(float(residu.detach().abs().max()), 1e-12)

    def test_derivees_reseau_et_gradients_parametres(self):
        modele = PINN()
        points = generer_points(n_ic=20, n_bc=20, n_res=20)["RES"]["points"]
        for derivee in derivees_temperature(modele, points).values():
            self.assertEqual(tuple(derivee.shape), (20, 1))
            self.assertTrue(bool(torch.isfinite(derivee).all()))
        # Vérifie la propagation jusqu'aux paramètres ; ce n'est pas un entraînement.
        (residu_physique(modele, points).square().mean()
         + modele(points).square().mean()).backward()
        for parametre in modele.parameters():
            self.assertIsNotNone(parametre.grad)
            self.assertTrue(bool(torch.isfinite(parametre.grad).all()))
        with self.assertRaises(ValueError):
            derivees_temperature(modele, points.detach())


if __name__ == "__main__":
    unittest.main()
