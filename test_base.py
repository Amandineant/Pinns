"""Vérifications de la base physique avant l'implémentation du PINN."""
import contextlib
import importlib
import io
import unittest
import numpy as np
import torch
from finite_difference import resoudre_diffusion
from initial_condition import masque_objet
from nondimensionalization import (temperature_reduite, temperature_celsius,
                                  temps_reduit, temps_secondes, BETA_X, BETA_Y)
from sampling import generer_points
from validation import calculer_erreurs


class TestBase(unittest.TestCase):
    def test_imports_sans_execution(self):
        sortie = io.StringIO()
        with contextlib.redirect_stdout(sortie):
            for nom in ("parameters", "initial_condition", "nondimensionalization",
                        "finite_difference", "sampling", "validation", "pinn", "losses"):
                importlib.reload(importlib.import_module(nom))
        self.assertEqual(sortie.getvalue(), "")

    def test_normalisation(self):
        valeurs = np.array([20., 50., 80.])
        np.testing.assert_allclose(temperature_celsius(temperature_reduite(valeurs)), valeurs)
        np.testing.assert_allclose(temperature_reduite(valeurs), [0, .5, 1])
        self.assertEqual(temps_secondes(temps_reduit(1800)), 1800)
        self.assertAlmostEqual(BETA_X, .072)
        self.assertAlmostEqual(BETA_Y, .072)

    def test_solveur_scenario_original(self):
        x, y, temps, cartes = resoudre_diffusion(dt=4, temps=[0, 1001, 3600])
        np.testing.assert_array_equal(temps, [0, 1001, 3600])
        self.assertEqual(cartes.shape, (3, 51, 51))
        self.assertTrue(np.all(cartes >= 20 - 1e-10))
        self.assertTrue(np.all(cartes <= 80 + 1e-10))
        for murs in (cartes[:, 0, :], cartes[:, -1, :],
                     cartes[:, :, 0], cartes[:, :, -1]):
            np.testing.assert_allclose(murs, 20)
        from fourier_reference import temperature_fourier
        exacte = temperature_fourier(x, y, 3600)
        erreur = np.linalg.norm(cartes[-1]-exacte) / np.linalg.norm(exacte-20)
        self.assertLess(erreur, .002)  # <0.2 % sur le champ à une heure.
        self.assertTrue(np.all(np.diff(np.sum(cartes - 20, axis=(1, 2))) <= 0))
        self.assertFalse(np.shares_memory(cartes[0], cartes[-1]))
        _, _, temps_defaut, cartes_defaut = resoudre_diffusion()
        self.assertEqual(temps_defaut[0], 0)
        self.assertEqual(temps_defaut[-1], 3600)
        np.testing.assert_allclose(cartes_defaut[0], cartes[0], atol=1e-12)
        np.testing.assert_allclose(cartes_defaut[-1], cartes[-1], atol=1e-5)

    def test_stabilite_et_arguments(self):
        with self.assertRaises(ValueError):
            resoudre_diffusion(n_x=101, n_y=101, dt=4, temps=[0])
        for temps in ([1, 0], [0, 0], [-1], [float("nan")], []):
            with self.assertRaises(ValueError):
                resoudre_diffusion(temps=temps)
        with self.assertRaises(ValueError):
            resoudre_diffusion(temps=[0], centre_x=.05)
        _, _, _, cartes = resoudre_diffusion(n_x=101, n_y=101, temps=[0, 10])
        self.assertTrue(np.all(np.isfinite(cartes)))

    def test_solution_analytique_et_raffinement(self):
        # Mode sinus : solution exacte sur un rectangle, murs à 20 °C.
        longueur, largeur, alpha, instant = 1., .8, 2e-5, 1000.
        amplitude = np.exp(-alpha * np.pi**2 * (1 / longueur**2 + 1 / largeur**2) * instant)
        erreurs = []
        for n in (21, 41, 81):
            def initiale(X, Y):
                return 20 + 60 * np.sin(np.pi * X / longueur) * np.sin(np.pi * Y / largeur)
            x, y, _, cartes = resoudre_diffusion(
                n_x=n, n_y=n, longueur=longueur, largeur=largeur,
                temps=[instant], condition_initiale=initiale)
            X, Y = np.meshgrid(x, y)
            exacte = 20 + amplitude * (initiale(X, Y) - 20)
            erreurs.append(np.linalg.norm(cartes[0] - exacte) / np.linalg.norm(exacte - 20))
        self.assertLess(erreurs[1], erreurs[0] / 2)
        self.assertLess(erreurs[2], erreurs[1] / 2)
        # Tolérance explicite : erreur relative < 0.1 % sur la grille 81x81.
        self.assertLess(erreurs[-1], .001)

    def test_echantillonnage(self):
        avant = torch.random.get_rng_state().clone()
        groupes = generer_points(n_ic=1000, n_bc=400, n_res=500)
        self.assertTrue(torch.equal(avant, torch.random.get_rng_state()))
        autres = generer_points(n_ic=1000, n_bc=400, n_res=500)
        for nom, n in (("IC", 1000), ("BC", 400), ("RES", 500)):
            points = groupes[nom]["points"]
            self.assertEqual(tuple(points.shape), (n, 3))
            self.assertTrue(torch.equal(points, autres[nom]["points"]))
            self.assertTrue(bool(torch.all((points >= 0) & (points <= 1))))
        ic = groupes["IC"]
        self.assertTrue(bool(torch.all(ic["points"][:, 2] == 0)))
        attendu = masque_objet(ic["points"][:, 0:1], ic["points"][:, 1:2])
        self.assertTrue(torch.equal(ic["valeurs"].bool(), attendu))
        self.assertEqual(int(ic["valeurs"].sum()), 250)
        self.assertAlmostEqual(float(ic["poids"].mean()), 1, places=6)
        self.assertAlmostEqual(float((ic["poids"] * ic["valeurs"]).mean()), .04, places=6)
        bc = groupes["BC"]["points"]
        for coordonnee, valeur in ((0, 0), (0, 1), (1, 0), (1, 1)):
            self.assertEqual(int((bc[:, coordonnee] == valeur).sum()), 100)
        self.assertTrue(bool(torch.all(groupes["BC"]["valeurs"] == 0)))
        res = groupes["RES"]["points"]
        self.assertTrue(bool(torch.all((res > 0) & (res < 1))))
        # Vérifie que les dérivées secondes sont effectivement possibles.
        u = res[:, 0:1]**2 + res[:, 1:2]**2 + res[:, 2:3]
        derivees = torch.autograd.grad(u.sum(), res, create_graph=True)[0]
        u_xx = torch.autograd.grad(derivees[:, 0].sum(), res)[0][:, 0]
        self.assertTrue(torch.allclose(u_xx, torch.full_like(u_xx, 2)))
        with self.assertRaises(ValueError):
            generer_points(n_bc=7)

    def test_objet_parametrable(self):
        groupes = generer_points(centre_x=.3, centre_y=.7, taille=.1)
        ic = groupes["IC"]
        attendu = masque_objet(ic["points"][:, 0:1], ic["points"][:, 1:2],
                              centre_x=.3, centre_y=.7, taille=.1)
        self.assertTrue(torch.equal(ic["valeurs"].bool(), attendu))
        self.assertEqual(int(ic["valeurs"].sum()), 250)
        self.assertAlmostEqual(float((ic["poids"] * ic["valeurs"]).mean()), .01, places=6)
        _, _, _, cartes = resoudre_diffusion(n_x=101, n_y=101, temps=[0], centre_x=.3, centre_y=.7, taille=.1)
        self.assertEqual(float(cartes[0].max()), 80)

    def test_metriques(self):
        erreurs = calculer_erreurs([2, 4], [1, 2])
        self.assertEqual(erreurs["mse"], 2.5)
        self.assertEqual(erreurs["l2_relative"], 1)
        self.assertTrue(np.isnan(calculer_erreurs([0], [0])["l2_relative"]))
        with self.assertRaises(ValueError):
            calculer_erreurs([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
