"""Référence Fourier et conservation de l'aire du carré aligné."""
import unittest
import numpy as np
from finite_difference import resoudre_diffusion
from fourier_reference import temperature_fourier
from initial_condition import temperature_initiale


class TestReference(unittest.TestCase):
    def test_aire_et_temperatures(self):
        for n in (51, 101, 201):
            x, y, _, cartes = resoudre_diffusion(n_x=n, n_y=n, temps=[0])
            carte = cartes[0]
            aire = np.sum((carte-20)/60)*(x[1]-x[0])*(y[1]-y[0])
            self.assertAlmostEqual(float(aire), .04, places=12)
            i = int(np.argmin(abs(x-.4)))
            j = int(np.argmin(abs(y-.4)))
            milieu = n//2
            self.assertEqual(carte[milieu, milieu], 80)
            self.assertEqual(carte[milieu, i], 50)
            self.assertEqual(carte[j, i], 35)
        self.assertEqual(float(temperature_initiale(np.array([.4]), np.array([.5]))[0]), 80)

    def test_fourier_et_convergence_carre(self):
        erreurs = []
        for n in (51, 101, 201):
            x, y, _, cartes = resoudre_diffusion(n_x=n, n_y=n, temps=[900, 3600])
            erreur_grille = []
            for t, carte in zip((900, 3600), cartes):
                exacte = temperature_fourier(x, y, t, n_modes=100)
                plus = temperature_fourier(x, y, t, n_modes=200)
                np.testing.assert_allclose(exacte, plus, atol=1e-12, rtol=0)
                erreur_grille.append(np.linalg.norm(carte-exacte)/np.linalg.norm(exacte-20))
            erreurs.append(erreur_grille)
        erreurs = np.array(erreurs)
        self.assertTrue(np.all(erreurs[1] < erreurs[0]/2))
        self.assertTrue(np.all(erreurs[2] < erreurs[1]/2))
        self.assertTrue(np.all(erreurs[-1] < .001))  # <0.1 % à 900 et 3600 s.
        self.assertAlmostEqual(float(temperature_fourier([.5], [.5], 3600)[0,0]),
                               22.255682408, places=8)

    def test_non_aligne_refuse(self):
        with self.assertRaises(ValueError):
            resoudre_diffusion(temps=[0], centre_x=.503)


if __name__ == "__main__":
    unittest.main()
