"""Référence par différences finies explicites, avec stabilité contrôlée."""
import numpy as np
from parameters import L_X, L_Y, ALPHA, T_MAX, T_AMBIANTE, T_OBJET
from initial_condition import creer_grille, temperature_initiale_discrete


def resoudre_diffusion(n_x=51, n_y=51, dt=None, temps=None,
                      longueur=L_X, largeur=L_Y, alpha=ALPHA,
                      temperature_ambiante=T_AMBIANTE,
                      temperature_objet=T_OBJET, condition_initiale=None,
                      **geometrie):
    """Retourne x, y, les temps (secondes) et les cartes (°C).

    Les cartes ont la forme (nombre de temps, n_y, n_x).
    dt=None choisit 80 % de la limite de stabilité (4 s sur la grille 51x51).
    Un dt explicite instable est refusé. Le dernier pas est raccourci pour
    atteindre exactement chaque instant demandé, sans interpolation temporelle.
    condition_initiale(X,Y) permet notamment une validation analytique.
    """
    x, y, X, Y = creer_grille(n_x, n_y, longueur, largeur)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("La diffusivité doit être positive et finie.")
    if not (np.isfinite(temperature_ambiante) and np.isfinite(temperature_objet)):
        raise ValueError("Les températures doivent être finies.")
    dx, dy = longueur / (n_x - 1), largeur / (n_y - 1)
    dt_max = 1 / (2 * alpha * (1 / dx**2 + 1 / dy**2))
    if dt is None:
        dt = 0.8 * dt_max
    if not np.isfinite(dt) or dt <= 0 or dt > dt_max * (1 + 1e-12):
        raise ValueError(f"Pas de temps instable ou invalide : dt <= {dt_max:.6g} s requis.")
    temps = np.asarray([0, 900, 1800, T_MAX] if temps is None else temps,
                       dtype=float)
    if (temps.ndim != 1 or temps.size == 0 or not np.all(np.isfinite(temps))
            or np.any(temps < 0) or np.any(np.diff(temps) <= 0)):
        raise ValueError("Les temps doivent être finis, positifs ou nuls et strictement croissants.")
    if condition_initiale is None:
        temperature = temperature_initiale_discrete(
            X, Y, temperature_ambiante, temperature_objet,
            longueur=longueur, largeur=largeur, **geometrie)
    else:
        temperature = condition_initiale(X, Y)
    temperature = np.array(temperature, dtype=float, copy=True)
    if temperature.shape != X.shape or not np.all(np.isfinite(temperature)):
        raise ValueError("La condition initiale doit être une carte finie de la taille de la grille.")
    # Dirichlet : imposés à t=0 puis conservés à chaque pas.
    temperature[[0, -1], :] = temperature_ambiante
    temperature[:, [0, -1]] = temperature_ambiante
    cartes = []
    instant = 0.0
    for cible in temps:
        while instant < cible:
            pas = min(dt, cible - instant)
            nouvelle = temperature.copy()
            derivee_x = (temperature[1:-1, 2:] - 2 * temperature[1:-1, 1:-1]
                         + temperature[1:-1, :-2]) / dx**2
            derivee_y = (temperature[2:, 1:-1] - 2 * temperature[1:-1, 1:-1]
                         + temperature[:-2, 1:-1]) / dy**2
            nouvelle[1:-1, 1:-1] += alpha * pas * (derivee_x + derivee_y)
            temperature = nouvelle
            instant = cible if pas == cible - instant else instant + pas
        cartes.append(temperature.copy())
    return x, y, temps.copy(), np.stack(cartes)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y, temps, cartes = resoudre_diffusion()
    print("Instants sauvegardés (s) :", temps)
    print("Températures finales (°C) :", cartes[-1].min(), cartes[-1].max())
    figure, axes = plt.subplots(1, len(temps), figsize=(14, 4),
                                layout="constrained")
    for axe, instant, carte in zip(axes, temps, cartes):
        image = axe.imshow(carte, origin="lower", extent=[0, L_X, 0, L_Y],
                           cmap="coolwarm", vmin=T_AMBIANTE, vmax=T_OBJET)
        axe.set_title(f"t = {instant:g} s")
        axe.set_xlabel("x (m)")
        axe.set_ylabel("y (m)")
    figure.colorbar(image, ax=axes, label="Température (°C)")
    plt.show()
