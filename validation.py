"""Métriques à utiliser sur des températures réduites, aux mêmes points."""
import numpy as np


def calculer_erreurs(prediction, reference):
    """MSE et erreur L2 relative sur u=(T-20)/60, pas directement sur T.

    Si la référence est nulle partout, l'erreur relative est indéfinie (NaN).
    Le résultat doit alors être interprété à l'aide de la MSE.
    """
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if prediction.shape != reference.shape or reference.size == 0:
        raise ValueError("Les cartes doivent être non vides et de même forme.")
    if not (np.all(np.isfinite(prediction)) and np.all(np.isfinite(reference))):
        raise ValueError("Les cartes doivent contenir uniquement des valeurs finies.")
    difference = prediction - reference
    norme_reference = np.linalg.norm(reference.ravel())
    return {"mse": float(np.mean(difference**2)),
            "l2_relative": (float(np.linalg.norm(difference.ravel()) / norme_reference)
                            if norme_reference > 0 else float("nan"))}
