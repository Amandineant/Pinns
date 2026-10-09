"""Convention commune : X=x/L_X, Y=y/L_Y, tau=t/T_MAX.

u=(T-T_AMBIANTE)/(T_OBJET-T_AMBIANTE).
Équation : u_tau - BETA_X*u_XX - BETA_Y*u_YY = 0.
Pour la pièce carrée actuelle, BETA_X = BETA_Y = 0.072.
"""
from parameters import L_X, L_Y, T_AMBIANTE, T_OBJET, ALPHA, T_MAX


def temperature_reduite(temperature_celsius):
    return (temperature_celsius - T_AMBIANTE) / (T_OBJET - T_AMBIANTE)


def temperature_celsius(temperature_reduite):
    return T_AMBIANTE + temperature_reduite * (T_OBJET - T_AMBIANTE)


def temps_reduit(temps_secondes):
    return temps_secondes / T_MAX


def temps_secondes(temps_normalise):
    return temps_normalise * T_MAX


def coordonnees_reduites(x, y):
    return x / L_X, y / L_Y


BETA_X = ALPHA * T_MAX / L_X**2
BETA_Y = ALPHA * T_MAX / L_Y**2
BETA = BETA_X  # Compatibilité avec le premier code ; égalité pour une pièce carrée.


if __name__ == "__main__":
    print("20 °C devient :", temperature_reduite(20.0))
    print("80 °C devient :", temperature_reduite(80.0))
    print("0.5 redevient :", temperature_celsius(0.5), "°C")
    print("1800 secondes devient :", temps_reduit(1800.0))
    print("BETA_X, BETA_Y :", BETA_X, BETA_Y)
