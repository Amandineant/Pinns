from parameters import (
    L_X,
    T_AMBIANTE,
    T_OBJET,
    ALPHA,
    T_MAX,
)


# Transformation d'une température en valeur comprise entre 0 et 1
def temperature_reduite(temperature_celsius):
    return (
        (temperature_celsius - T_AMBIANTE)
        / (T_OBJET - T_AMBIANTE)
    )


# Transformation d'une température réduite en degrés Celsius
def temperature_celsius(temperature_reduite):
    return (
        T_AMBIANTE
        + temperature_reduite * (T_OBJET - T_AMBIANTE)
    )


# Transformation du temps en valeur comprise entre 0 et 1
def temps_reduit(temps_secondes):
    return temps_secondes / T_MAX


# Coefficient de diffusion dans l'équation adimensionnée
BETA = ALPHA * T_MAX / L_X**2


# Vérification des transformations
print("20 °C devient :", temperature_reduite(20.0))
print("80 °C devient :", temperature_reduite(80.0))
print("0.5 redevient :", temperature_celsius(0.5), "°C")
print("1800 secondes devient :", temps_reduit(1800.0))
print("Coefficient BETA :", BETA)