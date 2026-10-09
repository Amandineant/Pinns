# Dimensions de la pièce en mètres
L_X = 1.0
L_Y = 1.0

# Températures en degrés Celsius
T_AMBIANTE = 20.0
T_OBJET = 80.0

# Diffusivité thermique de l'air en m²/s
ALPHA = 2e-5

# Durée totale étudiée : 1 heure, exprimée en secondes
T_MAX = 3600.0

# L'objet est un carré de 20 cm placé au centre de la pièce
OBJET_CENTRE_X = 0.5
OBJET_CENTRE_Y = 0.5
OBJET_TAILLE = 0.2

if __name__ == "__main__":
    print("Dimensions :", L_X, "m x", L_Y, "m")
    print("Températures :", T_AMBIANTE, "à", T_OBJET, "°C")
    print("Durée étudiée :", T_MAX, "secondes")
