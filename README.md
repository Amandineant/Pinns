# PINN : diffusion thermique 2D

Projet M2 : comparer un réseau informé par la physique (PINN) à un solveur
par différences finies, puis présenter les résultats dans Gradio.

## État actuel

La base physique, les points d'entraînement et le réseau de base sont disponibles.
Les dérivées, le résidu normalisé et les trois pertes sont testés. Adam puis
L-BFGS, la technique avancée et Gradio restent à réaliser.
Un seul scénario d'objet est visé à ce stade ; aucun PINN n'est encore entraîné.

## Installation

Depuis le dossier du dépôt, avec un Python compatible avec les versions fixées
(développement initial sous Python 3.13) :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ne pas partager `.venv` : chacun recrée son environnement.

## Convention physique

Pièce de 1 m x 1 m, murs à 20 °C, carré central de côté 0,20 m initialement
à 80 °C. Le carré refroidit ensuite : il ne constitue pas une source permanente.
Diffusivité : 2e-5 m²/s ; durée étudiée : 3600 s.

Entrées futures du PINN : `X=x/L_X`, `Y=y/L_Y`, `tau=t/T_MAX`.
Sortie : `u=(T-T_AMBIANTE)/(T_OBJET-T_AMBIANTE)`.

L'équation réduite est :

```text
u_tau - BETA_X * u_XX - BETA_Y * u_YY = 0
BETA_X = ALPHA * T_MAX / L_X**2
BETA_Y = ALPHA * T_MAX / L_Y**2
```

Les deux coefficients valent 0.072 pour la pièce actuelle. Les points IC ont
`tau=0` et `u=1` dans l'objet, `u=0` ailleurs. Les points BC ont `u=0`.
Les points RES sont strictement intérieurs et permettent les dérivées autograd.
Les graphiques du solveur utilisent des mètres, des secondes et des °C.

## Fichiers et commandes

- `parameters.py` : constantes du scénario.
- `initial_condition.py` : masque commun de l'objet et température initiale.
- `nondimensionalization.py` : transformations et coefficients physiques.
- `finite_difference.py` : solveur vectorisé et cartes à plusieurs instants.
- `sampling.py` : points IC, BC et RES reproductibles (générateur local).
- `validation.py` : MSE et erreur L2 relative, à appliquer aux températures réduites.
- `pinn.py` : réseau 3-32-32-32-1 avec tanh, dérivées et résidu normalisé.
- `test_base.py` : stabilité, solution analytique, raffinement et échantillonnage.
- `test_pinn.py` : sorties, dérivées exactes, résidu et gradients des paramètres.
- `losses.py` : pertes IC/BC/RES et total, coefficients globaux à 1 par défaut.
- `test_losses.py` : pertes scalaires, formules exactes et propagation des gradients.

```bash
python initial_condition.py
python nondimensionalization.py
python finite_difference.py
python sampling.py
python pinn.py
python losses.py
python -m unittest -v test_base test_pinn test_losses
```

Importer ces modules ne lance ni simulation ni graphique.

## Réutiliser le solveur

```python
from finite_difference import resoudre_diffusion
x, y, temps, cartes = resoudre_diffusion(temps=[0, 600, 1800, 3600])
# cartes[k] correspond exactement à temps[k] ; forme (n_y, n_x).
```

Le schéma explicite exige :
`ALPHA * dt * (1/dx**2 + 1/dy**2) <= 0.5`.
Sans `dt`, le solveur prend 80 % du maximum autorisé, soit 4 s sur la grille
51 x 51. Un pas fourni instable est refusé. Les pas sont raccourcis pour
atteindre exactement les instants demandés. Le carré doit rester strictement
à l'intérieur de la pièce pour respecter les murs froids.

La validation analytique utilise un profil sinusoidal lisse, dont la solution
exacte est connue, sur plusieurs grilles. La convergence pour le carré discontinu aligné est également testée contre
la référence Fourier (voir la correction documentée ci-dessous).

## Réutiliser les points

```python
from sampling import generer_points
points = generer_points(n_ic=1000, n_bc=1000, n_res=5000, seed=0)
xyz_ic = points["IC"]["points"]  # (N_IC, 3), colonnes X, Y, tau
u_ic = points["IC"]["valeurs"]   # (N_IC, 1)
xyz_bc = points["BC"]["points"]
u_bc = points["BC"]["valeurs"]
xyz_res = points["RES"]["points"]  # requires_grad=True
```

`n_bc` doit être multiple de quatre pour équilibrer les murs. IC est stratifié :
par défaut 250 points dans l'objet et 750 hors de l'objet. Le masque commun
accepte `centre_x`, `centre_y`, `taille` : ces paramètres sont transmis au tirage.
Cette configuration reste un scénario fixe pour chaque modèle, sans apprentissage
sur plusieurs objets à ce stade.

Chaque point IC possède un poids d'importance pour représenter l'aire réelle,
malgré le suréchantillonnage de l'objet :

```python
poids_ic = points["IC"]["poids"]
# À utiliser lors de l'implémentation des pertes :
# loss_ic = (poids_ic * (prediction_ic - u_ic).square()).mean()
```

Les poids valent 0.16 dans le carré actuel (4 % de l'aire) et 1.28 ailleurs.
Cette correction ne constitue pas la pondération dynamique des trois pertes
et ne remplace pas l'expérience avancée imposée par le sujet.

## Réseau de base : étape actuelle

`PINN()` reçoit `(N,3)` contenant X,Y,tau et retourne `(N,1)` contenant u.
Trois couches cachées de 32 neurones utilisent tanh ; la sortie est linéaire.
Le réseau possède 2273 paramètres entraînables. `derivees_temperature` conserve
le graphe autograd pour permettre la future optimisation des paramètres.
`residu_physique` utilise BETA_X et BETA_Y, tous deux égaux à 0.072 ici.
Les pertes sont disponibles dans `losses.py`, sans boucle d'optimisation.
Les prédictions affichées par `python pinn.py` ne sont pas physiquement valides
avant apprentissage.

Les tests contrôlent les dérivées sur un polynôme et un résidu quasi nul sur
une solution exacte normalisée (tolérance absolue 1e-12 en float64). Pour le
solveur, l'erreur L2 relative analytique doit être inférieure à 0.001, soit
0.1 %, sur la grille 81x81. Les cartes à 0 et 3600 s sont vérifiées explicitement.

## Validation du futur PINN

Comparer aux mêmes coordonnées et instants, en convertissant les cartes de
référence en température réduite. Calculer les métriques par instant et afficher
les erreurs, notamment au début. Une référence nulle donne une L2 relative
indéfinie (`NaN`) : utiliser alors la MSE. Une erreur relative peut aussi être
sensible lorsque la référence devient très proche de zéro.

## Suite du projet

1. Examiner les trois pertes sur le modèle non entraîné avant de choisir
   d'éventuels coefficients globaux différents de 1.
2. Entraîner avec Adam puis L-BFGS et sauvegarder modèle/configuration.
3. Comparer aux différences finies, puis tester une technique avancée.
4. Décider comment traiter les paramètres variables de l'objet, requis dans
   l'interface finale : modèle paramétré entraîné sur plusieurs configurations,
   ou autre stratégie validée avec l'enseignant.
5. Construire Gradio, documenter les expériences et préparer l'oral de 5 minutes.

Chaque membre travaille sur sa branche. Les changements sont relus dans une
Pull Request avant intégration dans `main`.

## Formule exacte des poids IC

Soit `a = taille**2 / (L_X * L_Y)`, fraction de l'aire occupée par l'objet.
Soit `q = n_objet / N_IC`, fraction effective de points tirés dans l'objet
(après l'arrondi du nombre de points, pas seulement la fraction demandée).
Les deux régions sont échantillonnées uniformément, chacune séparément.

```text
poids_objet = a / q
poids_exterieur = (1 - a) / (1 - q)
loss_ic = mean(poids * (u_pred - u_cible)**2)
        = a * mean(erreur² dans l'objet)
          + (1 - a) * mean(erreur² à l'extérieur)
```

Dans le scénario actuel : a=0.04, q=250/1000=0.25, donc poids_objet=0.16 et
poids_exterieur=1.28. Sans correction, une moyenne simple attribuerait 25 %
de l'objectif IC à une région qui occupe seulement 4 % de la pièce. Les poids
conservent l'objectif d'une erreur moyenne uniforme sur la surface, tout en
assurant davantage de points dans la zone chaude. Ils ne renforcent donc pas
à eux seuls l'importance physique de l'objet dans l'objectif. Si cette région
reste mal apprise, une autre stratégie devra être évaluée et documentée.

Ces poids locaux sont distincts de `w_ic`, `w_bc`, `w_res`, qui équilibrent
les trois objectifs globaux. Ils ne sont pas une pondération dynamique.

## Pertes : étape validée, sans entraînement

```text
loss_bc = mean((u_pred_murs - 0)**2)
loss_res = mean((u_tau - BETA_X*u_XX - BETA_Y*u_YY)**2)
loss_total = w_ic*loss_ic + w_bc*loss_bc + w_res*loss_res
```

```python
from pinn import PINN
from sampling import generer_points
from losses import calculer_pertes
modele = PINN()
groupes = generer_points()
pertes = calculer_pertes(modele, groupes)  # w_ic=w_bc=w_res=1
pertes["loss_total"].backward()  # Gradients uniquement, aucun optimizer.step().
```

Chaque perte est un tenseur scalaire conservant son graphe autograd. Seuls
les affichages utilisent `detach()`. Les points RES ne sont ni remplacés ni
détachés pendant le calcul. `backward()` ne met pas à jour les paramètres :
l'entraînement reste à implémenter après validation de l'étape suivante.

Le modèle reste conditionné uniquement par X,Y,tau pour un scénario fixe.
Les paramètres géométriques de l'échantillonnage ne rendent pas le PINN
paramétrique : un même modèle couvrant plusieurs objets nécessitera des
entrées supplémentaires et un entraînement sur plusieurs configurations.

## Correction du carré sur la grille

Le solveur utilise temperature_initiale_discrete : 80 °C à l'intérieur,
50 °C sur le contour et 35 °C aux coins. La formule est
T=T_AMBIANTE+fraction*(T_OBJET-T_AMBIANTE). Les fractions 1/2 et 1/4
sont valides pour les bords alignés sur les nœuds de la grille. Les géométries
non alignées sont actuellement refusées ; il faudra calculer la couverture
réelle des cellules pour les déplacements/redimensionnements libres.
Le PINN conserve sa condition ponctuelle u=1 dans l'objet, u=0 ailleurs.

Le carré représente maintenant 0.04 m², contre 0.0484 m² auparavant sur 51x51.
Le maximum à une heure vaut environ 22.25258 °C sur 51x51 ; la référence
indépendante Fourier donne 22.25568 °C. fourier_reference.py intègre
analytiquement le carré dans une base sinus adaptée aux murs Dirichlet.
Le test compare 100 et 200 modes et vérifie la convergence sur 51,101,201
points à 900 et 3600 s ; erreur relative <0.1 % sur la grille la plus fine.
L'ancien maximum 22.70898 n'est plus une cible de validation.

```bash
python -m unittest -v test_base test_pinn test_losses test_reference
```

## Premier diagnostic Adam (300 étapes)

Commande : `python diagnostic_adam.py`. Graine modèle/entraînement 0,
graine des points de contrôle indépendants 123, Adam lr=0.001, coefficients
IC/BC/RES tous à 1, 1000 IC, 1000 BC et 2000 RES fixes. Historique,
configuration et contrôles sont enregistrés dans experiments/diagnostic_adam/
resultats.json ; modèle local dans modele.pt (ignoré par Git).

Sur les points de contrôle : perte totale 0.048458 -> 0.036698, IC
0.038924 -> 0.036296, BC 0.004759 -> 0.000339, RES 0.004776 -> 0.000064.
Malgré cette baisse, la température moyenne prédite dans l'objet à t=0 n'est
que 24.55 °C (u=0.07576), contre 80 °C attendus. Le RMSE chaud vaut encore
0.92424 en température réduite. C'est un diagnostic court, pas un résultat
scientifiquement satisfaisant ni l'entraînement final. Les prochaines décisions
(durée d'Adam, L-BFGS, équilibrage) devront être validées et comparées.
