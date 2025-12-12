Compte Rendu Complet : Détection de Joueurs de Football & Analyse par Régression
📌 1. Introduction

Ce projet combine deux approches de Data Science :

YOLOv8 → pour détecter automatiquement les joueurs, le ballon et les objets dans les images/vidéos de football.

La régression → pour analyser et prédire des relations entre les données extraites par YOLO (vitesse, position, distance au ballon, etc.).

L'objectif est d’utiliser la vision par ordinateur pour extraire des informations, puis employer des modèles statistiques pour les analyser.

📦 2. Importation des Bibliothèques
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np


Ces bibliothèques permettent :

d’utiliser YOLOv8,

de lire des images/vidéos,

de visualiser les résultats,

et de manipuler les données.

🧠 3. Chargement du Modèle YOLOv8
model = YOLO("yolov8n.pt")


yolov8n.pt : version légère du modèle YOLOv8.

Optimisée pour la détection en temps réel.

📂 4. Chargement des Données (data.yaml)
yaml_path = "/kaggle/input/data-updated/data.yaml"
model = YOLO("yolov8n.pt")


Ce fichier contient :

les chemins des images d'entraînement,

les annotations des objets,

les noms des classes (player, ball, referee…).

🏋️‍♂️ 5. Entraînement du Modèle
model.train(data=yaml_path, epochs=50, imgsz=640)


YOLOv8 ajuste ses poids pour détecter correctement :

les joueurs,

le ballon,

les zones du terrain.

📊 6. Évaluation du Modèle
metrics = model.val()
print(metrics)


L’évaluation fournit :

précision,

rappel (recall),

mAP (mean Average Precision).

🔍 7. Détection sur Image
results = model("image.jpg")
results[0].show()


Affiche :

les boîtes de détection,

les classes détectées,

les scores de confiance.

🎥 8. Détection sur Vidéo
model.predict(source="video.mp4", show=True)


YOLO détecte les objets image par image pour une analyse en temps réel.

🔢 9. Pourquoi utiliser la régression ?

YOLO détecte les objets…
👉 mais il n’explique pas pourquoi certaines variables changent.

La régression sert à :

analyser les relations entre variables,

comprendre les comportements des joueurs,

prédire des valeurs futures (distance, position, vitesse…).

Elle donne du sens aux données produites par YOLO.

📘 10. Analyse de Régression
🔹 10.1 Régression Linéaire

Utilisée pour prédire une variable continue, par exemple :

distance entre un joueur et le ballon,

vitesse en fonction de la position,

déplacement dans une direction.

Exemple de code :
from sklearn.linear_model import LinearRegression

X = np.array(df["player_speed"]).reshape(-1,1)
y = df["distance_to_ball"]

model_reg = LinearRegression()
model_reg.fit(X, y)

print(model_reg.coef_, model_reg.intercept_)

Interprétation :

coef_ = impact de la vitesse sur la distance,

un coefficient négatif = plus le joueur court vite, plus il se rapproche du ballon.

🔹 10.2 Régression Polynomiale

Utilisée si la relation n’est pas linéaire, par exemple une courbe.

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression().fit(X_poly, y)

🔹 10.3 Visualisation
plt.scatter(X, y)
plt.plot(X, model_reg.predict(X), linewidth=3)
plt.xlabel("Vitesse du joueur")
plt.ylabel("Distance au ballon")
plt.title("Régression linéaire")
plt.show()

🎯 11. Comment YOLOv8 et la Régression travaillent ensemble
YOLOv8	Régression
Détecte les objets	Analyse les relations
Donne des positions, distances, vitesses	Explique pourquoi ces valeurs changent
Produit des données	Prédit les valeurs futures
Vision	Intelligence
✔️ 12. Conclusion

Ce projet montre comment combiner :

YOLOv8 pour détecter les joueurs et objets dans des images de football,

la régression pour analyser et prédire les comportements des joueurs.

Ainsi, on obtient un système complet capable :

d’observer,

d’analyser,

et de comprendre les actions sur le terrain.

C’est une approche puissante pour :

l’analyse tactique,

les statistiques sportives,

les systèmes d’aide à l’arbitrage.
