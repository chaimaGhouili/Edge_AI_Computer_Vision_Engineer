# 🧠 Edge AI Surveillance System

Système de **reconnaissance faciale en temps réel** basé sur **Edge AI**, déployé sur **Raspberry Pi 4**.  
Le traitement est divisé en deux parties :  
➡️ **Google Colab** pour le prétraitement et la création des embeddings  
➡️ **Raspberry Pi** pour la reconnaissance locale et rapide en temps réel.

https://colab.research.google.com/drive/1KncoWIBLE3Z5hOwaJEZbqT7PI7PNaw9R?usp=sharing
---


### 🖥️ Étape 1 :   Google Colab — Prétraitement & Génération des embeddings

| Étape | Outil utilisé | Description |
|-------|----------------|-------------|
| 📸 Détection et alignement des visages | **MTCNN (facenet-pytorch)** | Localise et aligne automatiquement les visages depuis les images sources. |
| 🧠 Extraction des embeddings | **InceptionResNetV1 (FaceNet)** | Convertit chaque visage en un vecteur numérique 512D unique. |
| 💾 Sauvegarde du modèle | **NumPy (.npy)** | Stocke les embeddings de toutes les personnes dans un seul fichier (`all_embeddings.npy`). |

👉 **Objectif :** Générer des données optimisées et légères à transférer sur la Raspberry Pi.

---

### 🍓 Étape 2 : Raspberry Pi — Reconnaissance en temps réel

| Étape | Outil utilisé | Description |
|-------|----------------|-------------|
| 🎥 Capture vidéo | **OpenCV** | Capture le flux de  PiCam. |
| 👁️ Détection des visages | **OpenCV DNN ** | Détecte rapidement les visages dans chaque frame. |
| 🔍 Génération d’embedding local | **MobileNet-FaceNet** | Produit un vecteur d’embedding pour chaque visage détecté. |
| 🤝 Comparaison | **NumPy + Euclidean Distance** | Compare l’embedding capturé à ceux stockés dans `all_embeddings.npy`. |
| 🧩 Résultat | **Identification ou “Inconnu”** | Affiche le nom de la personne reconnue et la distance de similarité. |

---

## 🚀 Fonctionnalités principales

- Détection & reconnaissance **temps réel** directement sur Raspberry Pi 4  
- Pipeline **Edge AI complète** : entraînement sur PC, inférence sur appareil embarqué  
- Modèle léger et rapide pour une **basse consommation énergétique**  

---

## 🧰 Technologies utilisées

| Technologie | Rôle |
|--------------|------|
| **Python** | Langage principal |
| **OpenCV DNN** | Détection rapide des visages |
| **MTCNN** | Alignement des visages |
| **InceptionResNetV1 (FaceNet)** | Extraction des embeddings |
| **MobileNet-FaceNet** | Reconnaissance temps réel sur Raspberry Pi |
| **Torch / NumPy** | Calcul vectoriel |
| **Raspberry Pi 4** | Exécution Edge AI |
| **Google Colab** | Environnement de prétraitement et génération des embeddings |

---

## 📁 Structure du projet
```text
FaceRecognitionEdgeAI/
├── README.md                 # Description du projet
├── preprocessing.ipynb       # Notebook Colab pour générer les embeddings
├── main.py # Script de reconnaissance faciale sur Raspberry Pi
├── model/
│   └── all_embeddings.npy    # Embeddings pré-entraînés
```
## Auteur: **Chaima Ghouili**
```text
🎓 Étudiante en ingénierie informatique
💡 Passionnée par la vision par ordinateur et l’IA embarquée
contact chaimaghouili691@gmail.com