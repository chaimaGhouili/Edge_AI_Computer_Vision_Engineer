# ---------------- test_recognition_final.py ----------------
import numpy as np
import cv2
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from numpy.linalg import norm

# ---------------- 1️⃣ Charger les embeddings ----------------
EMBEDDINGS_PATH = "/home/pi/all_embeddings.npy"  # Modifier selon ton système

try:
    embeddings_dict = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()
    print("✅ Embeddings chargés :", list(embeddings_dict.keys()))
except FileNotFoundError:
    print("❌ Fichier non trouvé :", EMBEDDINGS_PATH)
    exit()

# ---------------- 2️⃣ Initialiser MTCNN et FaceNet ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("💻 Device utilisé :", device)

mtcnn = MTCNN(image_size=160, margin=10, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---------------- 3️⃣ Fonction de reconnaissance ----------------
def recognize(face_embedding, embeddings_dict, threshold=0.7):
    min_dist = float('inf')
    identity = "Inconnu"

    for person, person_embeddings in embeddings_dict.items():
        for e in person_embeddings:
            dist = norm(face_embedding - e)
            if dist < min_dist:
                min_dist = dist
                identity = person

    if min_dist > threshold:
        identity = "Inconnu"

    return identity, min_dist

# ---------------- 4️⃣ Capture caméra ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Impossible d'ouvrir la caméra.")
    exit()

print("📸 Appuie sur ESPACE pour capturer, ESC pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Impossible de lire le flux caméra.")
        break

    cv2.imshow("Camera - ESPACE pour capturer", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # ESPACE
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        break
    elif key == 27:  # ESC
        print("🚪 Fermeture...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# ---------------- 5️⃣ Détection et recadrage ----------------
face = mtcnn(img)
if face is None:
    print("❌ Aucun visage détecté.")
    exit()

# ---------------- 6️⃣ Extraction de l'embedding ----------------
embedding = model(face.unsqueeze(0).to(device))
embedding_np = embedding.detach().cpu().numpy()[0]

# ---------------- 7️⃣ Reconnaissance ----------------
person, distance = recognize(embedding_np, embeddings_dict, threshold=0.7)
print(f"👤 Personne reconnue : {person} (distance = {distance:.4f})")
