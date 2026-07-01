# backend/init_demo.py - Script pour initialiser la démo avec des images
"""
Script d'initialisation pour la démo BioSaaS.
Placez vos images dans backend/images/ avec la structure:
  backend/images/
    DZ-1985-038472/
        face_1.jpg
        face_2.jpg
    DZ-1992-001122/
        face_1.jpg
    ...

Puis exécutez ce script pour initialiser la base.
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

BASE_URL = "http://localhost:8000"
IMAGES_DIR = Path("backend/images")
MODEL_PATH = Path("backend/models/face_elm_bba.pkl")

# Sujets par défaut
DEFAULT_SUBJECTS = [
    {"id": "DZ-000", "name": "Person 0", "role": "Administrateur"},
    {"id": "DZ-001", "name": "Person 1", "role": "Opérateur"},
    {"id": "DZ-002", "name": "Person 2", "role": "Agent terrain"},
    {"id": "DZ-003", "name": "Person 3", "role": "Administrateur"},
    {"id": "DZ-004", "name": "Person 4", "role": "Administrateur"},
    {"id": "DZ-005", "name": "Person 5", "role": "Administrateur"},
    {"id": "DZ-006", "name": "Person 6", "role": "Administrateur"},
    {"id": "DZ-007", "name": "Person 7", "role": "Administrateur"},
    {"id": "DZ-008", "name": "Person 8", "role": "Administrateur"},
    {"id": "DZ-009", "name": "Person 9", "role": "Administrateur"},
    {"id": "DZ-010", "name": "Person 10", "role": "Administrateur"},
    {"id": "DZ-011", "name": "Person 11", "role": "Administrateur"},
    {"id": "DZ-012", "name": "Person 12", "role": "Administrateur"},
    {"id": "DZ-013", "name": "Person 13", "role": "Administrateur"},
    {"id": "DZ-014", "name": "Person 14", "role": "Administrateur"},
    {"id": "DZ-015", "name": "Person 15", "role": "Administrateur"},
    {"id": "DZ-016", "name": "Person 16", "role": "Administrateur"},
    {"id": "DZ-017", "name": "Person 17", "role": "Administrateur"},
    {"id": "DZ-018", "name": "Person 18", "role": "Administrateur"},
    {"id": "DZ-019", "name": "Person 19", "role": "Administrateur"},
    {"id": "DZ-020", "name": "Person 20", "role": "Administrateur"},
    {"id": "DZ-021", "name": "Person 21", "role": "Administrateur"},
    {"id": "DZ-022", "name": "Person 22", "role": "Administrateur"},
    {"id": "DZ-023", "name": "Person 23", "role": "Administrateur"},
    {"id": "DZ-024", "name": "Person 24", "role": "Administrateur"},
]


def image_to_base64(image_path: Path) -> str:
    """Convertit une image en base64"""
    with open(image_path, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode("utf-8")


def enroll_subject(subject_id: str, name: str, role: str, image_path: Path) -> bool:
    """Enrôle un sujet via l'API"""
    if not image_path.exists():
        print(f"⚠️ Image non trouvée: {image_path}")
        return False

    try:
        # Charger l'image
        img = Image.open(image_path)
        # Convertir en JPEG pour l'API
        import io
        buf = io.BytesIO()
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(buf, format='JPEG', quality=92)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Appeler l'API
        response = requests.post(
            f"{BASE_URL}/api/v1/enroll",
            json={
                "subject_id": subject_id,
                "name": name,
                "face_image": f"data:image/jpeg;base64,{img_base64}",
                "quality_threshold": 0.45,
                "role": role
            }
        )

        if response.status_code == 200:
            print(f"✅ Enrôlé: {name} ({subject_id})")
            return True
        else:
            print(f"❌ Erreur pour {name}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception pour {name}: {e}")
        return False


def find_face_images(subject_id: str) -> list:
    """Trouve les images de visage pour un sujet"""
    subject_dir = IMAGES_DIR / subject_id
    if not subject_dir.exists():
        return []
    images = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        images.extend(subject_dir.glob(f"*{ext}"))
    return sorted(images)


def main():
    """Initialise la démo avec les images disponibles"""
    print("=" * 60)
    print("🔬 Initialisation de la démo BioSaaS")
    print("=" * 60)

    # Vérifier que le serveur tourne
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("⚠️ Le serveur n'est pas accessible.")
            print("   Lancez d'abord: python main.py")
            sys.exit(1)
        print("✅ Serveur accessible")
    except:
        print("⚠️ Le serveur n'est pas accessible.")
        print("   Lancez d'abord: python main.py")
        sys.exit(1)

    # Pour chaque sujet, trouver une image et l'enrôler
    enrolled_count = 0
    for subject in DEFAULT_SUBJECTS:
        subject_id = subject["id"]
        images = find_face_images(subject_id)

        if images:
            # Utiliser la première image
            img_path = images[0]
            print(f"📷 Utilisation de {img_path} pour {subject['name']}")
            if enroll_subject(subject_id, subject["name"], subject["role"], img_path):
                enrolled_count += 1
        else:
            print(f"⚠️ Aucune image trouvée pour {subject['name']} ({subject_id})")
            print(f"   Placez une image dans: {IMAGES_DIR}/{subject_id}/")

    print("=" * 60)
    print(f"📊 {enrolled_count}/{len(DEFAULT_SUBJECTS)} sujets enrôlés")
    print("🌐 Ouvrez http://localhost:8000/demo pour la démo")
    print("=" * 60)


if __name__ == "__main__":
    main()