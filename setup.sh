#!/bin/bash
# backend/setup.sh - Script d'installation et de démarrage

echo "🔬 Installation des dépendances..."
pip install -r requirements.txt

echo "📁 Création des dossiers..."
mkdir -p backend/static backend/models backend/images

echo "📄 Copie du fichier HTML..."
cp ../biometric_saas_final_demo.html backend/static/

echo "📦 Vérification du modèle..."
if [ -f "backend/models/face_elm_bba.pkl" ]; then
    echo "✅ Modèle trouvé"
else
    echo "⚠️ Modèle non trouvé: backend/models/face_elm_bba.pkl"
    echo "   Placez votre fichier .pkl à cet emplacement"
fi

echo "🚀 Démarrage du serveur..."
python main.py