# backend/main_v2.py

import os, io, base64, json, pickle, numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime

# importing modules of the biometric system
from face_detection import detect_face
from feature_extraction import LBP
#from models import ELM


from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from skimage import color, feature
import cv2
import uvicorn
from pathlib import Path
from fastapi.staticfiles import StaticFiles


# ---------- Modèles Pydantic (DÉPLACÉS ICI) ----------
class EnrollRequest(BaseModel):
    subject_id: str
    name: str
    face_image: str
    quality_threshold: float = 0.35
    role: str = "Utilisateur"

class VerifyRequest(BaseModel):
    subject_id: str
    probe_face: str
    mode: str = "face"
    threshold: float = 0.99

class IdentifyRequest(BaseModel):
    probe_face: str
    max_candidates: int = 5
    #min_threshold: float = 0.987
# ---------- ELM & BBA ----------

class ELM:
    def __init__(self, input_dim, hidden_dim=300):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = self.b = self.beta = self.encoder = None
    def _sigmoid(self, X): return 1/(1+np.exp(-X))

    def fit(self, X, y):
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        T = encoder.fit_transform(y.reshape(-1,1))
        self.encoder = encoder
        self.W = np.random.randn(self.input_dim, self.hidden_dim)
        self.b = np.random.randn(self.hidden_dim)
        H = self._sigmoid(X @ self.W + self.b)
        self.beta = np.linalg.pinv(H) @ T

    def predict(self, X):
        H = self._sigmoid(X @ self.W + self.b)
        return np.argmax(H @ self.beta, axis=1)
    
    def predict_proba(self, X):
        H = self._sigmoid(X @ self.W + self.b)
        output = H @ self.beta
        exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def predict_probas(self, x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
    
class BBAFeatureSelector:
    def __init__(self, n_bats=20, n_iter=20, fmin=0, fmax=2):
        self.n_bats, self.n_iter, self.fmin, self.fmax = n_bats, n_iter, fmin, fmax
        self.best_mask = None

# ---------- Extraction LBP ----------
def extract_lbp_features_from_image(image: np.ndarray, target_size: int = 800) -> np.ndarray:
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float64)/255.0 if image.dtype != np.float64 else image
    h, w = gray.shape
    if h != 128 or w != 128:
        gray = cv2.resize(gray.astype(np.float32), (128,128))
    radius, n_points, method = 1, 8, 'uniform'
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = 128//grid_rows, 128//grid_cols
    histograms = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            cell = gray[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            lbp = feature.local_binary_pattern(cell, n_points, radius, method)
            hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0,59))
            hist = hist.astype(np.float64)
            hist /= (hist.sum() + 1e-7)
            histograms.append(hist)
    all_hist = np.concatenate(histograms)
    if len(all_hist) >= target_size:
        features = all_hist[:target_size]
    else:
        features = np.pad(all_hist, (0, target_size - len(all_hist)))
    return features

# ---------- Utilitaires ----------

#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ---------- FastAPI ----------
BASE_DIR = Path(__file__).parent.absolute()
app = FastAPI(title="BioSaaS Biometric API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/debug/images")
async def debug_images():
    import os
    from pathlib import Path
    #images_dir = Path(__file__).parent.absolute() / "backend" / "images"
    images_dir = Path(__file__).parent.absolute() / "images"
    if not images_dir.exists():
        return {"error": f"Le dossier {images_dir} n'existe pas"}
    files = []
    for root, dirs, filenames in os.walk(images_dir):
        for f in filenames:
            files.append(str(Path(root) / f))
    return {
        "directory": str(images_dir),
        "exists": images_dir.exists(),
        "files": files[:20]  # limite à 20 fichiers
    }

images_dir = Path(__file__).parent / "backend" / "images"
if images_dir.exists():
    #app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")
    print(f"✅ Dossier images monté : {images_dir}")
else:
    print(f"⚠️ Dossier images introuvable : {images_dir}")

MODEL = None
SUBJECTS = []
SUBJECT_ID_MAP = {}

@app.on_event("startup")
async def startup_event():
    global MODEL
    model_path = BASE_DIR / "models" / "face_elm_bba.pkl"
    if not model_path.exists():
        MODEL = {'mask': np.ones(800, dtype=bool)[:411]}
        print("⚠️ Modèle non trouvé, utilisation d'un masque similaire.")
    else:
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                # Si c'est un dict avec 'mask', on le prend
                if isinstance(data, dict) and 'mask' in data:
                    MODEL = {'mask': data['mask']}
                    print(f"✅ Masque chargé (longueur {len(MODEL['mask'])})")
                else:
                    # Fallback : masque factice
                    MODEL = {'mask': np.ones(800, dtype=bool)[:411]}
                    print("⚠️ Fichier modèle inattendu, masque factice utilisé.")
        except Exception as e:
            print(f"❌ Erreur de chargement du modèle: {e}")
            MODEL = {'mask': np.ones(800, dtype=bool)[:411]}
            print("⚠️ Utilisation du masque de remplacement.")

#@app.on_event("startup")
#async def startup_event():
    #global MODEL
    #model_path = BASE_DIR / "models" / "face_elm_bba.pkl"
    #if not model_path.exists():
        #MODEL = {'mask': np.ones(800, dtype=bool)[:411]}
        #print("⚠️ Modèle non trouvé, utilisation d'un masque factice.")
    #else:
        #with open(model_path, 'rb') as f:
            #MODEL = pickle.load(f)
        #print(f"✅ Modèle chargé: {model_path}")
        #print(f"   Masque longueur: {len(MODEL.get('mask', []))}")

# ---------- API Endpoints ----------
elm_classifier = ELM(5000, 300)

@app.get("/", response_class=HTMLResponse)
async def root():

    html_path_v4 = BASE_DIR / "static" / "biometric_saas_final_demo_v4.html"
    
    if html_path_v4.exists():
        path = html_path_v4
    else:
        return HTMLResponse("<h1>Fichier HTML non trouvé</h1>")
    
    with open(path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get("/demo", response_class=HTMLResponse)
async def demo():
    return await root()

@app.get("/api/v1/device/status")
async def device_status():
    return {"status":"connected","device":"VistaFA2","sdk":"MegaMatcher 12.1"}

@app.get("/api/v1/subjects")
async def list_subjects():
    return {"subjects": SUBJECTS, "count": len(SUBJECTS)}

@app.delete("/api/v1/subjects/{subject_id}")
async def delete_subject(subject_id: str):
    if subject_id not in SUBJECT_ID_MAP:
        raise HTTPException(404, "Sujet non trouvé")
    SUBJECTS[:] = [s for s in SUBJECTS if s["id"] != subject_id]
    del SUBJECT_ID_MAP[subject_id]
    return {"status":"deleted"}

@app.post("/api/v1/enroll")
async def enroll(request: EnrollRequest):
    try:
        # conversion into a compatibe format to send it via a web server
        image = detect_face.base64_to_image(request.face_image)
        # computing the face quality
        quality = detect_face.compute_face_quality(image)
        if quality < request.quality_threshold * 100:
            raise HTTPException(400, f"Qualité insuffisante: {quality:.1f}%")
        # preprocessing of the image
        enhanced, _ = detect_face.preprocess_gradient(image)
        # detection of the face image

        results = detect_face.detection_model(enhanced)
        if not results or len(results[0].boxes) == 0:
            raise HTTPException(400, "Aucun visage détecté")
        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        face_crop = enhanced[y1:y2, x1:x2]

        # fature extraction from the face image
        features = LBP.get_face_features(face_crop, MODEL)
        print(f"✅ Enroll features shape: {features.shape}")

        if request.subject_id in SUBJECT_ID_MAP:
            raise HTTPException(400, "ID existe déjà")
        
        # feature storing
        new_subject = {
            "id": request.subject_id,
            "name": request.name,
            "role": request.role,
            "face_quality": round(quality),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "features": features.tolist()
        }
        SUBJECTS.append(new_subject)
        SUBJECT_ID_MAP[request.subject_id] = new_subject
        return {"status":"success", "subject":{"id":request.subject_id,"name":request.name,
                "role":request.role,"quality":round(quality),"date":new_subject["date"]}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/v1/verify")
async def verify(request: VerifyRequest):
    try:
        if request.subject_id not in SUBJECT_ID_MAP:
            raise HTTPException(404, "Sujet non trouvé")
        subject = SUBJECT_ID_MAP[request.subject_id]
        probe_image = detect_face.base64_to_image(request.probe_face)

        # preprocessing of the image
        enhanced, _ = detect_face.preprocess_gradient(probe_image)
        # detection of the face image

        results = detect_face.detection_model(enhanced)
        if not results or len(results[0].boxes) == 0:
            raise HTTPException(400, "Aucun visage détecté")
        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        face_crop = enhanced[y1:y2, x1:x2]

        probe_features = LBP.get_face_features(face_crop, MODEL)
        print(f"✅ verification features shape: {probe_features.shape}")

        if "features" not in subject:
            score, is_match = 0.0, False
        else:
            enrolled_features = np.array(subject["features"])
            score = elm_classifier.predict_probas(probe_features, enrolled_features)

            is_match = score >= request.threshold
        return {
            "status":"success",
            "decision":"MATCH" if is_match else "NO MATCH",
            "scores":{"face":round(score,3),"fusion":round(score,3)},
            "liveness":{"live":True,"score":0.99},
            "processing_ms":200,
            "megamatcher_ver":"12.1",
            "subject": subject["name"] if is_match else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    

@app.post("/api/v1/identify")
async def identify(request: IdentifyRequest):
    try:
        probe_image = detect_face.base64_to_image(request.probe_face)

        # preprocessing of the image
        enhanced, _ = detect_face.preprocess_gradient(probe_image)

        # detection of the face image
        results = detect_face.detection_model(enhanced)

        if not results or len(results[0].boxes) == 0:
            raise HTTPException(400, "Aucun visage détecté")
        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        face_crop = enhanced[y1:y2, x1:x2]

        probe_features = detect_face.get_face_features(face_crop, MODEL)
        print(f"✅ identification features shape: {probe_features.shape}")
        candidates = []
        for subject in SUBJECTS:
            if "features" in subject:
                enrolled_features = np.array(subject["features"])
                score = elm_classifier.predict_probas(probe_features, enrolled_features)
                candidates.append({"id":subject["id"],"name":subject["name"],
                                   "role":subject["role"],"score":round(score,3),"rank":0})
        candidates.sort(key=lambda x: x["score"], reverse=True)
        for i, c in enumerate(candidates):
            c["rank"] = i+1
        candidates = candidates[:request.max_candidates]
        return {"status":"success","candidates":candidates,
                "total_subjects":len(SUBJECTS),"processing_ms":150}
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- Lancer ----------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    # uvicorn.run("main_v2:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    
