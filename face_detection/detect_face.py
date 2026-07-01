import os
import base64
import io
import numpy as np
from pathlib import Path
from PIL import Image

import cv2 as cv
from ultralytics import YOLO



def preprocess_gradient(img):
    """
    NIR face preprocessing:
    - Convert to grayscale if needed
    - Morphological gradient enhancement
    - CLAHE local contrast enhancement
    
    Returns:
        enhanced : final processed image
        combined : gradient-enhanced image before CLAHE
    """

    # Convert to grayscale if input is RGB/BGR
    if len(img.shape) == 3:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()


    # Ensure uint8
    gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)


    # Morphological gradient
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (3, 3)
    )

    gradient = cv.morphologyEx(
        gray_img,
        cv.MORPH_GRADIENT,
        kernel
    )


    # Blend original image with edges
    # Keep original dominant to avoid over-enhancement
    combined = cv.addWeighted(
        gray_img,
        0.8,
        gradient,
        0.2,
        0
    )


    combined = np.clip(
        combined,
        0,
        255
    ).astype(np.uint8)


    # CLAHE instead of equalizeHist
    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_gray = clahe.apply(combined)

    # IMPORTANT:
    # convert back to 3 channels for YOLO
    enhanced_bgr = cv.cvtColor(
        enhanced_gray,
        cv.COLOR_GRAY2BGR
    )

    combined_bgr = cv.cvtColor(
            combined,
            cv.COLOR_GRAY2BGR
        )

    return enhanced_bgr, combined_bgr

def compute_face_quality(image: np.ndarray) -> float:
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) if len(image.shape)==3 else image
    laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
    brightness = np.mean(gray) / 255.0
    sharp_score = min(laplacian_var / 100, 1.0)
    bright_score = 1.0 - abs(brightness - 0.5)*2
    quality = (sharp_score*0.6 + bright_score*0.4) * 100
    return min(max(quality, 50), 99)

def base64_to_image(base64_str: str) -> np.ndarray:
    """Convertit une chaîne base64 en image numpy (format RGB)"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    # Convertir en RGB si l'image est en RGBA ou autre mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

"""
load_detection_model():
a utility function to load the face detection model from memory

"""


def load_detection_model():
    # Chemin relatif par rapport au fichier detect_face.py
    model_path = Path(__file__).parent / "models" / "yolov8n_100e.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    model = YOLO(str(model_path))
    print("Loading the face detection model...")
    return model


# def load_detection_model():
#     model = YOLO("yolov8m_200e.pt")
#     print("Loading the face detection model...")
#     return model

detection_model = load_detection_model()
