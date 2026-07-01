import numpy as np
from typing import Optional, List, Dict, Any
from PIL import Image
from skimage import color, feature
import cv2 as cv


# ---------- Extraction LBP ----------
def extract_lbp_features_from_image(image: np.ndarray, target_size: int = 800) -> np.ndarray:
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float64)/255.0 if image.dtype != np.float64 else image
    h, w = gray.shape
    if h != 128 or w != 128:
        gray = cv.resize(gray.astype(np.float32), (128,128))
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

def get_face_features(image: np.ndarray, model_dict: Dict) -> np.ndarray:
    lbp_hist = extract_lbp_features_from_image(image, target_size=800)
    mask = model_dict.get('mask')
    if mask is not None:
        if len(lbp_hist) != len(mask):
            if len(lbp_hist) > len(mask):
                lbp_hist = lbp_hist[:len(mask)]
            else:
                lbp_hist = np.pad(lbp_hist, (0, len(mask)-len(lbp_hist)))
        return lbp_hist[mask == 1]
    return lbp_hist