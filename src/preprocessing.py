import cv2
import numpy as np


def preprocess_image(image, target_size=(512, 512)):
    """
    Preprocessing: Normalizare + Redimensionare + Imbunatatirea contrastului
    """
    original_shape = image.shape
    
    # Redimensionare
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalizare
    normalized = resized.astype(np.float32) / 255.0
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    
    # Reduce estomparea gaussiana
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    print(f"Preprocesare completa: {original_shape} → {denoised.shape}")
    return denoised, normalized, original_shape