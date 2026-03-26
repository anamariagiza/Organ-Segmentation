import os
import cv2


def load_image(path: str):
    """Incarca imaginea de pe disc in alb si negru."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fisierul nu exista: {path}")
    img_gs = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gs is None:
        raise ValueError(f"Nu s-a putut citi imaginea: {path}")
    print(f"Imagine incarcata cu succes:{img_gs.shape}")
    return img_gs
