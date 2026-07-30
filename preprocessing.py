"""
Fonctions de prétraitement d'image basées sur OpenCV.

Ces étapes sont classiques en préparation d'un pipeline OCR :
- passage en niveaux de gris (réduit la dimensionnalité, requis par la plupart des moteurs)
- débruitage (limite les faux positifs dus au bruit du scanner/photo)
- binarisation adaptative (accentue le contraste texte/fond, utile sur documents mal éclairés)
"""

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convertit une image PIL (RGB) en tableau OpenCV (BGR)."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convertit un tableau OpenCV (BGR ou niveaux de gris) en image PIL (RGB)."""
    if len(image.shape) == 2:
        # image en niveaux de gris / binaire
        return Image.fromarray(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convertit une image BGR en niveaux de gris."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray) -> np.ndarray:
    """Débruite une image en niveaux de gris (Non-Local Means)."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Binarise l'image avec un seuillage adaptatif gaussien.
    Nécessite une image en niveaux de gris en entrée.
    """
    gray = to_grayscale(image)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


def preprocess_image(
    image: Image.Image,
    use_grayscale: bool = True,
    use_denoise: bool = True,
    use_binarization: bool = True,
) -> Image.Image:
    """
    Applique la chaîne de prétraitement sélectionnée par l'utilisateur.

    Chaque étape est optionnelle et s'applique dans l'ordre :
    niveaux de gris -> débruitage -> binarisation adaptative.

    Retourne une image PIL prête à être envoyée aux moteurs OCR.
    """
    cv_image = pil_to_cv2(image)

    if use_grayscale or use_binarization:
        # la binarisation adaptative requiert de toute façon une image en niveaux de gris
        cv_image = to_grayscale(cv_image)

    if use_denoise:
        cv_image = denoise(cv_image)

    if use_binarization:
        cv_image = adaptive_threshold(cv_image)

    return cv2_to_pil(cv_image)
