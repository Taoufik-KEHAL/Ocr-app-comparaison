"""
Moteur OCR Tesseract (moteur LSTM, via le binding Python pytesseract).

Tesseract nécessite le binaire système `tesseract-ocr` (voir README).
"""

import time

import pytesseract
from PIL import Image


class EngineError(Exception):
    """Erreur métier remontée à l'interface Streamlit."""


def is_available() -> bool:
    """Vérifie que le binaire tesseract est bien installé et accessible."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def run_tesseract(image: Image.Image, lang: str = "fra+eng") -> dict:
    """
    Exécute Tesseract sur l'image fournie.

    Retourne un dictionnaire :
        text        : texte reconnu
        elapsed     : temps d'exécution en secondes
        confidence  : confiance moyenne (0-100) calculée à partir des mots détectés
    """
    start = time.perf_counter()
    try:
        text = pytesseract.image_to_string(image, lang=lang)
        # image_to_data fournit une confiance par mot détecté, on en fait la moyenne
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data.get("conf", []) if c not in ("-1", -1)]
        confidence = sum(confidences) / len(confidences) if confidences else None
    except pytesseract.TesseractNotFoundError as exc:
        raise EngineError(
            "Le binaire Tesseract est introuvable. Installez-le avec "
            "`sudo apt-get install tesseract-ocr tesseract-ocr-fra` (voir README)."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise EngineError(f"Erreur Tesseract : {exc}") from exc
    except Exception as exc:  # image illisible, langue absente, etc.
        raise EngineError(f"Échec de l'OCR Tesseract : {exc}") from exc
    elapsed = time.perf_counter() - start

    return {
        "engine": "Tesseract",
        "text": text.strip(),
        "elapsed": elapsed,
        "confidence": confidence,
    }
