"""
Moteur OCR EasyOCR (architecture CRNN : CNN + BiLSTM + CTC).

Le modèle est téléchargé automatiquement au premier lancement (~ quelques
centaines de Mo) et mis en cache disque par EasyOCR, ainsi qu'en cache
mémoire Streamlit (st.cache_resource) pour éviter de le recharger à chaque
interaction utilisateur.
"""

import time

import numpy as np
import streamlit as st
from PIL import Image


class EngineError(Exception):
    """Erreur métier remontée à l'interface Streamlit."""


@st.cache_resource(show_spinner="Chargement du modèle EasyOCR (première fois seulement)...")
def _load_reader(langs: tuple):
    import easyocr

    try:
        return easyocr.Reader(list(langs), gpu=False)
    except Exception as exc:
        raise EngineError(f"Impossible de charger le modèle EasyOCR : {exc}") from exc


def run_easyocr(image: Image.Image, langs=("fr", "en")) -> dict:
    """
    Exécute EasyOCR sur l'image fournie.

    Retourne un dictionnaire :
        text        : texte reconnu (lignes concaténées)
        elapsed     : temps d'exécution en secondes
        confidence  : confiance moyenne (0-100) renvoyée nativement par EasyOCR
    """
    reader = _load_reader(tuple(langs))

    start = time.perf_counter()
    try:
        results = reader.readtext(np.array(image.convert("RGB")))
    except Exception as exc:
        raise EngineError(f"Échec de l'OCR EasyOCR : {exc}") from exc
    elapsed = time.perf_counter() - start

    lines = [text for (_, text, _) in results]
    confidences = [conf for (_, _, conf) in results]
    confidence = (sum(confidences) / len(confidences) * 100) if confidences else None

    return {
        "engine": "EasyOCR",
        "text": "\n".join(lines).strip(),
        "elapsed": elapsed,
        "confidence": confidence,
    }
