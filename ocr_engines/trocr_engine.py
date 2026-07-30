"""
Moteur OCR TrOCR (Transformer ViT encoder-decoder, Microsoft Research).

Deux variantes pré-entraînées sont proposées :
    - microsoft/trocr-base-printed      : optimisée pour le texte imprimé
    - microsoft/trocr-base-handwritten  : optimisée pour l'écriture manuscrite

TrOCR est conçu à l'origine pour reconnaître UNE ligne de texte à la fois
(contrairement à Tesseract/EasyOCR qui gèrent nativement une page entière).
Sur un document multi-lignes, le résultat peut donc être partiel : c'est un
point de comparaison intéressant à souligner dans le rapport académique.
"""

import time

import streamlit as st
from PIL import Image

MODEL_IDS = {
    "printed": "microsoft/trocr-base-printed",
    "handwritten": "microsoft/trocr-base-handwritten",
}


class EngineError(Exception):
    """Erreur métier remontée à l'interface Streamlit."""


@st.cache_resource(show_spinner="Chargement du modèle TrOCR (première fois seulement)...")
def _load_model(variant: str):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_id = MODEL_IDS[variant]
    try:
        processor = TrOCRProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
    except Exception as exc:
        raise EngineError(
            f"Impossible de télécharger/charger le modèle TrOCR '{model_id}'. "
            f"Vérifiez votre connexion internet et l'espace disque disponible. "
            f"Détail : {exc}"
        ) from exc
    return processor, model


def run_trocr(image: Image.Image, variant: str = "printed") -> dict:
    """
    Exécute TrOCR sur l'image fournie.

    Retourne un dictionnaire :
        text        : texte reconnu
        elapsed     : temps d'exécution en secondes
        confidence  : None (TrOCR ne fournit pas nativement de score de confiance)
    """
    if variant not in MODEL_IDS:
        raise EngineError(f"Variante TrOCR inconnue : {variant}")

    processor, model = _load_model(variant)

    start = time.perf_counter()
    try:
        pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as exc:
        raise EngineError(f"Échec de l'OCR TrOCR : {exc}") from exc
    elapsed = time.perf_counter() - start

    return {
        "engine": f"TrOCR ({variant})",
        "text": text.strip(),
        "elapsed": elapsed,
        "confidence": None,
    }
