"""
Application Streamlit de comparaison de moteurs OCR.

Projet académique - Master 1 SDIA, module Computer Vision.
Compare trois architectures de reconnaissance de caractères sur un même
document uploadé par l'utilisateur :
    - Tesseract   : moteur LSTM (classique, open-source, Google)
    - EasyOCR     : CRNN (CNN + BiLSTM + CTC)
    - TrOCR       : Transformer ViT encoder-decoder (Microsoft Research)
"""

import io

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from ocr_engines import easyocr_engine, tesseract_engine, trocr_engine
from preprocessing import preprocess_image

st.set_page_config(
    page_title="Comparaison de moteurs OCR",
    page_icon="🔎",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Chargement du document (image ou PDF)
# ----------------------------------------------------------------------------
def load_document(uploaded_file) -> Image.Image:
    """Charge le fichier uploadé (image ou PDF) et renvoie une image PIL RGB."""
    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]

    if suffix == "pdf":
        try:
            from pdf2image import convert_from_bytes
        except ImportError as exc:
            raise RuntimeError(
                "Le support PDF nécessite le paquet `pdf2image`. "
                "Installez-le avec `pip install pdf2image` (voir requirements.txt)."
            ) from exc

        try:
            pages = convert_from_bytes(uploaded_file.getvalue())
        except Exception as exc:
            raise RuntimeError(
                "Conversion PDF impossible. Vérifiez que `poppler-utils` est installé "
                "sur le système (`sudo apt-get install poppler-utils`, voir README). "
                f"Détail : {exc}"
            ) from exc

        if not pages:
            raise RuntimeError("Le PDF ne contient aucune page exploitable.")

        if len(pages) > 1:
            page_num = st.selectbox(
                "Le PDF contient plusieurs pages, choisissez celle à analyser :",
                options=list(range(1, len(pages) + 1)),
            )
            return pages[page_num - 1].convert("RGB")
        return pages[0].convert("RGB")

    try:
        return Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except UnidentifiedImageError as exc:
        raise RuntimeError(
            "Le fichier fourni n'est pas une image lisible (png/jpg/jpeg)."
        ) from exc


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("🔎 Comparaison de moteurs OCR")
st.markdown(
    """
Ce projet accompagne un rapport académique **(M1-SDIA, module Computer Vision)**
comparant trois architectures OCR :

| Moteur | Architecture |
|---|---|
| **Tesseract** | LSTM |
| **EasyOCR** | CRNN (CNN + BiLSTM + CTC) |
| **TrOCR** | Transformer ViT encoder-decoder (Microsoft) |
"""
)

uploaded_file = st.file_uploader(
    "Déposez un document (image ou PDF)",
    type=["png", "jpg", "jpeg", "pdf"],
)

if uploaded_file is None:
    st.info("Veuillez uploader une image (.png, .jpg, .jpeg) ou un PDF pour commencer.")
    st.stop()

try:
    original_image = load_document(uploaded_file)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

# ----------------------------------------------------------------------------
# Barre latérale : options
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Options")

engine_choice = st.sidebar.radio(
    "Moteur(s) à appliquer",
    options=["Tesseract", "EasyOCR", "TrOCR", "Comparer les 3"],
)

trocr_variant_label = None
if engine_choice in ("TrOCR", "Comparer les 3"):
    trocr_variant_label = st.sidebar.radio(
        "Variante TrOCR",
        options=["Texte imprimé (printed)", "Écriture manuscrite (handwritten)"],
    )
trocr_variant = "handwritten" if trocr_variant_label and "manuscrite" in trocr_variant_label else "printed"

st.sidebar.subheader("Prétraitement OpenCV")
use_preprocessing = st.sidebar.checkbox("Activer le prétraitement", value=False)
use_grayscale = use_denoise = use_binarization = False
if use_preprocessing:
    use_grayscale = st.sidebar.checkbox("Niveaux de gris", value=True)
    use_denoise = st.sidebar.checkbox("Débruitage", value=True)
    use_binarization = st.sidebar.checkbox("Binarisation adaptative", value=True)

# ----------------------------------------------------------------------------
# Prétraitement et affichage avant/après
# ----------------------------------------------------------------------------
image_to_process = original_image
if use_preprocessing:
    try:
        processed_image = preprocess_image(
            original_image,
            use_grayscale=use_grayscale,
            use_denoise=use_denoise,
            use_binarization=use_binarization,
        )
    except Exception as exc:
        st.error(f"Erreur lors du prétraitement : {exc}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Avant")
        st.image(original_image, use_container_width=True)
    with col2:
        st.subheader("Après")
        st.image(processed_image, use_container_width=True)

    image_to_process = processed_image
else:
    st.subheader("Document")
    st.image(original_image, use_container_width=True)

# ----------------------------------------------------------------------------
# Exécution OCR
# ----------------------------------------------------------------------------
ENGINE_RUNNERS = {
    "Tesseract": lambda img: tesseract_engine.run_tesseract(img),
    "EasyOCR": lambda img: easyocr_engine.run_easyocr(img),
    "TrOCR": lambda img: trocr_engine.run_trocr(img, variant=trocr_variant),
}


def run_selected_engines(image: Image.Image, choice: str) -> list:
    """Exécute le(s) moteur(s) sélectionné(s) et renvoie la liste des résultats."""
    engines_to_run = list(ENGINE_RUNNERS.keys()) if choice == "Comparer les 3" else [choice]
    results = []

    for name in engines_to_run:
        with st.spinner(f"Exécution de {name}..."):
            try:
                result = ENGINE_RUNNERS[name](image)
                result["error"] = None
            except (tesseract_engine.EngineError, easyocr_engine.EngineError, trocr_engine.EngineError) as exc:
                result = {"engine": name, "text": "", "elapsed": None, "confidence": None, "error": str(exc)}
            except Exception as exc:  # filet de sécurité pour toute erreur imprévue
                result = {
                    "engine": name,
                    "text": "",
                    "elapsed": None,
                    "confidence": None,
                    "error": f"Erreur inattendue : {exc}",
                }
        results.append(result)
    return results


if st.button("🚀 Lancer la reconnaissance", type="primary"):
    results = run_selected_engines(image_to_process, engine_choice)

    st.header("Résultats")

    for result in results:
        with st.expander(f"📄 {result['engine']}", expanded=True):
            if result["error"]:
                st.error(result["error"])
                continue

            st.text_area(
                "Texte reconnu",
                value=result["text"] or "(aucun texte détecté)",
                height=180,
                key=f"text_{result['engine']}",
            )
            metric_cols = st.columns(2)
            metric_cols[0].metric("Temps d'exécution", f"{result['elapsed']:.2f} s")
            if result["confidence"] is not None:
                metric_cols[1].metric("Confiance moyenne", f"{result['confidence']:.1f} %")
            else:
                metric_cols[1].metric("Confiance moyenne", "N/A")

    # Tableau comparatif et graphique si plusieurs moteurs ont été exécutés
    if engine_choice == "Comparer les 3":
        st.header("📊 Comparaison")

        valid_results = [r for r in results if r["error"] is None]
        if valid_results:
            df = pd.DataFrame(
                [
                    {
                        "Moteur": r["engine"],
                        "Temps (s)": round(r["elapsed"], 3),
                        "Confiance (%)": round(r["confidence"], 1) if r["confidence"] is not None else None,
                        "Longueur du texte": len(r["text"]),
                    }
                    for r in valid_results
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("Moteur")["Temps (s)"])
        else:
            st.warning("Aucun moteur n'a produit de résultat exploitable.")

st.sidebar.markdown("---")
st.sidebar.caption("Projet académique M1-SDIA · Module Computer Vision")
