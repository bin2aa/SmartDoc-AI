"""OCR configuration settings rendering for SmartDoc AI."""

import streamlit as st

from src.views.components import UIComponents
from src.utils.constants import (
    DEFAULT_OCR_LANG,
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_PSM,
    DEFAULT_OCR_OEM,
    DEFAULT_OCR_PREPROCESS,
    DEFAULT_OCR_AUTO_PDF,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def render_ocr_settings(
    components: UIComponents,
    persist_fn,
):
    """Render OCR configuration settings.

    Args:
        components: UIComponents instance for UI rendering
        persist_fn: Callback function to persist settings to disk
    """
    st.subheader("OCR Configuration")

    col1, col2 = st.columns(2)

    with col1:
        ocr_lang = st.text_input(
            "OCR Language Pack",
            value=st.session_state.get("ocr_lang", DEFAULT_OCR_LANG),
            help="Tesseract language codes (example: vie+eng).",
        )

        ocr_dpi = st.slider(
            "PDF Render DPI",
            min_value=150,
            max_value=400,
            value=int(st.session_state.get("ocr_dpi", DEFAULT_OCR_DPI)),
            step=25,
            help="Higher DPI can improve OCR but is slower.",
        )

    with col2:
        ocr_psm = st.selectbox(
            "Page Segmentation Mode (PSM)",
            options=[3, 4, 6, 11],
            index=[3, 4, 6, 11].index(
                int(st.session_state.get("ocr_psm", DEFAULT_OCR_PSM))
            )
            if int(st.session_state.get("ocr_psm", DEFAULT_OCR_PSM)) in [3, 4, 6, 11]
            else 2,
            help="PSM 6 works well for most documents.",
        )

        ocr_oem = st.selectbox(
            "OCR Engine Mode (OEM)",
            options=[1, 3],
            index=[1, 3].index(
                int(st.session_state.get("ocr_oem", DEFAULT_OCR_OEM))
            )
            if int(st.session_state.get("ocr_oem", DEFAULT_OCR_OEM)) in [1, 3]
            else 1,
            help="OEM 3 uses the default LSTM engine.",
        )

    col3, col4 = st.columns(2)
    with col3:
        ocr_preprocess = st.toggle(
            "Enable Image Preprocessing",
            value=bool(st.session_state.get("ocr_preprocess", DEFAULT_OCR_PREPROCESS)),
            help="Apply grayscale, contrast, and denoise before OCR.",
        )
    with col4:
        ocr_auto_pdf = st.toggle(
            "Auto OCR Scanned PDFs Only",
            value=bool(st.session_state.get("ocr_auto_pdf", DEFAULT_OCR_AUTO_PDF)),
            help="Skip OCR when the PDF already contains usable text.",
        )

    if st.button("Apply OCR Settings", type="primary"):
        st.session_state.ocr_lang = ocr_lang.strip() or DEFAULT_OCR_LANG
        st.session_state.ocr_dpi = int(ocr_dpi)
        st.session_state.ocr_psm = int(ocr_psm)
        st.session_state.ocr_oem = int(ocr_oem)
        st.session_state.ocr_preprocess = bool(ocr_preprocess)
        st.session_state.ocr_auto_pdf = bool(ocr_auto_pdf)
        persist_fn()
        components.success_alert("OCR settings updated and saved")
