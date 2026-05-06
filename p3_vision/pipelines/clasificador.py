# vision/pipelines/clasificador.py
# ─────────────────────────────────────────────────────────────
# Clasificador de imágenes: ¿balda de precios o etiqueta nutricional?
#
# ESTRATEGIA DEFINITIVA
# ─────────────────────
# Una sola pasada de EasyOCR sobre la imagen cruda (ocr_inicial) es la
# fuente de verdad. El texto resultante se analiza con keywords:
#
#   · Si aparecen keywords nutricionales (kcal, proteinas, hidratos…)
#     → ETIQUETA_NUTRI.  Confianza proporcional al nº de keywords.
#   · Si no hay keywords nutricionales pero sí patrones de precio decimal
#     (X,XX o X.XX) en cantidad → BALDA_PRECIOS.
#   · Señal de respaldo morfológica (contornos blancos alargados) solo
#     se usa para desempatar cuando el OCR no es concluyente.
#
# El texto + raw OCR se devuelven en el estado LangGraph para que
# pipeline_nutricional los reutilice sin repetir el OCR.
# ─────────────────────────────────────────────────────────────

import re
import cv2
import numpy as np

from vision.domain.schemas import ClasificacionImagen, TipoImagen
from vision.ocr.easyocr_client import ocr_inicial
from vision.ocr.parsers import contiene_keywords_nutri, contiene_keywords_precio

_RE_DECIMAL = re.compile(r"\d+[.,]\d{1,2}")


# ── Señal morfológica de respaldo ─────────────────────────────

def _contar_contornos_etiqueta(img_bgr: np.ndarray) -> int:
    """
    Cuenta contornos blancos con proporciones de etiqueta de precio.
    Solo se usa como desempate cuando el OCR no es concluyente.
    """
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 55, 255]))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if not (300 <= area <= 8000):
            continue
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if h == 0:
            continue
        if 1.5 <= max(w, h) / min(w, h) <= 8.0:
            n += 1
    return n


# ── Clasificador principal ────────────────────────────────────

def clasificar_imagen(img_bgr: np.ndarray) -> tuple:
    """
    Clasifica la imagen y devuelve (ClasificacionImagen, texto_ocr, raw_ocr).

    El texto_ocr y raw_ocr se propagan al estado LangGraph para que
    pipeline_nutricional los reutilice sin repetir la pasada OCR.

    Lógica de decisión:
      1. OCR inicial sobre imagen cruda → texto normalizado.
      2. Contar keywords nutricionales en el texto.
      3. Contar patrones decimales de precio (X,XX).
      4. Si keywords_nutri > 0  → ETIQUETA_NUTRI  (confianza ∝ n_keywords).
         Si decimales > 0 y sin keywords → BALDA_PRECIOS.
         Empate o ambos cero → morfología de respaldo.
    """
    texto, raw = ocr_inicial(img_bgr)

    kw_nutri_ok,  n_nutri  = contiene_keywords_nutri(texto)
    kw_precio_ok, n_precio_kw = contiene_keywords_precio(texto)
    n_decimales = len(_RE_DECIMAL.findall(texto))

    print(f"  [Clasificador] keywords_nutri={n_nutri}  "
          f"keywords_precio={n_precio_kw}  decimales={n_decimales}")
    print(f"  [Clasificador] texto_ocr[:120]: {texto[:120]!r}")

    # ── Decisión por keywords nutricionales ───────────────────
    # Cualquier aparición de kcal/proteinas/hidratos es señal fuerte
    if n_nutri >= 2:
        conf = min(0.60 + n_nutri * 0.06, 0.98)
        clf  = ClasificacionImagen(
            TipoImagen.ETIQUETA_NUTRI, round(conf, 3),
            f"{n_nutri} keyword(s) nutricional(es) detectada(s)"
        )
        print(f"  [Clasificador] → ETIQUETA_NUTRI (conf={conf:.0%})")
        return clf, texto, raw

    if n_nutri == 1:
        # Una sola keyword: usar morfología para confirmar o descartar
        n_cnt = _contar_contornos_etiqueta(img_bgr)
        if n_cnt < 3:          # pocos contornos de etiqueta de precio → nutri
            conf = 0.62
            clf  = ClasificacionImagen(
                TipoImagen.ETIQUETA_NUTRI, conf,
                f"1 keyword nutricional + sin contornos de balda (n={n_cnt})"
            )
            print(f"  [Clasificador] → ETIQUETA_NUTRI conf={conf:.0%}")
            return clf, texto, raw
        # Si hay muchos contornos, seguimos al bloque de precios

    # ── Decisión por patrones de precio decimal ────────────────
    if n_decimales >= 2 or kw_precio_ok:
        n_cnt = _contar_contornos_etiqueta(img_bgr)
        conf  = min(0.55 + n_decimales * 0.04 + n_cnt * 0.03, 0.97)
        clf   = ClasificacionImagen(
            TipoImagen.BALDA_PRECIOS, round(conf, 3),
            f"{n_decimales} precio(s) decimal(es), {n_cnt} contorno(s) etiqueta"
        )
        print(f"  [Clasificador] → BALDA_PRECIOS (conf={conf:.0%})")
        return clf, texto, raw

    # ── Fallback morfológico ───────────────────────────────────
    n_cnt = _contar_contornos_etiqueta(img_bgr)
    print(f"  [Clasificador] fallback morfológico: contornos={n_cnt}")

    if n_cnt >= 3:
        clf = ClasificacionImagen(
            TipoImagen.BALDA_PRECIOS, 0.55,
            f"Sin keywords claras, {n_cnt} contorno(s) tipo balda"
        )
        print("  [Clasificador] → BALDA_PRECIOS (fallback morfológico)")
        return clf, texto, raw

    if n_nutri >= 1:
        clf = ClasificacionImagen(
            TipoImagen.ETIQUETA_NUTRI, 0.52,
            "1 keyword nutricional, fallback nutricional"
        )
        print("  [Clasificador] → ETIQUETA_NUTRI (fallback keyword)")
        return clf, texto, raw

    clf = ClasificacionImagen(TipoImagen.DESCONOCIDO, 0.0, "Señales insuficientes")
    print("  [Clasificador] → DESCONOCIDO")
    return clf, texto, raw
