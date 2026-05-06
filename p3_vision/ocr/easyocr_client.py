# vision/ocr/easyocr_client.py
# ─────────────────────────────────────────────────────────────
# Singleton EasyOCR + funciones OCR del módulo.
#
# Tres funciones de lectura:
#   · ocr_inicial()            — imagen cruda, para clasificación.
#   · ocr_nutricional_hires()  — preprocesado CLAHE + upscale 2000px,
#                                para extracción de macros con máxima
#                                resolución. Devuelve (texto, raw).
#   · ocr_roi_precio()         — ROI rotado ×2 para pipeline_precios.
# ─────────────────────────────────────────────────────────────

import cv2
import numpy as np
import easyocr

# ── Singleton ─────────────────────────────────────────────────
print("Inicializando EasyOCR (español)...")
lector_ocr = easyocr.Reader(['es'], gpu=False, verbose=False)
print("✓ EasyOCR listo")


# ── Helpers ───────────────────────────────────────────────────

def _preprocesar_hires(img_bgr: np.ndarray, target: int = 2000) -> np.ndarray:
    """
    Preprocesado para etiquetas nutricionales:
      1. Upscale al lado mayor → target px (INTER_CUBIC)
      2. CLAHE sobre canal L (LAB) para igualar contraste local
      3. Sharpening suave (unsharp mask) para realzar bordes de texto

    Devuelve imagen BGR lista para pasar a EasyOCR como RGB.
    """
    h, w  = img_bgr.shape[:2]
    lado  = max(h, w)
    if lado < target:
        s   = target / lado
        img = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    else:
        img = img_bgr.copy()

    # CLAHE sobre L del espacio LAB
    lab        = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe    = clahe.apply(l)
    img        = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

    # Unsharp mask
    blur       = cv2.GaussianBlur(img, (0, 0), 2.0)
    img        = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return img


# ── OCR inicial (clasificación) ───────────────────────────────

def ocr_inicial(img_bgr: np.ndarray) -> tuple:
    """
    Pasada rápida de EasyOCR sobre la imagen cruda (sin preprocesado).
    Usada exclusivamente por el clasificador.

    Retorna:
        texto : str  — fragmentos con conf > 0.35, minúsculas.
        raw   : list — [(bbox, texto, conf), ...]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    raw     = lector_ocr.readtext(img_rgb)
    texto   = " ".join(r[1].lower() for r in raw if r[2] > 0.35)
    return texto, raw


# ── OCR de alta resolución para etiquetas nutricionales ───────

def ocr_nutricional_hires(img_bgr: np.ndarray,
                           min_conf: float = 0.25) -> tuple:
    """
    OCR sobre imagen ya preprocesada (el preprocesado lo hace nutricional.py).
    Recibe la imagen BGR procesada (upscale + CLAHE + unsharp) y devuelve
    solo (texto, raw) — la imagen procesada ya la tiene el pipeline.

    Retorna:
        texto : str  — todos los fragmentos concatenados.
        raw   : list — [(bbox, texto, conf), ...] con conf > min_conf.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    raw_all = lector_ocr.readtext(img_rgb)
    raw     = [r for r in raw_all if r[2] > min_conf]
    texto   = " ".join(r[1] for r in raw)
    return texto, raw


# ── OCR imagen completa (alias de compatibilidad) ─────────────

def ocr_imagen_completa(img_bgr: np.ndarray) -> list:
    """Alias para compatibilidad con service.py."""
    _, raw = ocr_nutricional_hires(img_bgr)
    return raw


# ── OCR de ROI para precios ───────────────────────────────────

def ocr_roi_precio(img_bgr: np.ndarray, rect: tuple) -> str:
    """
    Extrae texto de un rectángulo mínimo (cv2.minAreaRect).
    Corrige ángulo, recorta, escala ×2 y aplica EasyOCR.
    """
    centro, size, angulo = rect
    ancho, alto = size
    if ancho < alto:
        angulo += 90
        ancho, alto = alto, ancho
    if ancho == 0 or alto == 0:
        return ""

    M      = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotada = cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]),
                            flags=cv2.INTER_CUBIC)
    x       = max(0, int(centro[0] - ancho / 2))
    y       = max(0, int(centro[1] - alto  / 2))
    recorte = rotada[y: y + int(alto), x: x + int(ancho)]
    if recorte.size == 0:
        return ""

    ampliado = cv2.resize(recorte, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img_rgb  = cv2.cvtColor(ampliado, cv2.COLOR_BGR2RGB)
    results  = lector_ocr.readtext(img_rgb, detail=0, paragraph=True)
    return " ".join(results).strip()
