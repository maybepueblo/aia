# vision/pipelines/precios.py
# ─────────────────────────────────────────────────────────────
# Pipeline de detección de precios en baldas de supermercado.
# ─────────────────────────────────────────────────────────────

import cv2
import numpy as np

from vision.domain.schemas import PrecioDetectado, ResultadoPrecios
from vision.ocr.easyocr_client import ocr_roi_precio
from vision.ocr.parsers import tiene_precio, extraer_precio_texto, parsear_precio_float


# ── Helpers del pipeline ──────────────────────────────────────

def _mascara_adaptativa(img_bgr: np.ndarray, tile_h: int = 150) -> np.ndarray:
    """Umbralización HSV adaptativa por bandas horizontales."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, W = img_bgr.shape[:2]
    mask_total = np.zeros((H, W), dtype=np.uint8)
    for y0 in range(0, H, tile_h):
        y1    = min(y0 + tile_h, H)
        p95   = int(np.percentile(hsv[y0:y1, :, 2], 95))
        v_min = max(170, p95 - 60)
        mask_total[y0:y1] = cv2.inRange(
            hsv[y0:y1, :], np.array([0, 0, v_min]), np.array([180, 55, 255])
        )
    kern1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN,  kern1)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kern2)
    return mask_total


def _densidad_canny(img_bgr: np.ndarray, rect: tuple) -> float:
    centro, size, _ = rect
    ancho, alto = int(size[0]), int(size[1])
    x   = max(0, int(centro[0] - ancho / 2))
    y   = max(0, int(centro[1] - alto / 2))
    roi = img_bgr[y: y + alto, x: x + ancho]
    if roi.size == 0 or ancho < 15 or alto < 10:
        return 0.0
    gris   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)
    return float(np.sum(bordes > 0)) / (ancho * alto)


# ── Pipeline principal ────────────────────────────────────────

def pipeline_precios(
    img_bgr:      np.ndarray,
    supermercado: str = "Desconocido",
) -> ResultadoPrecios:
    """
    Detección de precios en balda de supermercado:
    1. Redimensionar a 1000 px de ancho
    2. Máscara HSV adaptativa por bandas
    3. Filtrar contornos (área + aspect-ratio + densidad Canny)
    4. OCR con EasyOCR sobre cada ROI (corregido de perspectiva, ×2)
    5. Validar precio decimal con regex
    """
    if img_bgr is None or img_bgr.size == 0:
        return ResultadoPrecios(supermercado=supermercado, ok=False, error="Imagen vacía")
    try:
        s    = 1000 / img_bgr.shape[1]
        img  = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = _mascara_adaptativa(img)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis     = img.copy()
        precios = []

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if not (100 <= area <= 8000):
                continue
            rect = cv2.minAreaRect(cnt)
            w, h  = rect[1]
            rw, rh = max(w, h), min(w, h)
            if rh == 0 or not (1.5 <= rw / rh <= 7.5):
                continue
            if _densidad_canny(img, rect) < 0.08:
                continue

            texto      = ocr_roi_precio(img, rect)
            (cx, cy), _, _ = rect
            box        = np.int32(cv2.boxPoints(rect))

            if tiene_precio(texto):
                precio_str = extraer_precio_texto(texto)
                cv2.drawContours(vis, [box], 0, (0, 255, 0), 3)
                cv2.putText(vis, precio_str, (int(cx) - 50, int(cy) + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                precios.append(PrecioDetectado(
                    valor_texto = precio_str,
                    valor_float = parsear_precio_float(precio_str),
                    cx          = int(cx),
                    cy          = int(cy),
                ))
            else:
                cv2.drawContours(vis, [box], 0, (0, 255, 255), 2)

        return ResultadoPrecios(supermercado=supermercado, precios=precios, imagen_anotada=vis)

    except Exception as e:
        return ResultadoPrecios(supermercado=supermercado, ok=False, error=str(e))
