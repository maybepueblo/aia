# vision/pipelines/nutricional.py
# ─────────────────────────────────────────────────────────────
# Pipeline de extracción de información nutricional.
#
# PROBLEMA ORIGINAL — ROI que dejaba fuera grasas, proteínas, fibra, sal
# ──────────────────────────────────────────────────────────────────────
# _detectar_roi_nutricional usaba un kernel horizontal de W//8 px
# (mínimo 40px). En envases con tabla estrecha ese kernel era demasiado
# largo y fallaba. Además, el filtro w > W*0.25 descartaba tablas
# compactas (columna lateral del envase). El resultado: el ROI englobaba
# solo las primeras filas (energía, hidratos) y cortaba las de abajo.
#
# FIX — Detección de ROI en 3 estrategias en cascada
# ───────────────────────────────────────────────────
#  Estrategia 1 — Bounding box extendido de TODOS los contornos de texto
#   · Se obtiene la región con mayor densidad de líneas horizontales
#     usando un kernel adaptativo (entre W//16 y W//4).
#   · Se expande el bbox resultante un 15% hacia abajo para capturar
#     las últimas filas (sal, proteínas, fibra).
#
#  Estrategia 2 — Flood-fill desde zona de texto central
#   · Se busca el centroide del texto OCR inicial y se expande
#     buscando el rectángulo mínimo que englobe toda la región densa.
#
#  Estrategia 3 — Fallback: imagen completa
#   · Si las anteriores fallan o el ROI es demasiado pequeño,
#     se usa la imagen sin recortar.
#
# FLUJO COMPLETO:
#   1. Estrategia ROI en cascada → (x,y,w,h) robusto
#   2. Upscale 2000px + CLAHE LAB + unsharp mask
#   3. OCR hi-res EasyOCR (conf > 0.25)
#   4. Si completitud < 50%: segunda pasada sobre imagen completa
#      y fusión de textos (cubre casos de tabla partida)
#   5. Parseo regex → ValoresNutricionales
# ─────────────────────────────────────────────────────────────

from typing import Optional
import cv2
import numpy as np

from vision.domain.schemas import ResultadoNutricional, ValoresNutricionales
from vision.ocr.easyocr_client import ocr_nutricional_hires
from vision.ocr.parsers import parsear_nutricional


# ── Estrategia 1: densidad de líneas horizontales ────────────

def _roi_por_lineas_h(gray: np.ndarray) -> Optional[tuple]:
    """
    Localiza la tabla nutricional buscando la región con mayor
    densidad de líneas horizontales.

    Kernel adaptativo: entre W//16 y W//4 px de ancho.
    Esto funciona tanto en tablas anchas (ocupan todo el envase)
    como en tablas estrechas (columna lateral).
    """
    H, W = gray.shape

    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    binar = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # Probar múltiples anchos de kernel; quedarse con el que más contornos genera
    mejor_roi    = None
    mejor_area   = 0
    area_img     = H * W

    for factor in [16, 12, 8]:          # kernels: W//16 … W//8
        kw = max(W // factor, 20)
        kern_h     = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        lineas_h   = cv2.morphologyEx(binar, cv2.MORPH_OPEN, kern_h)

        # Dilatar verticalmente para unir filas próximas
        dil_h      = max(H // 8, 30)
        kern_v     = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dil_h))
        tabla_mask = cv2.dilate(lineas_h, kern_v)

        contornos, _ = cv2.findContours(
            tabla_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contornos:
            continue

        # Candidato: mayor contorno que supere el 5% del área
        for cnt in sorted(contornos, key=cv2.contourArea, reverse=True):
            a = cv2.contourArea(cnt)
            if a < area_img * 0.05:
                break
            x, y, w, h = cv2.boundingRect(cnt)
            # Relajar filtro de proporción: tabla puede ser estrecha
            if w > W * 0.15 and h > H * 0.08 and a > mejor_area:
                mejor_area = a
                mejor_roi  = (x, y, w, h)
                break

    if mejor_roi is None:
        return None

    x, y, w, h = mejor_roi

    # ── Extensión asimétrica: +15% hacia abajo, +5% resto ────
    # Las filas de fibra/sal/proteínas siempre están en la parte baja.
    pad_x  = max(int(W * 0.04), 8)
    pad_yu = max(int(H * 0.03), 6)    # arriba
    pad_yd = max(int(H * 0.18), 30)   # abajo (más margen)

    x  = max(0, x - pad_x)
    y  = max(0, y - pad_yu)
    w  = min(W - x, w + 2 * pad_x)
    h  = min(H - y, h + pad_yu + pad_yd)

    return (x, y, w, h)


# ── Estrategia 2: bounding box de texto OCR inicial ──────────

def _roi_por_ocr(img_bgr: np.ndarray, ocr_raw: list) -> Optional[tuple]:
    """
    Calcula el bounding box mínimo que engloba TODOS los fragmentos
    del OCR inicial. Incluye un margen del 8% en todos los lados.
    Solo se usa si la estrategia por líneas falla.
    """
    if not ocr_raw:
        return None

    H, W  = img_bgr.shape[:2]
    xs, ys = [], []

    for bbox, _, conf in ocr_raw:
        if conf < 0.25:
            continue
        for px, py in bbox:
            xs.append(px); ys.append(py)

    if not xs:
        return None

    pad_x = max(int(W * 0.05), 10)
    pad_y = max(int(H * 0.08), 15)    # más margen vertical

    x = max(0, int(min(xs)) - pad_x)
    y = max(0, int(min(ys)) - pad_y)
    w = min(W - x, int(max(xs)) - int(min(xs)) + 2 * pad_x)
    h = min(H - y, int(max(ys)) - int(min(ys)) + 2 * pad_y)

    return (x, y, w, h)


# ── Selector de ROI en cascada ────────────────────────────────

def _detectar_roi(
    img_bgr:  np.ndarray,
    ocr_raw:  Optional[list] = None,
) -> tuple:
    """
    Intenta las estrategias de ROI en orden. Devuelve siempre una tupla
    (x, y, w, h) válida; en el peor caso es toda la imagen.
    """
    H, W  = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    roi = _roi_por_lineas_h(gray)

    # Validar que el ROI cubre al menos el 15% de alto
    if roi is None or roi[3] < H * 0.15:
        if ocr_raw:
            roi = _roi_por_ocr(img_bgr, ocr_raw)

    if roi is None:
        # Fallback: imagen completa
        roi = (0, 0, W, H)
        print("  [ROI] Fallback: imagen completa")
    else:
        print(f"  [ROI] x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")

    return roi


# ── Preprocesado hi-res del ROI ───────────────────────────────

def _preprocesar_roi(roi_bgr: np.ndarray, target: int = 2000) -> np.ndarray:
    """
    Upscale + CLAHE LAB + unsharp mask sobre el ROI recortado.
    Idéntico al del easyocr_client pero aplicado al ROI ya recortado.
    """
    h, w = roi_bgr.shape[:2]
    if max(h, w) < target:
        s   = target / max(h, w)
        img = cv2.resize(roi_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    else:
        img = roi_bgr.copy()

    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img     = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    img  = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

    return img


# ── Pipeline principal ────────────────────────────────────────

def pipeline_nutricional(
    img_bgr:          np.ndarray,
    nombre_producto:  str            = "Producto desconocido",
    ocr_texto_previo: Optional[str]  = None,
    ocr_raw_previo:   Optional[list] = None,
) -> ResultadoNutricional:
    """
    Extracción de macronutrientes de etiqueta de envase.

    Pasos:
      1. Detectar ROI en cascada (líneas horizontales → bbox OCR → fallback).
         El ROI se extiende un 18% hacia abajo para capturar las últimas filas.
      2. Preprocesar ROI (upscale 2000px + CLAHE + unsharp).
      3. OCR hi-res EasyOCR.
      4. Si completitud < 50%: segunda pasada sobre imagen completa.
         Fusión de textos para máxima cobertura.
      5. Parseo regex → ValoresNutricionales.
    """
    if img_bgr is None or img_bgr.size == 0:
        return ResultadoNutricional(
            nombre_producto=nombre_producto, texto_ocr="",
            ok=False, error="Imagen vacía"
        )
    try:
        # 1. ROI
        roi_coords = _detectar_roi(img_bgr, ocr_raw=ocr_raw_previo)
        x, y, w, h = roi_coords
        img_roi    = img_bgr[y:y+h, x:x+w]

        # 2. Preprocesar
        img_proc = _preprocesar_roi(img_roi)

        # 3. OCR hi-res sobre ROI
        texto_roi, ocr_raw = ocr_nutricional_hires(img_proc)
        print(f"  [NutriPipeline] ROI OCR: {len(ocr_raw)} frags, texto_len={len(texto_roi)}")

        # 4. Segunda pasada si la completitud es baja
        valores_roi = parsear_nutricional(texto_roi)
        completitud = valores_roi.completitud()
        print(f"  [NutriPipeline] completitud ROI={completitud:.0%}")

        texto_final = texto_roi
        ocr_raw_final = ocr_raw

        if completitud < 0.50:
            print("  [NutriPipeline] Completitud baja — segunda pasada en imagen completa")
            img_full   = _preprocesar_roi(img_bgr, target=2000)
            texto_full, ocr_raw_full = ocr_nutricional_hires(img_full)
            # Fusión: concatenar ambos textos para que los regex busquen en más contexto
            texto_final   = texto_roi + " " + texto_full
            ocr_raw_final = ocr_raw + ocr_raw_full
            valores_final = parsear_nutricional(texto_final)
            print(f"  [NutriPipeline] completitud final={valores_final.completitud():.0%}")
        else:
            valores_final = valores_roi

        # Si el texto previo del clasificador aporta algo más, fusionarlo también
        if ocr_texto_previo:
            texto_fusionado = texto_final + " " + ocr_texto_previo
            valores_fusionados = parsear_nutricional(texto_fusionado)
            if valores_fusionados.completitud() > valores_final.completitud():
                valores_final = valores_fusionados
                texto_final   = texto_fusionado
                print("  [NutriPipeline] Fusión con texto clasificador mejoró completitud")

        print(f"  [NutriPipeline] completitud={valores_final.completitud():.0%}")

        return ResultadoNutricional(
            nombre_producto = nombre_producto,
            texto_ocr       = texto_final,
            valores         = valores_final,
            img_procesada   = img_proc,
            ocr_raw         = ocr_raw_final,
            roi_coords      = roi_coords,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ResultadoNutricional(
            nombre_producto=nombre_producto, texto_ocr="",
            ok=False, error=str(e)
        )
