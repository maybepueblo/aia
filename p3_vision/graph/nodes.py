# vision/graph/nodes.py
# ─────────────────────────────────────────────────────────────
# Nodos del grafo LangGraph.
# ─────────────────────────────────────────────────────────────

from vision.domain.schemas import ClasificacionImagen, TipoImagen, VisionState
from vision.pipelines.clasificador import clasificar_imagen
from vision.pipelines.precios import pipeline_precios
from vision.pipelines.nutricional import pipeline_nutricional


# ── Nodos ─────────────────────────────────────────────────────

def nodo_clasificar(state: VisionState) -> dict:
    """
    Ejecuta OCR inicial + clasificación.
    Propaga ocr_inicial_texto y ocr_inicial_raw al estado para
    que nodo_pipeline_nutri los reutilice sin repetir el OCR.
    """
    print("\n[Nodo] Clasificando imagen...")
    try:
        clf, texto, raw = clasificar_imagen(state["img_bgr"])
        print(f"  → tipo={clf.tipo.value}  confianza={clf.confianza:.0%}")
        print(f"  → razón: {clf.razon}")
        return {
            "clasificacion":    clf,
            "ocr_inicial_texto": texto,
            "ocr_inicial_raw":   raw,
        }
    except Exception as e:
        clf = ClasificacionImagen(TipoImagen.DESCONOCIDO, 0.0, str(e))
        return {"clasificacion": clf, "error": str(e)}


def nodo_pipeline_precios(state: VisionState) -> dict:
    print("\n[Nodo] Ejecutando pipeline de PRECIOS...")
    resultado = pipeline_precios(
        img_bgr      = state["img_bgr"],
        supermercado = state.get("supermercado", "Desconocido"),
    )
    print(f"  → {resultado.n_precios} precio(s) detectado(s)")
    return {"resultado_precios": resultado}


def nodo_pipeline_nutri(state: VisionState) -> dict:
    """
    Reutiliza el texto OCR ya extraído durante la clasificación.
    Si por alguna razón no está disponible, hace su propio OCR.
    """
    print("\n[Nodo] Ejecutando pipeline NUTRICIONAL...")
    resultado = pipeline_nutricional(
        img_bgr          = state["img_bgr"],
        nombre_producto  = state.get("nombre_producto", "Producto desconocido"),
        ocr_texto_previo = state.get("ocr_inicial_texto"),
        ocr_raw_previo   = state.get("ocr_inicial_raw"),
    )
    print(f"  → completitud={resultado.valores.completitud():.0%}")
    return {"resultado_nutricional": resultado}


def nodo_agregar(state: VisionState) -> dict:
    """Nodo final: no modifica el estado, el caller lee el resultado."""
    print("\n[Nodo] Agregando resultados...")
    return {}


def nodo_error(state: VisionState) -> dict:
    err = state.get("error", "Error desconocido")
    print(f"\n[Nodo] ERROR: {err}")
    return {}


# ── Función de routing condicional ────────────────────────────

def rutear_por_tipo(state: VisionState) -> str:
    clf: ClasificacionImagen = state.get("clasificacion")
    if clf is None or state.get("error"):
        return "nodo_error"
    if clf.tipo == TipoImagen.BALDA_PRECIOS:
        return "nodo_pipeline_precios"
    if clf.tipo == TipoImagen.ETIQUETA_NUTRI:
        return "nodo_pipeline_nutri"
    return "nodo_error"
