# demo.py
# ─────────────────────────────────────────────────────────────
# Script de demostración: procesa imágenes con el grafo LangGraph
# y muestra resultados en consola + ventanas matplotlib.
# ─────────────────────────────────────────────────────────────

import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from vision.graph.router import grafo_vision
from vision.domain.schemas import (
    VisionState, TipoImagen, ClasificacionImagen,
    ResultadoPrecios, ResultadoNutricional,
)


# ══════════════════════════════════════════════════════════════
#  Utilidades de anotación
# ══════════════════════════════════════════════════════════════

# Colores por campo nutricional (BGR para OpenCV, hex para matplotlib)
_COLORES_CAMPO = {
    "energia_kcal":       ((0,   200, 255), "#00C8FF"),
    "energia_kj":         ((0,   180, 220), "#00B4DC"),
    "grasas_g":           ((0,    80, 220), "#0050DC"),
    "grasas_saturadas_g": ((0,    50, 180), "#0032B4"),
    "hidratos_g":         ((0,   180,  60), "#00B43C"),
    "azucares_g":         ((0,   140,  40), "#008C28"),
    "fibra_g":            ((180, 120,   0), "#B47800"),
    "proteinas_g":        ((200,  50, 200), "#C832C8"),
    "sal_g":              ((50,   50, 200), "#3232C8"),
}
_COLOR_GENERICO_BGR = (200, 200, 200)
_COLOR_GENERICO_HEX = "#C8C8C8"


def _bbox_a_xywh(bbox) -> tuple:
    """Convierte bbox EasyOCR [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] → (x,y,w,h)."""
    pts = np.array(bbox, dtype=np.float32)
    x, y = pts[:, 0].min(), pts[:, 1].min()
    w = pts[:, 0].max() - x
    h = pts[:, 1].max() - y
    return int(x), int(y), int(w), int(h)


def _escalar_bbox(bbox, sx: float, sy: float):
    """Escala coordenadas de bbox por factores sx, sy."""
    return [[p[0] * sx, p[1] * sy] for p in bbox]


def _campo_de_texto(texto: str, valores) -> tuple:
    """
    Devuelve (nombre_campo, color_bgr, color_hex) si el texto OCR
    corresponde a algún macronutriente ya extraído; si no, genérico.
    """
    t = texto.lower()
    checks = [
        ("energia_kcal",       ["kcal", "energía", "energia", "caloría"]),
        ("energia_kj",         ["kj", "kilojulio"]),
        ("grasas_g",           ["grasa", "lipido", "lípido"]),
        ("grasas_saturadas_g", ["saturad"]),
        ("hidratos_g",         ["hidrato", "carbono", "carbohidrato"]),
        ("azucares_g",         ["azúcar", "azucar"]),
        ("fibra_g",            ["fibra"]),
        ("proteinas_g",        ["proteína", "proteina", "protein"]),
        ("sal_g",              ["sal", "sodio"]),
    ]
    for campo, keywords in checks:
        if any(kw in t for kw in keywords):
            val = getattr(valores, campo, None)
            color_bgr, color_hex = _COLORES_CAMPO.get(
                campo, (_COLOR_GENERICO_BGR, _COLOR_GENERICO_HEX))
            return campo, color_bgr, color_hex
    return None, _COLOR_GENERICO_BGR, _COLOR_GENERICO_HEX


# ══════════════════════════════════════════════════════════════
#  Anotar imagen procesada con todos los bboxes OCR
# ══════════════════════════════════════════════════════════════

def _anotar_ocr_sobre_imagen(img_bgr: np.ndarray,
                              ocr_raw: list,
                              valores) -> np.ndarray:
    """
    Dibuja rectángulos de colores sobre img_bgr para cada fragmento OCR.
    Verde brillante = valor numérico detectado.
    Color por campo = texto de la etiqueta del nutriente.
    Gris = texto genérico/ignorado.
    """
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    for bbox, texto, conf in ocr_raw:
        _, color_bgr, _ = _campo_de_texto(texto, valores)

        # Verde brillante para números puros
        es_numero = bool(__import__("re").search(r"\d[\d.,]*", texto))
        if es_numero and color_bgr == _COLOR_GENERICO_BGR:
            color_bgr = (0, 230, 80)

        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=color_bgr, thickness=2)

        # Etiqueta de texto encima del bbox
        x, y, bw, bh = _bbox_a_xywh(bbox)
        label = f"{texto[:18]} ({conf:.0%})"
        fs    = max(0.35, min(0.55, bw / 180))
        ty    = max(y - 4, 12)
        cv2.putText(vis, label, (x, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color_bgr, 1, cv2.LINE_AA)

    return vis


# ══════════════════════════════════════════════════════════════
#  Visualización NUTRICIONAL — 4 paneles
# ══════════════════════════════════════════════════════════════

def mostrar_resultado_nutricional(estado: VisionState) -> None:
    r: ResultadoNutricional = estado.get("resultado_nutricional")
    if not r:
        print("Sin resultado nutricional")
        return

    print(f"\n{'='*52}")
    print(f"  RESULTADO NUTRICIONAL — {r.nombre_producto}")
    print(f"{'='*52}")
    if not r.ok:
        print(f"  ✗ Error: {r.error}")
        return

    v = r.valores
    tabla_consola = [
        ("Energía",             v.energia_kcal,       "kcal"),
        ("Energía",             v.energia_kj,         "kJ"),
        ("Grasas totales",      v.grasas_g,           "g"),
        ("  saturadas",         v.grasas_saturadas_g, "g"),
        ("Hidratos de carbono", v.hidratos_g,         "g"),
        ("  azúcares",          v.azucares_g,         "g"),
        ("Fibra alimentaria",   v.fibra_g,            "g"),
        ("Proteínas",           v.proteinas_g,        "g"),
        ("Sal",                 v.sal_g,              "g"),
    ]
    for nombre, valor, unidad in tabla_consola:
        mark = "✓" if valor is not None else "—"
        val_str = f"{valor} {unidad}" if valor is not None else "—"
        print(f"  {mark:<2} {nombre:<28} {val_str}")

    completitud = v.completitud()
    barra = "█" * int(completitud * 20) + "░" * (20 - int(completitud * 20))
    print(f"\n  Completitud: [{barra}] {completitud:.0%}")

    # ── Preparar imágenes ─────────────────────────────────────
    img_orig = estado["img_bgr"]

    # Panel 2: imagen original con el ROI recuadrado
    img_con_roi = img_orig.copy()
    if r.roi_coords:
        rx, ry, rw, rh = r.roi_coords
        cv2.rectangle(img_con_roi, (rx, ry), (rx + rw, ry + rh),
                      (0, 220, 255), 3)
        cv2.putText(img_con_roi, "ROI tabla nutricional",
                    (rx, max(ry - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    # Panel 3: imagen procesada hi-res con bboxes OCR anotados
    if r.img_procesada is not None and r.ocr_raw:
        img_anotada = _anotar_ocr_sobre_imagen(r.img_procesada, r.ocr_raw, v)
    elif r.img_procesada is not None:
        img_anotada = r.img_procesada.copy()
    else:
        img_anotada = img_orig.copy()

    # ── Figura 4 paneles ──────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Análisis nutricional — {r.nombre_producto}",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    ax_orig   = fig.add_subplot(gs[0, 0])   # original
    ax_roi    = fig.add_subplot(gs[0, 1])   # original + ROI recuadrado
    ax_proc   = fig.add_subplot(gs[:, 2:])  # imagen procesada anotada (grande)
    ax_macros = fig.add_subplot(gs[1, 0:2]) # gráfico de barras

    # — Panel 1: imagen original —
    ax_orig.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    ax_orig.set_title("① Imagen original", fontsize=10, fontweight="bold")
    ax_orig.axis("off")

    # — Panel 2: original con ROI —
    ax_roi.imshow(cv2.cvtColor(img_con_roi, cv2.COLOR_BGR2RGB))
    ax_roi.set_title("② Área de interés detectada", fontsize=10, fontweight="bold")
    ax_roi.axis("off")

    # — Panel 3: imagen hi-res anotada con bboxes OCR —
    ax_proc.imshow(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB))
    ax_proc.set_title(
        f"③ Detecciones EasyOCR sobre imagen preprocesada"
        f"  ({len(r.ocr_raw)} fragmentos)",
        fontsize=10, fontweight="bold"
    )
    ax_proc.axis("off")

    # Leyenda de colores del panel 3
    leyenda_items = [
        mpatches.Patch(color="#00C8FF", label="Energía (kcal)"),
        mpatches.Patch(color="#0050DC", label="Grasas"),
        mpatches.Patch(color="#00B43C", label="Hidratos"),
        mpatches.Patch(color="#C832C8", label="Proteínas"),
        mpatches.Patch(color="#B47800", label="Fibra"),
        mpatches.Patch(color="#3232C8", label="Sal"),
        mpatches.Patch(color="#00E650", label="Valor numérico"),
        mpatches.Patch(color="#C8C8C8", label="Texto genérico"),
    ]
    ax_proc.legend(handles=leyenda_items, loc="lower right",
                   fontsize=7.5, framealpha=0.85, ncol=2)

    # — Panel 4: barras de macronutrientes —
    macros_plot = {
        "Energía\n(kcal)": v.energia_kcal,
        "Grasas\n(g)":     v.grasas_g,
        "Hidratos\n(g)":   v.hidratos_g,
        "Proteínas\n(g)":  v.proteinas_g,
        "Azúcares\n(g)":   v.azucares_g,
        "Fibra\n(g)":      v.fibra_g,
        "Sal\n(g)":        v.sal_g,
    }
    nombres_m = list(macros_plot.keys())
    valores_m = [macros_plot[k] if macros_plot[k] is not None else 0
                 for k in nombres_m]
    colores_m = ["#e74c3c" if macros_plot[k] is not None else "#cccccc"
                 for k in nombres_m]

    bars = ax_macros.bar(nombres_m, valores_m, color=colores_m,
                         edgecolor="white", linewidth=0.8)
    ax_macros.set_title(
        f"④ Macros por 100g/100ml  —  completitud: {completitud:.0%}",
        fontsize=10, fontweight="bold"
    )
    ax_macros.set_ylabel("Valor")
    ax_macros.tick_params(axis="x", labelsize=8)

    max_val = max(valores_m) if any(v > 0 for v in valores_m) else 1
    for bar, val, nombre in zip(bars, valores_m, nombres_m):
        if macros_plot[nombre] is not None:
            ax_macros.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.015,
                str(macros_plot[nombre]),
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax_macros.legend(
        handles=[mpatches.Patch(color="#e74c3c", label="Detectado"),
                 mpatches.Patch(color="#cccccc", label="No detectado")],
        fontsize=8
    )

    plt.show()

    # Texto OCR bruto en consola
    print(f"\n  Texto OCR bruto ({len(r.texto_ocr)} chars):")
    print(f"  {r.texto_ocr[:400]}{'...' if len(r.texto_ocr) > 400 else ''}")


# ══════════════════════════════════════════════════════════════
#  Visualización de PRECIOS
# ══════════════════════════════════════════════════════════════

def mostrar_resultado_precios(estado: VisionState) -> None:
    r: ResultadoPrecios = estado.get("resultado_precios")
    if not r:
        print("Sin resultado de precios")
        return

    print(f"\n{'='*52}")
    print(f"  RESULTADO PRECIOS — {r.supermercado}")
    print(f"{'='*52}")
    if not r.ok:
        print(f"  ✗ Error: {r.error}")
        return
    if r.n_precios == 0:
        print("  No se detectaron etiquetas de precio.")
    else:
        print(f"  {r.n_precios} precio(s) detectado(s):")
        for p in r.precios:
            print(f"    • {p.valor_texto} €   (pos: {p.cx},{p.cy})")

    if r.imagen_anotada is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Detección de precios — {r.supermercado}",
                     fontsize=14, fontweight="bold")

        img_orig = estado["img_bgr"]
        s = 1000 / img_orig.shape[1]
        axes[0].imshow(cv2.cvtColor(
            cv2.resize(img_orig, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB))
        axes[0].set_title("① Imagen original", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(r.imagen_anotada, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"② Detecciones: {r.n_precios} precio(s)", fontsize=11)
        axes[1].axis("off")
        axes[1].legend(
            handles=[mpatches.Patch(color="lime",   label="Precio detectado"),
                     mpatches.Patch(color="yellow", label="Candidato sin precio")],
            loc="lower right", fontsize=9
        )
        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════
#  Función principal
# ══════════════════════════════════════════════════════════════

def procesar_imagen(
    ruta:            str,
    supermercado:    str = "Mercadona",
    nombre_producto: str = "Producto",
) -> VisionState:
    img = cv2.imread(ruta)
    if img is None:
        print(f"Error: no se pudo cargar '{ruta}'")
        return {}

    print(f"\n{'#'*60}")
    print(f"  Procesando: {ruta}")
    print(f"{'#'*60}")

    estado_inicial: VisionState = {
        "imagen_path":     ruta,
        "img_bgr":         img,
        "supermercado":    supermercado,
        "nombre_producto": nombre_producto,
    }

    t0      = time.time()
    estado  = grafo_vision.invoke(estado_inicial)
    elapsed = time.time() - t0
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")

    clf: ClasificacionImagen = estado.get("clasificacion")
    if clf:
        icono = "🥫" if clf.tipo == TipoImagen.ETIQUETA_NUTRI else "🛒"
        print(f"\n{icono} Clasificación: {clf.tipo.value}  ({clf.confianza:.0%})")
        print(f"   Razón: {clf.razon}")

    if clf and clf.tipo == TipoImagen.BALDA_PRECIOS:
        mostrar_resultado_precios(estado)
    elif clf and clf.tipo == TipoImagen.ETIQUETA_NUTRI:
        mostrar_resultado_nutricional(estado)
    else:
        print("\n⚠  Imagen no clasificada con confianza suficiente.")
        print(f"   Texto OCR: {estado.get('ocr_inicial_texto', '')[:200]!r}")

    return estado


# ── Ejecución ─────────────────────────────────────────────────

if __name__ == "__main__":
    imagenes = sys.argv[1:]

    if not imagenes:
        print("Uso: python demo.py <imagen1> [imagen2] ...")
        print("Probando con imágenes de ejemplo...\n")
        ejemplos = [
            ("images/imagensoja.jpg", "Mercadona", "Bebida de soja"),
            ("images/imagen1.jpeg",   "Mercadona", "Producto"),
            ("images/imagen2.jpeg",   "Mercadona", "Producto"),
        ]
        for ruta, super_, prod in ejemplos:
            try:
                procesar_imagen(ruta, super_, prod)
            except Exception as e:
                print(f"  [{ruta}] Error: {e}")
    else:
        for ruta in imagenes:
            procesar_imagen(ruta)
