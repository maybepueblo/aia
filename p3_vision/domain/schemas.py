# vision/domain/schemas.py
# ─────────────────────────────────────────────────────────────
# Estructuras de datos del módulo de visión.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from typing_extensions import TypedDict


# ── Tipos de imagen ───────────────────────────────────────────

class TipoImagen(str, Enum):
    BALDA_PRECIOS  = "balda_precios"    # lineal de supermercado → extraer precios
    ETIQUETA_NUTRI = "etiqueta_nutri"   # envase con tabla nutricional → extraer macros
    DESCONOCIDO    = "desconocido"      # no se pudo clasificar con confianza suficiente


# ── Clasificación ─────────────────────────────────────────────

@dataclass
class ClasificacionImagen:
    tipo:       TipoImagen
    confianza:  float
    razon:      str


# ── Precios ───────────────────────────────────────────────────

@dataclass
class PrecioDetectado:
    valor_texto: str
    valor_float: Optional[float]
    cx: int
    cy: int


@dataclass
class ResultadoPrecios:
    supermercado:   str
    precios:        list = field(default_factory=list)       # list[PrecioDetectado]
    imagen_anotada: Optional[np.ndarray] = field(default=None, repr=False)
    ok:             bool = True
    error:          Optional[str] = None

    @property
    def n_precios(self) -> int:
        return len(self.precios)


# ── Nutricional ───────────────────────────────────────────────

@dataclass
class ValoresNutricionales:
    energia_kcal:       Optional[float] = None
    energia_kj:         Optional[float] = None
    grasas_g:           Optional[float] = None
    grasas_saturadas_g: Optional[float] = None
    hidratos_g:         Optional[float] = None
    azucares_g:         Optional[float] = None
    fibra_g:            Optional[float] = None
    proteinas_g:        Optional[float] = None
    sal_g:              Optional[float] = None

    def completitud(self) -> float:
        total     = len(self.__dataclass_fields__)
        detectados = sum(1 for v in self.__dict__.values() if v is not None)
        return detectados / total if total else 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ResultadoNutricional:
    nombre_producto:  str
    texto_ocr:        str
    valores:          ValoresNutricionales = field(default_factory=ValoresNutricionales)
    ok:               bool = True
    error:            Optional[str] = None
    # Campos de visualización — populados por pipeline_nutricional
    img_procesada:    Optional[np.ndarray] = field(default=None, repr=False)
    ocr_raw:          list = field(default_factory=list)   # [(bbox, texto, conf), ...]
    roi_coords:       Optional[tuple] = None               # (x, y, w, h) del área de interés


# ── Estado LangGraph (TypedDict) ──────────────────────────────

class VisionState(TypedDict, total=False):
    imagen_path:           str
    img_bgr:               np.ndarray
    supermercado:          str
    nombre_producto:       str
    # Texto y raw de la pasada OCR inicial (reutilizado por pipeline_nutricional)
    ocr_inicial_texto:     str
    ocr_inicial_raw:       list
    clasificacion:         ClasificacionImagen
    resultado_precios:     ResultadoPrecios
    resultado_nutricional: ResultadoNutricional
    error:                 Optional[str]
