# -*- coding: utf-8 -*-
"""
contratos.py — Estructuras de datos compartidas entre todos los módulos.

Contiene:
  - DIMS: 19 dimensiones organolépticas (orden fijo)
  - MACROS: 8 macronutrientes (etiquetado europeo)
  - FuzzyMacro: rango difuso con penalización lineal
  - UserProfile: contrato entre Módulo Cognitivo y Optimizador
  - ArticuloCompra: unidad mínima del catálogo de alimentos

NOTA: Modificar el orden de DIMS o MACROS rompe compatibilidad con BBDD.
"""

from typing import List
from pydantic import BaseModel, Field
import numpy as np


# ── Dimensiones organolépticas (19D) ─────────────────────────────────
DIMS: List[str] = [
    "sweet",   "salty",   "sour",    "bitter",  "umami",    # Sabor
    "fatty",   "spicy",   "fresh",   "fruity",  "earthy",   # Aroma
    "toasty",  "hard",    "elastic", "viscous", "crunchy",  # Textura
    "grainy",  "creamy",  "juicy",   "dry",
]

# ── Macronutrientes (etiquetado europeo) ─────────────────────────────
MACROS: List[str] = [
    "energy_kcal", "fat", "sat_fat", "carbs",
    "sugars", "fiber", "protein", "salt",
]


class FuzzyMacro(BaseModel):
    """
    Rango difuso con penalización LINEAL (no exponencial).

    Ventaja vs exponencial: el Algoritmo Genético siempre recibe un
    gradiente positivo (guía de dirección). Con exponenciales, cestas
    muy alejadas del óptimo obtienen score ≈ 0 y DEAP no sabe en qué
    dirección mejorar ("desvanecimiento de gradiente").

    is_max_limit_only=True: para azúcares y sal. Estar por debajo
    del óptimo es un éxito nutricional, no un error. Score = 1.0.
    """
    min_val:           float
    optimal:           float
    max_val:           float
    unit:              str  = "g"
    is_max_limit_only: bool = False

    def score(self, value: float) -> float:
        if self.is_max_limit_only and value <= self.optimal:
            return 1.0
        if self.min_val <= value <= self.max_val:
            return 1.0
        if value < self.min_val and not self.is_max_limit_only:
            deviation = (self.min_val - value) / (self.optimal + 1e-8)
        else:
            deviation = (value - self.max_val) / (self.optimal + 1e-8)
        return float(max(0.0, 1.0 - deviation))


class UserProfile(BaseModel):
    """
    Contrato entre el Módulo Cognitivo y el Optimizador.
    Toda la información necesaria para evaluar cualquier cesta.
    """
    user_id:             str
    alpha:               float = Field(ge=0.0, le=1.0)
    # alpha=0 → hedonista (solo sabor) | alpha=1 → fitness (solo nutrición)
    organoleptic_target: List[float]          # Vector 19D [0, 1]
    organoleptic_dims:   List[str] = Field(default_factory=lambda: list(DIMS))
    # Macros SEMANALES (el GA evalúa toda la cesta)
    energy_kcal: FuzzyMacro
    fat:         FuzzyMacro
    sat_fat:     FuzzyMacro
    carbs:       FuzzyMacro
    sugars:      FuzzyMacro
    fiber:       FuzzyMacro
    protein:     FuzzyMacro
    salt:        FuzzyMacro
    allergens_excluded: List[str] = []
    dietary_flags:      List[str] = []

    def organoleptic_array(self) -> np.ndarray:
        return np.array(self.organoleptic_target, dtype=float)


class ArticuloCompra(BaseModel):
    """
    Unidad mínima del catálogo de alimentos.
    El módulo ETL genera esta estructura desde el CSV/BBDD.
    """
    food_id:              str
    food_name:            str
    quantity_grams:       float        # Gramos asignados para la semana
    organoleptic_vector:  List[float]  # Vector 19D [0,1]
    # Macros por 100g (etiquetado europeo)
    energy_kcal_per_100g: float
    fat_per_100g:         float
    sat_fat_per_100g:     float
    carbs_per_100g:       float
    sugars_per_100g:      float
    fiber_per_100g:       float
    protein_per_100g:     float
    salt_per_100g:        float
    allergens: List[str] = []

    def macros_escalados(self) -> dict:
        """Macros totales según los gramos asignados en la cesta."""
        s = self.quantity_grams / 100.0
        return {
            "energy_kcal": self.energy_kcal_per_100g * s,
            "fat":         self.fat_per_100g         * s,
            "sat_fat":     self.sat_fat_per_100g     * s,
            "carbs":       self.carbs_per_100g       * s,
            "sugars":      self.sugars_per_100g      * s,
            "fiber":       self.fiber_per_100g       * s,
            "protein":     self.protein_per_100g     * s,
            "salt":        self.salt_per_100g        * s,
        }
