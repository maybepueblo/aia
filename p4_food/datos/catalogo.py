# -*- coding: utf-8 -*-
"""
catalogo.py — Catálogo de alimentos (ETL + acceso a BBDD).

Responsabilidades:
  - Cargar y cachear el CSV del supermercado
  - Generar CSV de demo si no existe
  - Exponer alimentos como ArticuloCompra

En producción, el CSV proviene de:
  - OCR nutricional de etiquetas
  - BEDCA / Open Food Facts / USDA
  - Enriquecimiento organoléptico con agente LLM (se hace UNA vez al añadir)
"""

import os
import json
import pandas as pd
from typing import List, Optional

from ..modelos.contratos import ArticuloCompra


# ── Datos de demo (8 alimentos) ──────────────────────────────────────
_DEMO = [
    {
        "food_id": "P001", "name": "Salmón fresco",
        "kcal": 208, "fat": 13.4, "sat_fat": 3.1,
        "carbs": 0.0, "sugars": 0.0, "fiber": 0.0,
        "protein": 20.4, "salt": 0.1, "allergens": "fish",
        "org": "[0.1,0.5,0.1,0.0,0.8, 0.6,0.0,0.3,0.1,0.1, 0.2,0.2,0.3,0.1,0.2, 0.0,0.5,0.8,0.1]",
    },
    {
        "food_id": "P002", "name": "Arroz blanco",
        "kcal": 370, "fat": 2.7, "sat_fat": 0.6,
        "carbs": 77.0, "sugars": 0.1, "fiber": 3.5,
        "protein": 7.5, "salt": 0.0, "allergens": "",
        "org": "[0.1,0.1,0.0,0.0,0.2, 0.1,0.0,0.1,0.0,0.2, 0.2,0.5,0.1,0.0,0.2, 0.4,0.1,0.1,0.4]",
    },
    {
        "food_id": "P003", "name": "Pechuga de pollo",
        "kcal": 165, "fat": 3.6, "sat_fat": 1.0,
        "carbs": 0.0, "sugars": 0.0, "fiber": 0.0,
        "protein": 31.0, "salt": 0.1, "allergens": "",
        "org": "[0.0,0.3,0.0,0.0,0.4, 0.2,0.0,0.2,0.0,0.1, 0.3,0.5,0.4,0.0,0.3, 0.1,0.2,0.5,0.3]",
    },
    {
        "food_id": "P004", "name": "Brócoli",
        "kcal": 34, "fat": 0.4, "sat_fat": 0.1,
        "carbs": 6.6, "sugars": 1.7, "fiber": 2.6,
        "protein": 2.8, "salt": 0.1, "allergens": "",
        "org": "[0.1,0.1,0.1,0.3,0.2, 0.1,0.0,0.6,0.1,0.5, 0.1,0.6,0.1,0.0,0.5, 0.1,0.1,0.7,0.3]",
    },
    {
        "food_id": "P005", "name": "Nueces peladas",
        "kcal": 654, "fat": 65.2, "sat_fat": 6.1,
        "carbs": 13.7, "sugars": 2.6, "fiber": 6.7,
        "protein": 15.2, "salt": 0.0, "allergens": "nuts",
        "org": "[0.1,0.2,0.1,0.4,0.4, 0.8,0.1,0.1,0.1,0.7, 0.6,0.8,0.1,0.1,0.9, 0.5,0.3,0.1,0.8]",
    },
    {
        "food_id": "P006", "name": "Lentejas cocidas",
        "kcal": 116, "fat": 0.4, "sat_fat": 0.1,
        "carbs": 20.1, "sugars": 1.8, "fiber": 7.9,
        "protein": 9.0, "salt": 0.0, "allergens": "",
        "org": "[0.1,0.1,0.1,0.3,0.3, 0.1,0.1,0.2,0.1,0.7, 0.3,0.4,0.1,0.2,0.2, 0.5,0.2,0.3,0.3]",
    },
    {
        "food_id": "P007", "name": "Aguacate",
        "kcal": 160, "fat": 14.7, "sat_fat": 2.1,
        "carbs": 8.5, "sugars": 0.7, "fiber": 6.7,
        "protein": 2.0, "salt": 0.0, "allergens": "",
        "org": "[0.1,0.1,0.1,0.1,0.2, 0.6,0.0,0.4,0.2,0.3, 0.1,0.1,0.1,0.7,0.1, 0.1,0.9,0.5,0.1]",
    },
    {
        "food_id": "P008", "name": "Huevos",
        "kcal": 155, "fat": 11.0, "sat_fat": 3.3,
        "carbs": 1.1, "sugars": 1.1, "fiber": 0.0,
        "protein": 13.0, "salt": 0.4, "allergens": "eggs",
        "org": "[0.1,0.3,0.0,0.0,0.5, 0.4,0.0,0.1,0.0,0.2, 0.3,0.4,0.2,0.3,0.1, 0.1,0.4,0.4,0.2]",
    },
]


class CatalogoAlimentos:
    """
    Catálogo de alimentos con soporte para búsqueda y listado.
    Carga desde CSV; genera uno de demo si no existe.
    """

    def __init__(self, csv_path: str = "supermercado.csv"):
        if not os.path.exists(csv_path):
            self._crear_csv_demo(csv_path)
        self.df = pd.read_csv(csv_path).fillna({"allergens": ""}).fillna(0.0)
        print(f"Catálogo cargado: {len(self.df)} alimentos desde '{csv_path}'")

    @staticmethod
    def _crear_csv_demo(path: str) -> None:
        """Genera el CSV de demostración si no existe."""
        filas = [
            {
                "food_id": d["food_id"], "name": d["name"],
                "kcal": d["kcal"], "fat": d["fat"], "sat_fat": d["sat_fat"],
                "carbs": d["carbs"], "sugars": d["sugars"], "fiber": d["fiber"],
                "protein": d["protein"], "salt": d["salt"],
                "allergens": d["allergens"], "org_vector": d["org"],
            }
            for d in _DEMO
        ]
        pd.DataFrame(filas).to_csv(path, index=False)
        print(f"CSV de demo generado: '{path}'")

    def _fila_a_articulo(self, fila: pd.Series, cantidad_g: float) -> ArticuloCompra:
        alergenos = [
            a.strip()
            for a in str(fila.get("allergens", "")).split(",")
            if a.strip()
        ]
        try:
            vector = json.loads(str(fila.get("org_vector", "[]")))
        except (json.JSONDecodeError, ValueError):
            vector = [0.5] * 19
        if len(vector) != 19:
            vector = [0.5] * 19

        return ArticuloCompra(
            food_id=str(fila["food_id"]),
            food_name=str(fila["name"]),
            quantity_grams=cantidad_g,
            organoleptic_vector=vector,
            energy_kcal_per_100g=float(fila["kcal"]),
            fat_per_100g=float(fila["fat"]),
            sat_fat_per_100g=float(fila["sat_fat"]),
            carbs_per_100g=float(fila["carbs"]),
            sugars_per_100g=float(fila["sugars"]),
            fiber_per_100g=float(fila["fiber"]),
            protein_per_100g=float(fila["protein"]),
            salt_per_100g=float(fila["salt"]),
            allergens=alergenos,
        )

    def listar_todos(self, cantidad_g: float = 100.0) -> List[ArticuloCompra]:
        """Devuelve todos los alimentos con la cantidad indicada."""
        return [self._fila_a_articulo(fila, cantidad_g) for _, fila in self.df.iterrows()]

    def buscar(self, nombre: str, cantidad_g: float = 100.0) -> Optional[ArticuloCompra]:
        """Búsqueda por nombre (insensible a mayúsculas)."""
        mask = self.df["name"].str.lower().str.contains(nombre.lower())
        if mask.any():
            return self._fila_a_articulo(self.df[mask].iloc[0], cantidad_g)
        return None

    def buscar_por_id(self, food_id: str, cantidad_g: float = 100.0) -> Optional[ArticuloCompra]:
        """Búsqueda por ID exacto."""
        mask = self.df["food_id"] == food_id
        if mask.any():
            return self._fila_a_articulo(self.df[mask].iloc[0], cantidad_g)
        return None
