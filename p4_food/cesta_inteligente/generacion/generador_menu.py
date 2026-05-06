# -*- coding: utf-8 -*-
"""
generador_menu.py — Fase 3: Generación de menú diario con LLM creativo.

El LLM toma la cesta matemáticamente óptima y la transforma en un menú
diario concreto con consejos de emplatado.
"""

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..modelos.contratos import ArticuloCompra
from ..nucleo.llm_utils import llm_call


PROMPT_MENU = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un chef profesional especializado en nutrición deportiva. "
     "Diseña un menú para UN día (Comida + Cena) usando los ingredientes dados. "
     "Las cantidades son para toda la semana; sugiere raciones lógicas para 1 día. "
     "Incluye OBLIGATORIAMENTE consejos de emplatado para cada plato. "
     "Responde en español con formato Markdown, siendo conciso pero creativo."),
    ("human",
     "Lista de la compra semanal optimizada matemáticamente:\n\n"
     "{cesta_str}\n\n"
     "Genera un menú de ejemplo para un día con instrucciones de emplatado."),
])


def generar_menu(cesta: List[ArticuloCompra], llm_smart) -> str:
    """
    Genera un menú diario sugerido a partir de la cesta óptima.

    Args:
        cesta     : lista de ArticuloCompra con cantidades semanales
        llm_smart : instancia ChatGroq (max_tokens=1024)

    Returns:
        Texto en Markdown con el menú y consejos de emplatado.
    """
    if not cesta:
        return "⚠️  Cesta vacía: ejecuta primero el algoritmo genético."

    cesta_str = "\n".join(
        f"- {a.food_name} ({a.quantity_grams:.0f}g/semana)"
        for a in cesta
    )
    chain    = PROMPT_MENU | llm_smart | StrOutputParser()
    respuesta = llm_call(chain, {"cesta_str": cesta_str})

    print("\n  " + "═" * 56)
    print("  🍽️  MENÚ DIARIO SUGERIDO Y CONSEJOS DE EMPLATADO")
    print("  " + "═" * 56)
    print(respuesta)
    return respuesta
