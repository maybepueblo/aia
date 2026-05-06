"""
cesta_inteligente — Sistema de recomendación de compra y recetario.

Arquitectura: LangGraph + Llama 3.3 70B (Groq) + DEAP (Algoritmo Genético)

Uso rápido:
    from cesta_inteligente import Pipeline

    pipe = Pipeline()
    perfil, cesta, menu = pipe.ejecutar(
        texto_usuario = "Quiero perder tripa, voy al gym 3 veces por semana...",
        peso_kg=88, altura_cm=182, edad=34, sexo="male",
    )
"""

from .pipeline import Pipeline
from .modelos.contratos import DIMS, MACROS, FuzzyMacro, UserProfile, ArticuloCompra
