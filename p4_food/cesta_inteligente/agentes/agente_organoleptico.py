# -*- coding: utf-8 -*-
"""
agente_organoleptico.py — Grafo 3: Vectorización Organoléptica (LangGraph).

Diseño: usa LangGraph porque necesita razonamiento semántico:
inferir preferencias de sabor desde pistas culturales
("me gusta el sushi" → umami↑, elastic↑, fresh↑).

Flujo SECUENCIAL en tres etapas:
  1. explicito → Convierte menciones directas (−1..+1 → 0..1)
  2. cultural  → LLM infiere el vector 19D desde hints culturales
  3. fusionar  → Fusiona: EXPLÍCITO tiene PRIORIDAD sobre INFERIDO
"""

import re
import json
from typing import TypedDict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from ..modelos.contratos import DIMS
from ..nucleo.llm_utils import _parse_json, llm_call


# Mapa español → clave de dimensión inglesa
ES_A_DIM = {
    "dulce": "sweet",     "salado": "salty",    "ácido": "sour",
    "amargo": "bitter",   "umami": "umami",     "grasoso": "fatty",
    "picante": "spicy",   "fresco": "fresh",    "frutal": "fruity",
    "terroso": "earthy",  "tostado": "toasty",  "duro": "hard",
    "elástico": "elastic","viscoso": "viscous", "crujiente": "crunchy",
    "granular": "grainy", "cremoso": "creamy",  "jugoso": "juicy",
    "seco": "dry",
}

_DIMS_TEMPLATE = json.dumps({d: 0.5 for d in DIMS})
_DIMS_ESCAPED  = _DIMS_TEMPLATE.replace("{", "{{").replace("}", "}}")

PROMPT_CULTURAL = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un experto en gastronomía mundial. Asigna valores [0.0-1.0] a "
     "dimensiones organolépticas según las preferencias culturales del usuario. "
     "Responde ÚNICAMENTE con el objeto JSON, sin markdown ni explicaciones."),
    ("human",
     "Datos del usuario:\n"
     "- Cocinas/alimentos favoritos: {hints}\n"
     "- Valores ya conocidos (prioridad): {explicit}\n\n"
     "GUÍA DE INFERENCIA cultural:\n"
     "· Japonesa/sushi → umami=0.9, fresh=0.8, elastic=0.7, salty=0.65\n"
     "· Mediterránea   → fresh=0.8, fruity=0.6, earthy=0.6, fatty=0.5\n"
     "· Mexicana       → spicy=0.8, fresh=0.7, salty=0.6, sour=0.5\n"
     "· Italiana       → umami=0.7, toasty=0.6, creamy=0.5, fatty=0.5\n\n"
     "Para las dims en 'valores ya conocidos', úsalos sin cambiarlos.\n"
     f"Devuelve EXACTAMENTE este JSON con valores completados:\n{_DIMS_ESCAPED}"),
])


class EstadoOrganoleptico(TypedDict):
    intencion: dict
    explicito: dict
    cultural:  dict
    vector:    List[float]
    alpha:     float


# ── Nodos ─────────────────────────────────────────────────────────────

def nodo_explicito(estado: EstadoOrganoleptico) -> EstadoOrganoleptico:
    """Convierte flavor_mentions (−1..+1) al espacio [0,1]."""
    explicito = {}
    for mencion, puntuacion in estado["intencion"].get("flavor_mentions", {}).items():
        dim = ES_A_DIM.get(mencion.lower(), mencion.lower())
        if dim in DIMS:
            # -1 → 0.0 | 0 → 0.5 | +1 → 1.0
            explicito[dim] = round(max(0.0, min(1.0, (float(puntuacion) + 1) / 2)), 3)
    estado["explicito"] = explicito
    if explicito:
        print(f"  [G3·explicito] {explicito}")
    return estado


def nodo_cultural(estado: EstadoOrganoleptico, llm_smart) -> EstadoOrganoleptico:
    """LLM infiere el vector 19D desde pistas culturales."""
    hints = estado["intencion"].get("cultural_hints", [])
    if not hints and not estado["explicito"]:
        estado["cultural"] = {d: 0.5 for d in DIMS}
        print("  [G3·cultural] Sin pistas → vector neutro 0.5")
        return estado

    chain  = PROMPT_CULTURAL | llm_smart | StrOutputParser()
    raw    = llm_call(chain, {
        "hints":    hints or ["sin preferencias culturales declaradas"],
        "explicit": estado["explicito"] or "ninguno",
    })
    # Limpiar markdown residual
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

    parsed = _parse_json(raw)
    estado["cultural"] = {
        d: round(max(0.0, min(1.0, float(parsed.get(d, 0.5)))), 3)
        for d in DIMS
    }
    print(f"  [G3·cultural] Inferido OK para hints: {hints}")
    return estado


def nodo_fusionar(estado: EstadoOrganoleptico) -> EstadoOrganoleptico:
    """Fusiona explícito e inferido. EXPLÍCITO tiene prioridad."""
    fusionado = {
        d: estado["explicito"].get(d, estado["cultural"].get(d, 0.5))
        for d in DIMS
    }
    estado["vector"] = [fusionado[d] for d in DIMS]
    estado["alpha"]  = {
        "STRICT":   0.75,
        "BALANCED": 0.50,
        "HEDONIC":  0.25,
    }.get(estado["intencion"].get("alpha_category", "BALANCED"), 0.50)

    top5 = sorted(range(len(DIMS)), key=lambda i: estado["vector"][i], reverse=True)[:5]
    top5_str = " | ".join(f"{DIMS[i][:5]}={estado['vector'][i]:.2f}" for i in top5)
    print(f"  [G3·fusionar] alpha={estado['alpha']} | top5: {top5_str}")
    return estado


# ── Compilador del grafo ──────────────────────────────────────────────

def compilar_grafo_organoleptico(llm_smart):
    """Compila y devuelve el grafo de vectorización organoléptica."""

    def _cultural(e): return nodo_cultural(e, llm_smart)

    g = StateGraph(EstadoOrganoleptico)
    g.add_node("explicito", nodo_explicito)
    g.add_node("cultural",  _cultural)
    g.add_node("fusionar",  nodo_fusionar)
    g.set_entry_point("explicito")
    g.add_edge("explicito", "cultural")
    g.add_edge("cultural",  "fusionar")
    g.add_edge("fusionar",  END)
    return g.compile()
