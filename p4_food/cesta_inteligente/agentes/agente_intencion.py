# -*- coding: utf-8 -*-
"""
agente_intencion.py — Grafo 1: Clasificación de Intención (LangGraph).

Diseño: LangGraph es ideal aquí porque necesitamos un bucle CÍCLICO con
estado persistente. Si el LLM detecta que faltan datos obligatorios,
itera preguntando al usuario hasta completar el perfil (máximo 2 rondas).

Estado del grafo:
  raw_text       → texto libre de entrada
  extracted      → JSON acumulado con campos extraídos
  missing_fields → campos obligatorios aún sin valor
  question/answer → par de la ronda de aclaración activa
  iterations     → número de rondas completadas (máx. 2)

NOTA: En producción, los parámetros biométricos (peso, altura, edad, sexo)
se pasan directamente desde la app, evitando inferencia LLM para datos exactos.
"""

import json
from typing import TypedDict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from ..nucleo.llm_utils import _parse_json, llm_call


CAMPOS_OBLIGATORIOS = [
    "weight_kg", "height_cm", "age", "sex",
    "activity_level", "metabolic_goal",
]


class EstadoIntencion(TypedDict):
    raw_text:        str
    extracted:       dict
    missing_fields:  List[str]
    question:        str
    answer:          str
    iterations:      int


# ── Prompts ───────────────────────────────────────────────────────────

PROMPT_EXTRACCION = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente de nutrición. Tu única tarea es extraer datos "
     "del texto y devolverlos en un objeto JSON válido. "
     "NUNCA añadas explicaciones, markdown ni texto extra."),
    ("human", """Extrae la información y devuelve EXACTAMENTE este JSON
(rellena los campos encontrados, deja null los ausentes):

{{
  "age": null,
  "sex": null,
  "weight_kg": null,
  "height_cm": null,
  "activity_level": null,
  "metabolic_goal": null,
  "alpha_category": "BALANCED",
  "allergens": [],
  "dietary_flags": [],
  "flavor_mentions": {{}},
  "cultural_hints": []
}}

REGLAS DE INFERENCIA:
- sex             : "male" / "female"  (tío→male, mujer→female, chico→male)
- activity_level  : sedentary|light|moderate|active|very_active
                    (oficina→sedentary, gym 3x→active, gym 5x→very_active)
- metabolic_goal  : lose|maintain|gain
                    (perder tripa→lose, ganar músculo→gain)
- alpha_category  : STRICT si quiere cambiar peso; HEDONIC si prioriza sabor;
                    BALANCED si no hay señal clara
- height_cm       : convierte metros (1.82m → 182)
- allergens       : en inglés (nuts, gluten, dairy, fish, shellfish, eggs, soy)
- flavor_mentions : clave en español → intensidad [-1 odio, +1 me encanta]
- cultural_hints  : lista de palabras clave. Ejemplo: ["japonesa", "sushi"]

Texto: {raw_text}""")
])

PROMPT_ACLARACION = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de nutrición amable. Haz preguntas directas y breves."),
    ("human",
     "Necesito estos datos para calcular las necesidades nutricionales: {missing}\n"
     "Formula UNA SOLA pregunta breve sobre el dato MÁS URGENTE. "
     "Solo la pregunta, sin frases introductorias."),
])

PROMPT_INTEGRACION = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un experto actualizando un perfil JSON.\n"
     "REGLA CRÍTICA: MANTÉN INTACTOS TODOS LOS CAMPOS PREVIOS (especialmente "
     "allergens, flavor_mentions y cultural_hints). "
     "Solo rellena los campos que estaban en null con la nueva respuesta.\n"
     "Devuelve ÚNICAMENTE el JSON completo actualizado."),
    ("human",
     "JSON actual: {existing}\n"
     "Pregunta realizada: \"{question}\"\n"
     "Respuesta del usuario: \"{answer}\"\n"
     "Devuelve el JSON sin omitir información anterior."),
])


# ── Nodos del grafo ───────────────────────────────────────────────────

def nodo_extraer(estado: EstadoIntencion, llm_fast) -> EstadoIntencion:
    chain = PROMPT_EXTRACCION | llm_fast | StrOutputParser()
    raw   = llm_call(chain, {"raw_text": estado["raw_text"]})
    extraido = _parse_json(raw)
    if not extraido:
        print("  ⚠️  Parser JSON falló en extracción — usando dict vacío")
        extraido = {}
    estado["extracted"]      = extraido
    estado["missing_fields"] = [f for f in CAMPOS_OBLIGATORIOS if not extraido.get(f)]
    print(f"  [G1·extraer] Faltan: {estado['missing_fields']}")
    return estado


def nodo_preguntar(estado: EstadoIntencion, llm_fast, respuesta_demo: str = None) -> EstadoIntencion:
    """
    En producción: muestra la pregunta al usuario y espera respuesta.
    En demo/test:  usa respuesta_demo si se proporciona.
    """
    chain = PROMPT_ACLARACION | llm_fast | StrOutputParser()
    pregunta = llm_call(chain, {"missing": estado["missing_fields"]})
    estado["question"] = pregunta.strip()
    print(f"\n  Sistema: {estado['question']}")

    if respuesta_demo:
        estado["answer"] = respuesta_demo
        print(f"  Usuario (demo): {estado['answer']}")
    return estado


def nodo_integrar(estado: EstadoIntencion, llm_fast) -> EstadoIntencion:
    chain = PROMPT_INTEGRACION | llm_fast | StrOutputParser()
    raw   = llm_call(chain, {
        "existing": json.dumps(estado["extracted"], ensure_ascii=False),
        "question": estado["question"],
        "answer":   estado["answer"],
    })
    actualizado = _parse_json(raw)

    if actualizado:
        # MERGE SEGURO: evita que la amnesia del LLM machaque datos previos
        for k, v in actualizado.items():
            if v is not None and v != [] and v != {}:
                estado["extracted"][k] = v

    estado["missing_fields"] = [f for f in CAMPOS_OBLIGATORIOS if not estado["extracted"].get(f)]
    estado["iterations"]    += 1
    return estado


def router_intencion(estado: EstadoIntencion) -> str:
    """Continúa el bucle si faltan campos y no se superaron 2 rondas."""
    if estado["missing_fields"] and estado["iterations"] < 2:
        return "preguntar"
    return "listo"


# ── Compilador del grafo ──────────────────────────────────────────────

def compilar_grafo_intencion(llm_fast, respuesta_demo: str = None):
    """
    Compila y devuelve el grafo de clasificación de intención.

    Args:
        llm_fast       : instancia ChatGroq (max_tokens=512)
        respuesta_demo : respuesta automática para testing sin usuario real
    """
    # Cerramos los nodos sobre los LLMs (patrón funcional sin estado global)
    def _extraer(e):  return nodo_extraer(e, llm_fast)
    def _preguntar(e): return nodo_preguntar(e, llm_fast, respuesta_demo)
    def _integrar(e): return nodo_integrar(e, llm_fast)

    g = StateGraph(EstadoIntencion)
    g.add_node("extraer",   _extraer)
    g.add_node("preguntar", _preguntar)
    g.add_node("integrar",  _integrar)
    g.set_entry_point("extraer")
    g.add_conditional_edges("extraer",   router_intencion, {"preguntar": "preguntar", "listo": END})
    g.add_edge("preguntar", "integrar")
    g.add_conditional_edges("integrar",  router_intencion, {"preguntar": "preguntar", "listo": END})

    return g.compile()
