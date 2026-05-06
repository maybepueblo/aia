# -*- coding: utf-8 -*-
"""
llm_utils.py — Utilidades compartidas para llamadas a LLMs.

  - _parse_json : parser de tres niveles para respuestas LLM
  - llm_call    : reintentos con backoff exponencial (maneja rate-limit 429)
  - inicializar_llms : devuelve (llm_fast, llm_smart) ya configurados
"""

import os
import re
import json
import time
from typing import Tuple

from langchain_groq import ChatGroq


def _parse_json(texto: str) -> dict:
    """
    Parser de tres niveles para respuestas LLM:
      1. Parse directo (mayoría de casos)
      2. Regex no-greedy {.*?} (texto extra antes/después)
      3. Regex greedy   {.*}  (último recurso)
    Limpia bloques markdown ```json``` antes de parsear.
    """
    texto = re.sub(r"```(?:json)?\s*", "", texto).strip().rstrip("`").strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    for patron in [r"\{.*?\}", r"\{.*\}"]:
        m = re.search(patron, texto, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


def llm_call(chain, inputs: dict, reintentos: int = 3, espera: float = 3.0) -> str:
    """
    Llama al chain LangChain con reintentos y backoff exponencial.
    Maneja rate-limit 429 de Groq (~14.400 tok/min en tier gratuito).
    Backoff: 3s → 6s → 12s
    """
    for intento in range(reintentos):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            err = str(e)
            if intento < reintentos - 1:
                pausa = espera * (2 ** intento)
                etiqueta = "Rate limit" if "429" in err or "rate" in err.lower() \
                           else f"Error ({err[:40]})"
                print(f"  ⏳ {etiqueta} — reintentando en {pausa:.0f}s...")
                time.sleep(pausa)
            else:
                raise
    return ""


def inicializar_llms(modelo: str = "llama-3.3-70b-versatile") -> Tuple[ChatGroq, ChatGroq]:
    """
    Inicializa y devuelve (llm_fast, llm_smart).
    La API key se lee de la variable de entorno GROQ_API_KEY.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("⚠️  GROQ_API_KEY no configurada. Exporta la variable de entorno.")

    llm_fast  = ChatGroq(model=modelo, temperature=0, max_tokens=512)
    llm_smart = ChatGroq(model=modelo, temperature=0, max_tokens=1024)
    print(f"✅ LLMs Groq inicializados: {modelo}")
    return llm_fast, llm_smart
