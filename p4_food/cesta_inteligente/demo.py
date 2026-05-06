#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo.py — Reproducción de la celda 13 del notebook original.

Ejecuta el pipeline completo con el mismo caso de prueba:
  "Quiero perder algo de tripa... comida japonesa... alergia a frutos secos."

Configura GROQ_API_KEY antes de ejecutar:
    export GROQ_API_KEY="gsk_..."
    python demo.py
"""

import os
import json

# ── API Key ───────────────────────────────────────────────────────────
# En Colab: descomenta las dos líneas siguientes
# from google.colab import userdata
# os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

from cesta_inteligente import Pipeline

TEXTO_USUARIO = """
Quiero perder algo de tripa sin obsesionarme con la dieta.
Voy al gym 3 veces por semana. Me encanta la comida japonesa,
el sushi especialmente. No soporto lo amargo (tipo endivias).
Tengo alergia a los frutos secos.
"""

if __name__ == "__main__":
    pipe = Pipeline(
        csv_catalogo="supermercado.csv",
        # Respuesta demo para el grafo de intención (no requiere input manual)
        respuesta_demo=(
            "Peso 88 kilos, mido 1.82 metros, tengo 34 años, "
            "soy hombre, voy al gym 3 veces y quiero perder tripa."
        ),
    )

    perfil, cesta, menu = pipe.ejecutar(
        texto_usuario = TEXTO_USUARIO,
        user_id       = "demo_001",
        # Datos biométricos exactos de la app (tienen prioridad sobre el LLM)
        peso_kg    = 88.0,
        altura_cm  = 182.0,
        edad       = 34,
        sexo       = "male",
        tam_poblacion = 80,
        generaciones  = 40,
    )

    # ── Resumen del perfil generado ───────────────────────────────────
    print(f"\n  UserProfile generado:")
    print(f"     alpha       = {perfil.alpha}  (0=hedonista | 1=fitness)")
    print(f"     Kcal/sem    = {perfil.energy_kcal.optimal:.0f}"
          f"  [{perfil.energy_kcal.min_val:.0f}–{perfil.energy_kcal.max_val:.0f}]")
    print(f"     Prot/sem    = {perfil.protein.optimal:.0f}g")
    print(f"     Alergenos   = {perfil.allergens_excluded}")

    from cesta_inteligente.modelos.contratos import DIMS
    top5 = sorted(range(len(DIMS)), key=lambda i: perfil.organoleptic_target[i], reverse=True)[:5]
    print(f"     Top5 sabores: "
          + " | ".join(f"{DIMS[i]}={perfil.organoleptic_target[i]:.2f}" for i in top5))

    # ── Exportar UserProfile como JSON (contrato para otros módulos) ──
    print("\n  JSON UserProfile (contrato con el optimizador):")
    print(json.dumps(perfil.model_dump(), indent=2, ensure_ascii=False))
