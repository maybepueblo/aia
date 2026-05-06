# -*- coding: utf-8 -*-
"""
pipeline.py — Ensamblador principal del sistema de recomendación.

Orquesta las 4 fases del notebook original:
  FASE 0 — ETL            : CatalogoAlimentos
  FASE 1 — Cognitiva      : Grafo1 (intención) + CalculadoraNutricional + Grafo3 (organoléptico)
  FASE 2 — Optimización   : BuscadorCesta (DEAP)
  FASE 3 — Generativa     : generador_menu (LLM)

Uso típico:
    from cesta_inteligente.pipeline import Pipeline

    pipe = Pipeline()
    perfil, cesta, menu = pipe.ejecutar(
        texto_usuario = "Quiero perder tripa, voy al gym 3 veces...",
        peso_kg=88, altura_cm=182, edad=34, sexo="male",
    )
"""

from typing import Optional, Tuple, List

from .nucleo.llm_utils import inicializar_llms
from .nucleo.calculadora_nutricional import CalculadoraNutricional
from .agentes.agente_intencion import compilar_grafo_intencion
from .agentes.agente_organoleptico import compilar_grafo_organoleptico
from .datos.catalogo import CatalogoAlimentos
from .optimizacion.optimizador import BuscadorCesta, EvaluadorCesta
from .generacion.generador_menu import generar_menu
from .modelos.contratos import FuzzyMacro, UserProfile, ArticuloCompra


class Pipeline:
    """
    Punto de entrada único para el sistema de recomendación.

    Instanciar una sola vez; reutilizar para varios usuarios.
    Los LLMs, catálogo y grafos se inicializan en el constructor.

    Args:
        csv_catalogo   : ruta al CSV del supermercado (se crea demo si no existe)
        modelo_llm     : nombre del modelo Groq a usar
        respuesta_demo : si se proporciona, el grafo de intención NO pregunta
                         al usuario sino que usa este texto como respuesta automática
                         (útil para testing y demo)
    """

    def __init__(
        self,
        csv_catalogo:   str = "supermercado.csv",
        modelo_llm:     str = "llama-3.3-70b-versatile",
        respuesta_demo: Optional[str] = (
            "Peso 88 kilos, mido 1.82 metros, tengo 34 años, "
            "soy hombre, voy al gym 3 veces y quiero perder tripa."
        ),
    ):
        self.llm_fast, self.llm_smart = inicializar_llms(modelo_llm)
        self.catalogo   = CatalogoAlimentos(csv_catalogo)
        self.grafo_int  = compilar_grafo_intencion(self.llm_fast, respuesta_demo)
        self.grafo_org  = compilar_grafo_organoleptico(self.llm_smart)

    # ── FASE 1: construir perfil de usuario ───────────────────────────

    def construir_perfil(
        self,
        texto_usuario: str,
        user_id:    str           = "usuario_001",
        peso_kg:    Optional[float] = None,
        altura_cm:  Optional[float] = None,
        edad:       Optional[int]   = None,
        sexo:       Optional[str]   = None,
    ) -> UserProfile:
        """
        Texto libre → UserProfile.

        Los parámetros biométricos explícitos (peso, altura, edad, sexo)
        tienen PRIORIDAD sobre lo que el LLM infiera del texto.
        Esto evita errores de inferencia y reduce llamadas LLM.
        """
        SEP = "=" * 52

        print(f"\n{SEP}\n  FASE 1A: Clasificación de Intención\n{SEP}")
        resultado = self.grafo_int.invoke({
            "raw_text":       texto_usuario,
            "extracted":      {},
            "missing_fields": [],
            "question":       "",
            "answer":         "",
            "iterations":     0,
        })
        intencion = resultado["extracted"]

        # Parámetros de la app sobreescriben los inferidos por el LLM
        if peso_kg   is not None: intencion["weight_kg"]  = peso_kg
        if altura_cm is not None: intencion["height_cm"]  = altura_cm
        if edad      is not None: intencion["age"]        = edad
        if sexo      is not None: intencion["sex"]        = sexo

        print(f"\n{SEP}\n  FASE 1B: Cálculo Nutricional (determinista)\n{SEP}")
        macros = CalculadoraNutricional.calcular(intencion)
        print(
            f"  [nutri] TDEE estimado: "
            f"{macros['energy_kcal']['optimal']/7:.0f} kcal/día | "
            f"Proteína: {macros['protein']['optimal']:.0f}g/sem"
        )

        print(f"\n{SEP}\n  FASE 1C: Vectorización Organoléptica\n{SEP}")
        organo = self.grafo_org.invoke({
            "intencion": intencion,
            "explicito": {},
            "cultural":  {},
            "vector":    [],
            "alpha":     0.5,
        })

        return UserProfile(
            user_id             = user_id,
            alpha               = organo["alpha"],
            organoleptic_target = organo["vector"],
            allergens_excluded  = intencion.get("allergens",     []),
            dietary_flags       = intencion.get("dietary_flags", []),
            **{k: FuzzyMacro(**v) for k, v in macros.items()},
        )

    # ── FASE 2: optimizar cesta ───────────────────────────────────────

    def optimizar_cesta(
        self,
        perfil:         UserProfile,
        tam_poblacion:  int = 80,
        generaciones:   int = 40,
        con_comparativa: bool = True,
    ) -> Tuple[List[ArticuloCompra], float]:
        """
        UserProfile + Catálogo → Cesta óptima.

        Returns:
            (cesta_final, mejor_j)
        """
        print("\n" + "█" * 58)
        print("  FASE 2 — OPTIMIZACIÓN EVOLUTIVA (DEAP)")
        print("█" * 58)

        pool    = self.catalogo.listar_todos(cantidad_g=100.0)
        buscador = BuscadorCesta(perfil, pool)

        if con_comparativa:
            _, j_aleatorio = buscador.busqueda_aleatoria(iteraciones=1000)
        else:
            j_aleatorio = None

        mejores_genes, mejor_j = buscador.busqueda_genetica(tam_poblacion, generaciones)

        if con_comparativa and j_aleatorio is not None:
            mejora = (mejor_j - j_aleatorio) / max(abs(j_aleatorio), 1e-8) * 100
            print(f"\n  COMPARATIVA:")
            print(f"     Búsqueda aleatoria: J = {j_aleatorio:.4f}")
            print(f"     Algoritmo genético: J = {mejor_j:.4f}   (mejora: +{mejora:.1f}%)")

        cesta = buscador.decodificar_e_imprimir(mejores_genes)
        evaluador = EvaluadorCesta(perfil)
        print(evaluador.informe(cesta))
        return cesta, mejor_j

    # ── FASE 3: generar menú ──────────────────────────────────────────

    def generar_menu(self, cesta: List[ArticuloCompra]) -> str:
        """Cesta óptima → Menú diario con consejos de emplatado."""
        print("\n" + "█" * 58)
        print("  FASE 3 — RECOMENDACIÓN DE MENÚ (LLM Creativo)")
        print("█" * 58)
        return generar_menu(cesta, self.llm_smart)

    # ── Pipeline completo ─────────────────────────────────────────────

    def ejecutar(
        self,
        texto_usuario:  str,
        user_id:        str           = "usuario_001",
        peso_kg:        Optional[float] = None,
        altura_cm:      Optional[float] = None,
        edad:           Optional[int]   = None,
        sexo:           Optional[str]   = None,
        tam_poblacion:  int = 80,
        generaciones:   int = 40,
    ) -> Tuple[UserProfile, List[ArticuloCompra], str]:
        """
        Ejecuta el pipeline completo de las 4 fases.

        Returns:
            (perfil, cesta_final, menu_texto)
        """
        perfil = self.construir_perfil(
            texto_usuario, user_id, peso_kg, altura_cm, edad, sexo
        )
        cesta, _ = self.optimizar_cesta(perfil, tam_poblacion, generaciones)
        menu     = self.generar_menu(cesta)
        return perfil, cesta, menu
