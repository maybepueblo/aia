# -*- coding: utf-8 -*-
"""
optimizador.py — Función de coste J y búsqueda evolutiva de cestas (DEAP).

Función de coste:
  J = α · ScoreNutri + (1−α) · ScoreSabor

  ScoreNutri = PRODUCTO(score_macro_i)   → multiplicativo: un macro en 0 destruye todo
  ScoreSabor = similitud coseno(vector_usuario, vector_cesta_ponderado)

Restricción hard: alergia → castigo −1000 (no −inf, para que DEAP mida distancia)

Algoritmo Genético (DEAP):
  - Cromosoma   : lista de N floats (gramos/semana por alimento)
  - Cruce       : cxBlend(alpha=0.5)
  - Mutación    : mutGaussian con sigma=200g + CLIPPING [0, 2500]
  - Selección   : selTournament(tornsize=3)
"""

import random
import time
import warnings
import numpy as np
from typing import List, Tuple

from ..modelos.contratos import MACROS, UserProfile, ArticuloCompra

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from deap import base, creator, tools, algorithms
    _DEAP_DISPONIBLE = True
except ImportError:
    _DEAP_DISPONIBLE = False
    print("⚠️  DEAP no disponible — instala con: pip install deap")


class EvaluadorCesta:
    """
    Calcula la función de coste J para una cesta dada.
    Separado de la búsqueda para poder reutilizarlo en otras partes
    (p.ej. módulo de exploración de combinaciones).
    """

    def __init__(self, perfil: UserProfile):
        self.perfil  = perfil
        self._target = perfil.organoleptic_array()

    def evaluar(self, cesta: List[ArticuloCompra]) -> float:
        """Calcula J. Devuelve −1000 si hay alérgenos excluidos."""
        for articulo in cesta:
            if any(a in self.perfil.allergens_excluded for a in articulo.allergens):
                return -1000.0
        return float(
            self.perfil.alpha       * self._score_nutri(cesta) +
            (1 - self.perfil.alpha) * self._score_sabor(cesta)
        )

    def _score_nutri(self, cesta: List[ArticuloCompra]) -> float:
        """Score nutricional multiplicativo sobre todos los macros."""
        totales = {m: 0.0 for m in MACROS}
        for articulo in cesta:
            for k, v in articulo.macros_escalados().items():
                totales[k] += v
        scores = [getattr(self.perfil, m).score(totales[m]) for m in MACROS]
        return float(np.prod(scores))

    def _score_sabor(self, cesta: List[ArticuloCompra]) -> float:
        """Score de sabor por similitud coseno entre target y cesta ponderada."""
        total_g = sum(max(0.0, a.quantity_grams) for a in cesta)
        if total_g < 1e-8:
            return 0.0
        vector_cesta = sum(
            (max(0.0, a.quantity_grams) / total_g) * np.array(a.organoleptic_vector)
            for a in cesta
        )
        n_t = np.linalg.norm(self._target)
        n_c = np.linalg.norm(vector_cesta)
        return float(np.dot(self._target, vector_cesta) / (n_t * n_c + 1e-8))

    def informe(self, cesta: List[ArticuloCompra]) -> str:
        """Devuelve un informe legible del fitness de la cesta."""
        sn = self._score_nutri(cesta)
        sf = self._score_sabor(cesta)
        j  = self.evaluar(cesta)
        totales = {m: 0.0 for m in MACROS}
        for articulo in cesta:
            for k, v in articulo.macros_escalados().items():
                totales[k] += v

        lineas = [
            f"\n  ┌─── INFORME DE FITNESS ──────────────────────────────────",
            f"  │  J = {j:.4f}   (α={self.perfil.alpha} | Nutri={sn:.3f} | Sabor={sf:.3f})",
            f"  │  Alergenos excluidos: {self.perfil.allergens_excluded or ['ninguno']}",
            f"  ├────────────────────────────────────────────────────────────",
            f"  │  {'MACRO':<12} {'REAL':>8}  {'RANGO / LÍMITE':^20}  SCORE  ESTADO",
            f"  ├────────────────────────────────────────────────────────────",
        ]
        for m in MACROS:
            fz  = getattr(self.perfil, m)
            val = totales[m]
            sc  = fz.score(val)
            icono = "✅" if sc >= 0.90 else ("⚠️ " if sc >= 0.60 else "❌")
            rango = f"MAX {fz.max_val:.0f}" if fz.is_max_limit_only else f"{fz.min_val:.0f} – {fz.max_val:.0f}"
            lineas.append(f"  │  {icono} {m:<11} {val:>7.1f} {fz.unit}  {rango:<20}  {sc:.3f}")
        lineas.append(f"  └────────────────────────────────────────────────────────────")
        return "\n".join(lineas)


class BuscadorCesta:
    """
    Motor de búsqueda evolutiva de cestas óptimas.
    Combina búsqueda aleatoria (baseline) y Algoritmo Genético (DEAP).
    """

    GRAMOS_MIN    = 0.0
    GRAMOS_MAX    = 2500.0
    CORTE_GRAMOS  = 50.0     # Alimentos con < 50g se ignoran en la cesta final

    def __init__(self, perfil: UserProfile, pool_alimentos: List[ArticuloCompra]):
        self.perfil        = perfil
        self.pool_alimentos = pool_alimentos
        self.evaluador     = EvaluadorCesta(perfil)
        print(f"  [BuscadorCesta] Pool: {len(pool_alimentos)} alimentos")

    def _decodificar(self, individuo) -> List[ArticuloCompra]:
        """Convierte cromosoma a lista de ArticuloCompra con clipping."""
        cesta = []
        for articulo, gramos in zip(self.pool_alimentos, individuo):
            g = max(self.GRAMOS_MIN, min(float(gramos), self.GRAMOS_MAX))
            if g >= self.CORTE_GRAMOS:
                cesta.append(articulo.model_copy(update={"quantity_grams": g}))
        return cesta

    def _fitness(self, individuo) -> Tuple[float]:
        """Función de fitness para DEAP (devuelve tupla)."""
        cesta = self._decodificar(individuo)
        if not cesta:
            return (-10.0,)
        return (self.evaluador.evaluar(cesta),)

    def _mutar_con_clip(self, individuo, mu, sigma, indpb) -> Tuple:
        """Mutación gaussiana con CLIPPING post-mutación. Evita gramos negativos."""
        tools.mutGaussian(individuo, mu, sigma, indpb)
        for i in range(len(individuo)):
            individuo[i] = max(self.GRAMOS_MIN, min(individuo[i], self.GRAMOS_MAX))
        return (individuo,)

    def busqueda_aleatoria(self, iteraciones: int = 2000) -> Tuple[list, float]:
        """Búsqueda aleatoria pura — sirve como baseline de comparación."""
        t0 = time.time()
        mejor_j, mejor_ind = -float("inf"), []
        for _ in range(iteraciones):
            ind = [random.uniform(self.GRAMOS_MIN, self.GRAMOS_MAX) for _ in self.pool_alimentos]
            j   = self._fitness(ind)[0]
            if j > mejor_j:
                mejor_j, mejor_ind = j, ind
        print(f"  [Aleatorio] Tiempo: {time.time()-t0:.2f}s | Mejor J: {mejor_j:.4f}")
        return mejor_ind, mejor_j

    def busqueda_genetica(self, tam_poblacion: int = 100, generaciones: int = 50) -> Tuple[list, float]:
        """
        Algoritmo Genético con DEAP.
        HallOfFame es elitista: guarda el mejor individuo absoluto de toda la ejecución.
        """
        if not _DEAP_DISPONIBLE:
            print("  ⚠️  DEAP no instalado — usando búsqueda aleatoria")
            return self.busqueda_aleatoria()

        print(f"\n  [DEAP] Iniciando GA (pop={tam_poblacion}, gen={generaciones})...")
        t0 = time.time()

        # Limpiar clases previas (necesario en entornos interactivos como Colab/Jupyter)
        for attr in ("FitnessMax", "Individual"):
            if hasattr(creator, attr):
                delattr(creator, attr)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        tb = base.Toolbox()
        tb.register("attr_g",     random.uniform, self.GRAMOS_MIN, self.GRAMOS_MAX)
        tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_g, n=len(self.pool_alimentos))
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate",   self._fitness)
        tb.register("mate",       tools.cxBlend, alpha=0.5)
        tb.register("mutate",     self._mutar_con_clip, mu=0, sigma=200, indpb=0.2)
        tb.register("select",     tools.selTournament, tournsize=3)

        poblacion = tb.population(n=tam_poblacion)
        hof       = tools.HallOfFame(1)
        stats     = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        algorithms.eaSimple(
            poblacion, tb, cxpb=0.7, mutpb=0.3, ngen=generaciones,
            stats=stats, halloffame=hof, verbose=True,
        )

        mejor_j = hof[0].fitness.values[0]
        print(f"\n  [DEAP] Tiempo: {time.time()-t0:.2f}s | Mejor J: {mejor_j:.4f}")
        return list(hof[0]), mejor_j

    def decodificar_e_imprimir(self, individuo) -> List[ArticuloCompra]:
        """Decodifica el cromosoma ganador e imprime la lista de la compra."""
        cesta = self._decodificar(individuo)
        print("\n  🛒 CESTA SEMANAL RECOMENDADA:")
        print("  " + "─" * 38)
        for articulo in cesta:
            print(f"  • {articulo.food_name:<22}: {articulo.quantity_grams:>6.0f} g")
        print("  " + "─" * 38)
        return cesta
