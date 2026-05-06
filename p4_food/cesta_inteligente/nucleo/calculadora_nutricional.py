# -*- coding: utf-8 -*-
"""
calculadora_nutricional.py — Cálculo determinista de objetivos macronutritivos.

Decisión arquitectónica: NO usa LangGraph ni LLM.
Ejecutar aritmética determinista (Mifflin-St Jeor) dentro de un motor de
orquestación de LLMs es sobreingeniería. Una clase pura es O(1),
reproducible y libre de errores LLM.

Fórmula: BMR (Mifflin-St Jeor) × Factor Actividad × Factor Objetivo
"""


class CalculadoraNutricional:
    """
    Calcula objetivos nutricionales SEMANALES a partir del perfil biométrico.

    Fórmula base: Mifflin-St Jeor (BMR) × Factor Actividad × Factor Objetivo
    Distribución de macros según guías nutricionales europeas.
    """

    FACTOR_ACTIVIDAD = {
        "sedentary":   1.200,   # Solo actividad diaria mínima
        "light":       1.375,   # Ejercicio ligero 1-3 días/semana
        "moderate":    1.550,   # Ejercicio moderado 3-5 días/semana
        "active":      1.725,   # Ejercicio intenso 6-7 días/semana
        "very_active": 1.900,   # Atleta o trabajo físico muy intenso
    }
    FACTOR_OBJETIVO = {
        "lose":     0.80,   # Déficit calórico del 20%
        "maintain": 1.00,   # Equilibrio energético
        "gain":     1.12,   # Superávit calórico del 12%
    }

    @classmethod
    def calcular(cls, intencion: dict) -> dict:
        """
        Recibe el dict extraído por el Módulo Cognitivo y devuelve
        un dict de kwargs FuzzyMacro listos para construir UserProfile.
        Todos los valores son SEMANALES.
        """
        peso   = float(intencion.get("weight_kg",     75))
        altura = float(intencion.get("height_cm",    170))
        edad   = float(intencion.get("age",            30))
        sexo   = str(intencion.get("sex",          "male")).lower()
        act    = str(intencion.get("activity_level", "moderate")).lower()
        obj    = str(intencion.get("metabolic_goal", "maintain")).lower()

        # ── BMR (Mifflin-St Jeor) ──────────────────────────────────
        if sexo == "male":
            bmr = 10 * peso + 6.25 * altura - 5 * edad + 5
        else:
            bmr = 10 * peso + 6.25 * altura - 5 * edad - 161

        tdee    = bmr * cls.FACTOR_ACTIVIDAD.get(act, 1.55) * cls.FACTOR_OBJETIVO.get(obj, 1.0)
        semanal = tdee * 7

        # ── Distribución semanal de macros ─────────────────────────
        proteina = peso * (1.9 if obj in ("lose", "gain") else 1.4) * 7
        grasa    = (semanal * 0.28) / 9
        carbos   = max(350, (semanal - proteina * 4 - grasa * 9) / 4)
        fibra    = (30 if sexo == "male" else 25) * 7
        azucar_max = (semanal * 0.10) / 4
        sal_max    = 6 * 7

        def _fuzzy(opt: float, margen: float = 0.20,
                   unidad: str = "g", solo_max: bool = False) -> dict:
            return {
                "min_val":           round(opt * (1 - margen), 1),
                "optimal":           round(opt, 1),
                "max_val":           round(opt * (1 + margen), 1),
                "unit":              unidad,
                "is_max_limit_only": solo_max,
            }

        return {
            "energy_kcal": _fuzzy(semanal,         0.10, "kcal"),
            "fat":         _fuzzy(grasa,            0.20),
            "sat_fat":     _fuzzy(grasa * 0.30,     0.30, solo_max=True),
            "carbs":       _fuzzy(carbos,           0.20),
            "sugars":      _fuzzy(azucar_max,       0.50, solo_max=True),
            "fiber":       _fuzzy(fibra,            0.30),
            "protein":     _fuzzy(proteina,         0.15),
            "salt":        _fuzzy(sal_max,          0.20, solo_max=True),
        }
