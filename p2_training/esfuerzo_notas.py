from pysentimiento import create_analyzer # https://github.com/pysentimiento/pysentimiento
 
# constantes
MODIFICADOR_NEUTRO  = 1.0   # sin información de esfuerzo
MODIFICADOR_RANGO   = 0.3   # amplitud máxima desde el neutro (→ 0.7 o 1.3)
 
# Dirección según polaridad:
#   POS → menos fatiga → modificador < 1.0 → dirección -1
#   NEG → más fatiga   → modificador > 1.0 → dirección +1
#   NEU → sin cambio   → modificador = 1.0 → dirección  0
_DIRECCION = {
    "POS": -1,
    "NEG": +1,
    "NEU":  0, 
}
 
 
def cargar_analizador():
    """
    cargar el modelo de análisis de sentimiento en español (pysentimiento) y devolver el analizador listo para usar.
    """
    return create_analyzer(task="sentiment", lang="es")
 
 
# analizar nota 
def analizar_nota(nota: str, analizador) -> float:
    """
    Recibe una nota y devuelve un modificador float [0.7 – 1.3].
 
    Lógica:
        modificador = 1.0 + dirección × confianza × MODIFICADOR_RANGO
 
    Ejemplos:
        "El peso se movía solo"     → POS, confianza alta  → ~0.7
        "Me costó más de lo normal" → NEG, confianza alta  → ~1.3
        "Serie completada"          → NEU                  → 1.0
    """
    if not nota or not nota.strip():
        return MODIFICADOR_NEUTRO
 
    resultado   = analizador.predict(nota)
    etiqueta    = resultado.output          # "POS", "NEG" o "NEU"
    confianza   = resultado.probas[etiqueta]
    direccion   = _DIRECCION.get(etiqueta, 0)
 
    modificador = MODIFICADOR_NEUTRO + direccion * confianza * MODIFICADOR_RANGO
 
    # Garantizar rango [0.7, 1.3] ante valores extremos del modelo
    return round(max(0.7, min(1.3, modificador)), 4)
 
 
# analizar sesión completa
def notas_a_modificadores(notas: list[str], analizador) -> list[float]:
    """
    lista de modificadores en mismo orden de entrada, aplicando analizar_nota a cada nota_raw
    """
    return [analizar_nota(nota, analizador) for nota in notas]