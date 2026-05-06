
import re
from dataclasses import dataclass


@dataclass
class EjercicioLog:
    nombre:   str
    series:   int
    reps:     int
    peso_kg:  float | None   # None si es bodyweight
    es_bw:    bool
    rir:      int | None     # None si no se especificó
    nota_raw: str            # texto libre para NLP


_RE_LINEA = re.compile(
    r"^[^\w]*"                              # Símbolos iniciales opcionales (*, -, etc)
    r"(?P<nombre>.+?)\s+"                   # Nombre del ejercicio
    r"(?P<series>\d+)\s*[xX]\s*(?P<reps>\d+)" # NxM (soporta espacios intermedios)
    r"(?:"                                  # Peso o BW
        r"\s+(?P<peso>[\d.]+)\s*(?:kg|kilos)?"  # 'kg' ya es opcional (acepta "60" a secas)
        r"|\s+(?P<bw>BW\b[^,]*)"
    r")?"
    r"(?:\s+(?P<rir>\d+)\s*RIR)?"           # RIR opcional
    r"(?:[\s,.;]*(?P<nota>.*))?$",          # Absorbe comas finales y notas libres
    re.IGNORECASE,
)

def parsear_entrenamiento(texto: str) -> list[EjercicioLog]:
    """
    Parsea el texto completo de una sesión y devuelve la lista de ejercicios.
    Las líneas que no empiecen con * se ignoran silenciosamente.
    """
    ejercicios: list[EjercicioLog] = []

    for linea in texto.splitlines():
        m = _RE_LINEA.match(linea.strip())
        if not m:
            continue

        peso_kg: float | None = None
        es_bw = False

        if m.group("bw") is not None:
            es_bw = True
        elif m.group("peso") is not None:
            peso_kg = float(m.group("peso"))

        ejercicios.append(EjercicioLog(
            nombre   = m.group("nombre").strip(),
            series   = int(m.group("series")),
            reps     = int(m.group("reps")),
            peso_kg  = peso_kg,
            es_bw    = es_bw,
            rir      = int(m.group("rir")) if m.group("rir") else None,
            nota_raw = (m.group("nota") or "").strip(),
        ))

    return ejercicios
