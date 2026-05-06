# vision/ocr/parsers.py
# ─────────────────────────────────────────────────────────────
# Parsers de texto OCR → entidades de dominio.
# ─────────────────────────────────────────────────────────────

import re
import unicodedata
from typing import Optional

from vision.domain.schemas import ValoresNutricionales


# ── Normalización ─────────────────────────────────────────────

def _norm(texto: str) -> str:
    """Quita tildes y pasa a minúsculas."""
    n = unicodedata.normalize("NFD", texto)
    return "".join(c for c in n if unicodedata.category(c) != "Mn").lower()


# ── Parsers de precio ─────────────────────────────────────────

_RE_PRECIO = re.compile(r"\d+[.,]\d{1,2}")

def tiene_precio(texto: str) -> bool:
    return bool(_RE_PRECIO.search(texto))

def extraer_precio_texto(texto: str) -> str:
    m = _RE_PRECIO.search(texto)
    return m.group(0) if m else ""

def parsear_precio_float(texto: str) -> Optional[float]:
    try:
        return float(texto.replace(",", "."))
    except (ValueError, AttributeError):
        return None


# ── Keywords para clasificación ───────────────────────────────

_KEYWORDS_NUTRI = {
    "kcal", "kj",
    "proteinas", "proteina",
    "hidratos", "carbono", "carbohidratos",
    "grasas", "grasa", "saturadas", "lipidos",
    "fibra", "nutricional", "nutricion",
    "sal", "sodio", "azucares", "azucar",
    "valor energetico", "energia", "calorias",
    "porcion", "por 100",
}

_KEYWORDS_PRECIO = {
    "precio", "pvp", "oferta", "descuento", "unidad", "pack",
    "ref", "cod", "euros", "eur",
    "mercadona", "carrefour", "lidl", "aldi", "alcampo",
    "hipercor", "eroski", "consum", "dia",
}

def contiene_keywords_nutri(texto: str) -> tuple:
    t = _norm(texto)
    n = sum(1 for kw in _KEYWORDS_NUTRI if kw in t)
    return n > 0, n

def contiene_keywords_precio(texto: str) -> tuple:
    t = _norm(texto)
    n = sum(1 for kw in _KEYWORDS_PRECIO if kw in t)
    return n > 0, n


# ── Parsers de nutrición ──────────────────────────────────────
# REGLAS ANTI-ERROR:
# · \d{1,4}[\.,]\d{1,2}  → exige separador decimal (evita "123" cuando es "12,3")
# · [\.,]?\d{1,4}         → acepta también enteros pequeños sin coma
# · Ambos patrones separados por | dentro de cada grupo capturador
#
# El patrón de captura es: (\d{1,4}[\.,]\d{1,2}|\d{1,3})
#   — máx. 4 dígitos enteros + decimal obligatorio, O entero ≤ 999
#   — evita capturar "1000", "2500" como valor de un macro (serían kJ)

_VAL = r"(\d{1,4}[\.,]\d{1,2}|\d{1,3})"   # número nutricional razonable
_SEP = r"[^0-9]{0,25}?"                    # separador libre corto

_PATRONES_NUTRI = {
    "energia_kcal": [
        rf"energ[ia]a{_SEP}(\d{{2,4}}[\.,]?\d*)\s*kcal",   # kcal puede ser 3-4 dígitos
        rf"(\d{{2,4}}[\.,]?\d*)\s*kcal",
    ],
    "energia_kj": [
        rf"(\d{{3,5}}[\.,]?\d*)\s*kj",                      # kJ: 3-5 dígitos (ej. 1467)
    ],
    "grasas_g": [
        rf"grasas?\s+totales?{_SEP}{_VAL}",
        rf"lipidos{_SEP}{_VAL}",
        rf"^grasas?\s*{_VAL}",
    ],
    "grasas_saturadas_g": [
        rf"saturadas?{_SEP}{_VAL}",
    ],
    "hidratos_g": [
        rf"hidratos?\s+de\s+carbono{_SEP}{_VAL}",
        rf"carbohidratos?{_SEP}{_VAL}",
    ],
    "azucares_g": [
        rf"az[uu]cares?{_SEP}{_VAL}",
    ],
    "fibra_g": [
        rf"fibra{_SEP}{_VAL}",
    ],
    "proteinas_g": [
        rf"prote[ii]nas?{_SEP}{_VAL}",
    ],
    "sal_g": [
        rf"\bsal\b{_SEP}{_VAL}",
        rf"sodio{_SEP}{_VAL}",
    ],
}


def parsear_nutricional(texto_ocr: str) -> ValoresNutricionales:
    """
    Extrae valores nutricionales del texto OCR mediante regex.
    Patrones ajustados para evitar capturas erróneas:
      · Los macros (g) aceptan máx. 3 dígitos o decimal explícito
      · La energía (kcal/kJ) acepta 3-5 dígitos
    """
    t = _norm(texto_ocr)
    campos = {}
    for campo, patrones in _PATRONES_NUTRI.items():
        valor = None
        for patron in patrones:
            m = re.search(patron, t, re.MULTILINE | re.IGNORECASE)
            if m:
                try:
                    valor = float(m.group(1).replace(",", "."))
                except (ValueError, IndexError):
                    pass
                break
        campos[campo] = valor
    return ValoresNutricionales(**campos)
