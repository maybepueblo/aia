from enum import IntEnum
import numpy as np
from numpy.typing import NDArray
import unicodedata
import re
import json
from pathlib import Path
from datetime import date

VectorMusculo = NDArray[np.float32]

# Convenio: v_f[i] ∈ [0, 10] = capacidad RESTANTE  (10=fresco, 0=agotado)
CAPACIDAD_MAX = 10.0


class Musculo(IntEnum):
    # Pecho
    PECTORAL                  = 0
    # Hombros
    DELTOIDES_ANTERIOR        = 1
    DELTOIDES_LATERAL         = 2
    DELTOIDES_POSTERIOR       = 3
    # Espalda
    TRAPECIO                  = 4
    ESPALDA_ALTA              = 5
    ESPALDA_BAJA              = 6
    # Brazo
    BICEPS                    = 7
    TRICEPS                   = 8
    ANTEBRAZO                 = 9
    # Core
    ABDOMEN                   = 10
    # Pierna
    CUADRICEPS                = 11
    ISQUIOTIBIALES            = 12
    GLUTEOS                   = 13
    GEMELO                    = 14
    SOLEO                     = 15

cuentaMusculo = len(Musculo)


def make_muscle_vector(values: dict[Musculo, float]) -> VectorMusculo:
    vec = np.full(cuentaMusculo, CAPACIDAD_MAX, dtype=np.float32)
    for musculo, nivel in values.items():
        if not 0.0 <= nivel <= CAPACIDAD_MAX:
            raise ValueError(f"{musculo.name}: nivel {nivel} fuera de rango")
        vec[musculo] = nivel
    return vec


class User:
    # exp_wear_factor: principiante (exp=0) se fatiga más con misma carga que avanzado (exp=5)
    _EXP_WEAR: dict[int, float] = {0: 1.30, 1: 1.18, 2: 1.08, 3: 1.00, 4: 0.92, 5: 0.80}

    def __init__(self, edad: int, experiencia: int, masa_magra: float):
        self.edad        = edad
        self.experiencia = min(max(experiencia, 0), max(self._EXP_WEAR.keys()))
        self.masa_magra  = masa_magra

    @property
    def exp_wear_factor(self) -> float:
        return self._EXP_WEAR.get(self.experiencia, 1.00)


# rango útil frente al vector [0-10]
ESCALA_GLOBAL = 0.05


def cargar_lexico(ruta: str | Path) -> dict[str, VectorMusculo]:
    """Lee lexico_ejercicios.json → { nombre_normalizado → array float32[16] }"""
    with open(ruta, encoding="utf-8") as f:
        raw: dict = json.load(f)

    lexico = {}
    for nombre, activaciones in raw.items():
        vector = np.zeros(cuentaMusculo, dtype=np.float32)
        for musculo, valor in activaciones.items():
            try:
                idx = Musculo[musculo]
            except KeyError:
                raise ValueError(f"Músculo desconocido '{musculo}' en '{nombre}'")
            vector[idx] = float(valor)
        lexico[_normalizar(nombre)] = vector

    return lexico


def _normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, espacios simples."""
    sin_tildes = unicodedata.normalize("NFD", texto)
    sin_tildes = "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", sin_tildes.strip().lower())


def buscar_ejercicio(nombre: str, lexico: dict[str, VectorMusculo]) -> VectorMusculo | None:
    """Búsqueda exacta primero, luego por substring."""
    clave = _normalizar(nombre)
    if clave in lexico:
        return lexico[clave]
    for clave_lexico, vector in lexico.items():
        if clave in clave_lexico or clave_lexico in clave:
            return vector
    return None


def f_wear(
    series:      int,
    reps:        int,
    rir:         int | None,
    pf:          float,
    m_ej:        VectorMusculo,
    wear_factor: float = 1.0,
) -> VectorMusculo:
    # D = (I · vol · P_f · wear_factor) · M_ej_normalizado · ESCALA
    I   = 1.0 - (rir / 10.0) if rir is not None else 1.0
    vol = series * reps
    D   = (I * vol * pf * wear_factor * ESCALA_GLOBAL) * (m_ej / 10.0)
    return D.astype(np.float32)


def f_apply(
    f_prev: VectorMusculo,
    d:      VectorMusculo,
) -> tuple[VectorMusculo, list[str]]:
    # f_nueva[i] = clip(f_prev[i] - D[i], 0, 10)
    # alerta si algún músculo llegaría a 0 (sobreentrenamiento)
    f_raw   = f_prev - d
    alertas = [Musculo(i).name for i in range(cuentaMusculo) if f_raw[i] < 0.0]
    f_nueva = np.clip(f_raw, 0.0, CAPACIDAD_MAX)
    return f_nueva.astype(np.float32), alertas


def procesar_sesion(
    ejercicios:            list[dict],
    lexico:                dict[str, VectorMusculo],
    f_prev:                VectorMusculo,
    usuario:               User | None = None,
    ruta_lexicon_personal: str | Path | None = None,
) -> tuple[VectorMusculo, list[str], list[str]]:
    d_total        = np.zeros(cuentaMusculo, dtype=np.float32)
    no_encontrados = []
    wear_factor    = usuario.exp_wear_factor if usuario is not None else 1.0

    for ej in ejercicios:
        m_ej = buscar_ejercicio(ej["nombre"], lexico)
        if m_ej is None:
            no_encontrados.append(ej["nombre"])
            continue
        d_total += f_wear(
            series      = ej["series"],
            reps        = ej["reps"],
            rir         = ej.get("rir"),
            pf          = ej.get("pf", 1.0),
            m_ej        = m_ej,
            wear_factor = wear_factor,
        )

    f_nueva, alertas = f_apply(f_prev, d_total)

    if ruta_lexicon_personal is not None:
        actualizar_lexicon_personal(ejercicios, ruta_lexicon_personal)

    return f_nueva, alertas, no_encontrados


def cargar_lexicon_personal(ruta: str | Path) -> dict:
    ruta = Path(ruta)
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def guardar_lexicon_personal(lexicon: dict, ruta: str | Path) -> None:
    with open(ruta, encoding="utf-8", mode="w") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)


def _es_mejor_marca(nuevo: dict, actual: dict) -> bool:
    if nuevo["es_bw"]:
        return (nuevo["series"] * nuevo["reps"]) > (actual["series"] * actual["reps"])
    peso_nuevo  = nuevo.get("peso_kg") or 0.0
    peso_actual = actual.get("peso_kg") or 0.0
    if peso_nuevo != peso_actual:
        return peso_nuevo > peso_actual
    return (nuevo["series"] * nuevo["reps"]) > (actual["series"] * actual["reps"])


def actualizar_lexicon_personal(
    ejercicios: list[dict],
    ruta:       str | Path,
) -> dict:
    lexicon = cargar_lexicon_personal(ruta)
    for ej in ejercicios:
        clave = _normalizar(ej["nombre"])
        nuevo = {
            "series":     ej["series"],
            "reps":       ej["reps"],
            "peso_kg":    ej.get("peso_kg"),
            "es_bw":      ej.get("es_bw", False),
            "updated_at": date.today().isoformat(),
        }
        if clave not in lexicon or _es_mejor_marca(nuevo, lexicon[clave]):
            lexicon[clave] = nuevo
    guardar_lexicon_personal(lexicon, ruta)
    return lexicon
