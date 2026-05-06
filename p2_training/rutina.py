import numpy as np
from dataclasses import dataclass, field
from metodos_musculos import VectorMusculo, Musculo, cuentaMusculo, f_wear, _normalizar, CAPACIDAD_MAX

# Parámetros de simulación y scoring
_RIR_SIM        = 3
_PF_SIM         = 1.0
_SERIES_DEFAULT = 3
_REPS_DEFAULT   = 10
BONUS_COMPUESTO = 0.4    # premio por músculo activo (activación ≥ 2/10) → prioriza compuestos
UMBRAL_MINIMO   = 1.5    # parar si algún músculo simulado baja de este valor

# Splits: filtra el léxico por músculo objetivo (argmax) antes de puntuar
_SPLIT: dict[str, set[Musculo]] = {
    "push": {Musculo.PECTORAL, Musculo.DELTOIDES_ANTERIOR, Musculo.DELTOIDES_LATERAL, Musculo.TRICEPS},
    "pull": {Musculo.ESPALDA_ALTA, Musculo.ESPALDA_BAJA, Musculo.DELTOIDES_POSTERIOR,
             Musculo.TRAPECIO, Musculo.BICEPS, Musculo.ANTEBRAZO},
    "legs": {Musculo.CUADRICEPS, Musculo.ISQUIOTIBIALES, Musculo.GLUTEOS, Musculo.GEMELO, Musculo.SOLEO},
    "core": {Musculo.ABDOMEN, Musculo.ESPALDA_BAJA},
    "full": set(Musculo),
}


@dataclass
class EjercicioSugerido:
    nombre:           str
    musculo_objetivo: str
    musculos_top:     list[str]
    series:           int
    reps:             int
    peso_kg:          float | None
    es_bw:            bool
    tiene_marca:      bool
    desgaste_sim:     VectorMusculo = field(repr=False)


def _top_musculos(m_ej: VectorMusculo, n: int = 3) -> list[str]:
    idx = np.argsort(m_ej)[::-1][:n]
    return [Musculo(i).name for i in idx if m_ej[i] > 0]


def _score(m_ej: VectorMusculo, v_f: VectorMusculo) -> float:
    # afinidad con músculos frescos + bonus por ejercicio compuesto
    afinidad  = float(np.dot(m_ej / 10.0, v_f))
    n_activos = int(np.count_nonzero(m_ej >= 2.0))
    return afinidad + BONUS_COMPUESTO * n_activos


def _desgaste_sim(m_ej: VectorMusculo, lexicon_personal: dict, nombre: str, wear_factor: float = 1.0) -> VectorMusculo:
    marca = lexicon_personal.get(_normalizar(nombre), {})
    return f_wear(
        series      = marca.get("series", _SERIES_DEFAULT),
        reps        = marca.get("reps",   _REPS_DEFAULT),
        rir         = _RIR_SIM,
        pf          = _PF_SIM,
        m_ej        = m_ej,
        wear_factor = wear_factor,
    )


def sugerir_rutina(
    f_actual:         VectorMusculo,
    lexico:           dict[str, VectorMusculo],
    lexicon_personal: dict,
    max_ejercicios:   int = 7,
    tipo_sesion:      str = "full",
    wear_factor:      float = 1.0,
) -> list[EjercicioSugerido]:
    # filtrar léxico por split
    musculos_split = _SPLIT.get(tipo_sesion.lower(), set(Musculo))
    lexico_sesion  = lexico if musculos_split == set(Musculo) else {
        n: m for n, m in lexico.items() if Musculo(int(np.argmax(m))) in musculos_split
    }

    f_sim       = f_actual.copy()
    ya_elegidos: set[str] = set()
    rutina:      list[EjercicioSugerido] = []

    for _ in range(max_ejercicios):
        if float(np.min(f_sim)) < UMBRAL_MINIMO:
            break

        candidatos = [
            (_score(m_ej, f_sim), nombre, m_ej)
            for nombre, m_ej in lexico_sesion.items()
            if nombre not in ya_elegidos
        ]
        if not candidatos:
            break
        candidatos.sort(key=lambda x: x[0], reverse=True)

        elegido = None
        for _, nombre, m_ej in candidatos:
            d = _desgaste_sim(m_ej, lexicon_personal, nombre, wear_factor)
            if not np.any(f_sim - d < 0.0):
                elegido = (nombre, m_ej, d)
                break

        if elegido is None:
            break

        nombre_ej, m_ej, d = elegido
        f_sim -= d
        ya_elegidos.add(nombre_ej)

        clave      = _normalizar(nombre_ej)
        marca      = lexicon_personal.get(clave)
        tiene_marca = marca is not None

        rutina.append(EjercicioSugerido(
            nombre           = nombre_ej,
            musculo_objetivo = Musculo(int(np.argmax(m_ej))).name,
            musculos_top     = _top_musculos(m_ej),
            series           = marca["series"]           if tiene_marca else _SERIES_DEFAULT,
            reps             = marca["reps"]             if tiene_marca else _REPS_DEFAULT,
            peso_kg          = marca.get("peso_kg")      if tiene_marca else None,
            es_bw            = marca.get("es_bw", False) if tiene_marca else False,
            tiene_marca      = tiene_marca,
            desgaste_sim     = d,
        ))

    return rutina


def imprimir_rutina(rutina: list[EjercicioSugerido], tipo_sesion: str = "full") -> None:
    label = tipo_sesion.upper() if tipo_sesion != "full" else "COMPLETO"
    print(f"\n{'─'*55}")
    print(f"  Rutina sugerida — {label}  ({len(rutina)} ejercicios)")
    print(f"{'─'*55}")
    for i, ej in enumerate(rutina, 1):
        peso_str  = "BW" if ej.es_bw else (f"{ej.peso_kg}kg" if ej.peso_kg else "—")
        marca_tag = "" if ej.tiene_marca else "  [sin marca]"
        print(f"  {i}. {ej.nombre:<28} {ej.series}×{ej.reps}  {peso_str:<8}{marca_tag}")
        print(f"     Target: {ej.musculo_objetivo}  |  Activa: {', '.join(ej.musculos_top)}")
    print(f"{'─'*55}\n")


def imprimir_estado_fatiga(f_vector: VectorMusculo, titulo: str = "Capacidad restante") -> None:
    print(f"\n{'─'*55}")
    print(f"  {titulo}  (10=fresco, 0=agotado)")
    print(f"{'─'*55}")
    for m in Musculo:
        v   = f_vector[m]
        pct = v / CAPACIDAD_MAX
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        print(f"  {m.name:<22}  {v:4.2f}/10  [{bar}]  {pct*100:5.1f}%")
    print(f"{'─'*55}\n")
