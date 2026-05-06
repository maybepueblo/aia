import numpy as np
from metodos_musculos import VectorMusculo, Musculo, cuentaMusculo, CAPACIDAD_MAX

# Tasa de recuperación por músculo en 24h con sleep=diet=1.0
# recuperado[i] = (10 - v_f[i]) * tasa[i] * sleep * diet
_TASAS: dict[Musculo, float] = {
    Musculo.PECTORAL:            0.50,
    Musculo.DELTOIDES_ANTERIOR:  0.50,
    Musculo.DELTOIDES_LATERAL:   0.55,
    Musculo.DELTOIDES_POSTERIOR: 0.55,
    Musculo.TRAPECIO:            0.50,
    Musculo.ESPALDA_ALTA:        0.45,
    Musculo.ESPALDA_BAJA:        0.35,
    Musculo.BICEPS:              0.55,
    Musculo.TRICEPS:             0.55,
    Musculo.ANTEBRAZO:           0.60,
    Musculo.ABDOMEN:             0.55,
    Musculo.CUADRICEPS:          0.38,
    Musculo.ISQUIOTIBIALES:      0.38,
    Musculo.GLUTEOS:             0.42,
    Musculo.GEMELO:              0.60,
    Musculo.SOLEO:               0.60,
}

VEC_TASAS: VectorMusculo = np.array(
    [_TASAS[Musculo(i)] for i in range(cuentaMusculo)],
    dtype=np.float32,
)


def daily_recovery(
    f_actual:    VectorMusculo,
    sleep_score: float = 1.0,
    diet_score:  float = 1.0,
) -> VectorMusculo:
    sleep_score = float(np.clip(sleep_score, 0.0, 1.0))
    diet_score  = float(np.clip(diet_score,  0.0, 1.0))
    recuperado  = (CAPACIDAD_MAX - f_actual) * VEC_TASAS * sleep_score * diet_score
    return np.clip(f_actual + recuperado, 0.0, CAPACIDAD_MAX).astype(np.float32)


def resumen_recuperacion(f_antes: VectorMusculo, f_despues: VectorMusculo) -> None:
    print("Recuperación nocturna (1 día):")
    for m in Musculo:
        ganado = f_despues[m] - f_antes[m]
        if ganado > 1e-4:
            print(f"  {m.name:<22}  {f_antes[m]:.2f} → {f_despues[m]:.2f}  (+{ganado:.2f})")
