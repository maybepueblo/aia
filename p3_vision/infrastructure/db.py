# vision/infrastructure/db.py
# ─────────────────────────────────────────────────────────────
# Persistencia SQLite para productos escaneados y despensa del usuario.
# ─────────────────────────────────────────────────────────────

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from vision.domain.schemas import ResultadoPrecios, ResultadoNutricional


_DDL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS productos (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    supermercado  TEXT    NOT NULL,
    precio_texto  TEXT,
    precio_float  REAL,
    imagen_origen TEXT,
    fecha         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS nutricional (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre_producto TEXT,
    texto_ocr       TEXT,
    energia_kcal    REAL,
    energia_kj      REAL,
    grasas_g        REAL,
    grasas_sat_g    REAL,
    hidratos_g      REAL,
    azucares_g      REAL,
    fibra_g         REAL,
    proteinas_g     REAL,
    sal_g           REAL,
    completitud     REAL,
    imagen_origen   TEXT,
    fecha           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS despensa (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    producto_id INTEGER NOT NULL,
    cantidad    REAL    NOT NULL DEFAULT 1.0,
    unidad      TEXT    NOT NULL DEFAULT 'ud',
    updated_at  TEXT    NOT NULL,
    UNIQUE(user_id, producto_id)
);

CREATE TABLE IF NOT EXISTS escaneos_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tipo         TEXT    NOT NULL,
    imagen       TEXT,
    n_resultados INTEGER,
    supermercado TEXT,
    user_id      INTEGER,
    fecha        TEXT    NOT NULL
);
"""


class SupermercadoDB:

    def __init__(self, db_path: str | Path = "supermercado.db"):
        self.db_path = str(db_path)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)
            conn.commit()

    # ── Guardar precios ───────────────────────────────────────

    def guardar_precios(
        self,
        resultado: ResultadoPrecios,
        user_id: Optional[int] = None,
    ) -> list[int]:
        fecha = datetime.now().isoformat(timespec="seconds")
        ids: list[int] = []
        with self._conn() as conn:
            for p in resultado.precios:
                cur = conn.execute(
                    "INSERT INTO productos (supermercado, precio_texto, precio_float, "
                    "imagen_origen, fecha) VALUES (?, ?, ?, ?, ?)",
                    (resultado.supermercado, p.valor_texto, p.valor_float, None, fecha),
                )
                ids.append(cur.lastrowid)
            conn.execute(
                "INSERT INTO escaneos_log (tipo, n_resultados, supermercado, user_id, fecha) "
                "VALUES (?, ?, ?, ?, ?)",
                ("precios", resultado.n_precios, resultado.supermercado, user_id, fecha),
            )
            conn.commit()
        return ids

    # ── Guardar nutricional ───────────────────────────────────

    def guardar_nutricional(
        self,
        resultado: ResultadoNutricional,
        user_id: Optional[int] = None,
    ) -> int:
        fecha = datetime.now().isoformat(timespec="seconds")
        v     = resultado.valores
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO nutricional (nombre_producto, texto_ocr, energia_kcal, "
                "energia_kj, grasas_g, grasas_sat_g, hidratos_g, azucares_g, fibra_g, "
                "proteinas_g, sal_g, completitud, imagen_origen, fecha) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (resultado.nombre_producto, resultado.texto_ocr,
                 v.energia_kcal, v.energia_kj, v.grasas_g, v.grasas_saturadas_g,
                 v.hidratos_g, v.azucares_g, v.fibra_g, v.proteinas_g, v.sal_g,
                 resultado.valores.completitud(), None, fecha),
            )
            conn.execute(
                "INSERT INTO escaneos_log (tipo, n_resultados, user_id, fecha) "
                "VALUES (?, ?, ?, ?)",
                ("nutricional", 1, user_id, fecha),
            )
            conn.commit()
            return cur.lastrowid

    # ── Consultas ─────────────────────────────────────────────

    def listar_productos(self, supermercado: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM productos"
        p = ()
        if supermercado:
            q += " WHERE supermercado = ?"
            p = (supermercado,)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(q + " ORDER BY fecha DESC", p).fetchall()]

    def listar_nutricional(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM nutricional ORDER BY fecha DESC"
            ).fetchall()]

    # ── Despensa ──────────────────────────────────────────────

    def upsert_despensa(
        self, user_id: int, producto_id: int,
        cantidad: float = 1.0, unidad: str = "ud",
    ) -> None:
        fecha = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO despensa (user_id, producto_id, cantidad, unidad, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, producto_id) DO UPDATE SET "
                "cantidad=excluded.cantidad, unidad=excluded.unidad, updated_at=excluded.updated_at",
                (user_id, producto_id, cantidad, unidad, fecha),
            )
            conn.commit()

    def get_despensa_usuario(self, user_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT d.cantidad, d.unidad, d.updated_at, "
                "       p.supermercado, p.precio_float, "
                "       n.nombre_producto, n.proteinas_g, n.energia_kcal, "
                "       n.hidratos_g, n.grasas_g "
                "FROM despensa d "
                "JOIN productos p ON p.id = d.producto_id "
                "LEFT JOIN nutricional n ON n.rowid = ("
                "  SELECT rowid FROM nutricional ORDER BY fecha DESC LIMIT 1) "
                "WHERE d.user_id = ? ORDER BY d.updated_at DESC",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
