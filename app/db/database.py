"""
app/db/database.py  ·  WellnessAI — Capa de persistencia completa
═══════════════════════════════════════════════════════════════════

Dos ficheros SQLite independientes — separación estricta de dominios:

  users.db  — Árbol del usuario (identidad + perfil + biometría + historial)
  ┌─ users                   identidad, hash del nombre → O(log n)
  ├─ face_embeddings          256-D cifrado AES-256-GCM
  ├─ user_profile             características personales fijas
  ├─ habitos_diarios          18 hábitos, 1 fila por día
  ├─ hist_bienestar           scores físico/mental diarios
  ├─ marcas_entreno           35 ejercicios (1RM / marca personal)
  ├─ preferencia_organoleptica 19-D vector de gustos alimentarios
  ├─ despensa                 inventario de alimentos del usuario
  └─ log_cambios              auditoría de cualquier mutación de perfil

  foods.db  — Catálogo global de alimentos (crece con cada escaneo P3)
  ├─ alimentos                datos maestros + v_nutricional + v_organoleptico
  ├─ escaneos_nutricionales   trazabilidad de cada escaneo OCR
  └─ precios_supermercado     histórico de precios por cadena

Estrategia de búsqueda O(log n):
  username_hash = SHA-256(nombre.lower().strip())
  → columna TEXT UNIQUE con índice B-Tree implícito en SQLite.
  La búsqueda por hash es O(log n) sin exponer el nombre real.

Cifrado biométrico:
  embedding (256-D float32 = 1024 bytes)
  → AES-256-GCM (clave = SHA-256(SECRET_KEY))
  → almacenado como BLOB
  El hash SHA-256 + salt permite verificar integridad sin descifrar.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import numpy as np

# ── Rutas ──────────────────────────────────────────────────────────────────────
_DB_DIR = Path(__file__).parent / "data"
USERS_DB_PATH = _DB_DIR / "users.db"
FOODS_DB_PATH = _DB_DIR / "foods.db"
_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Clave de cifrado AES-256 (derivada de SECRET_KEY) ─────────────────────────
_SECRET = os.getenv("SECRET_KEY", "wellnessai-dev-secret-32bytes!!")
_AES_KEY: bytes = hashlib.sha256(_SECRET.encode()).digest()   # 32 bytes


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades de seguridad
# ══════════════════════════════════════════════════════════════════════════════

def hash_username(nombre: str) -> str:
    """SHA-256 del nombre normalizado → clave de búsqueda O(log n)."""
    return hashlib.sha256(nombre.lower().strip().encode()).hexdigest()


def _encrypt_embedding(emb: np.ndarray) -> tuple[bytes, bytes]:
    """
    AES-256-GCM sobre el embedding float32[256] (1024 bytes).
    Devuelve (ciphertext_with_tag, nonce_12bytes).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    raw   = emb.astype(np.float32).tobytes()          # 256 * 4 = 1024 bytes
    nonce = os.urandom(12)
    ct    = AESGCM(_AES_KEY).encrypt(nonce, raw, None) # ct incluye tag GCM
    return ct, nonce


def _decrypt_embedding(ct: bytes, nonce: bytes) -> np.ndarray:
    """Descifra y devuelve np.ndarray float32[256]."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    raw = AESGCM(_AES_KEY).decrypt(nonce, ct, None)
    return np.frombuffer(raw, dtype=np.float32)


def _hash_embedding(emb: np.ndarray) -> tuple[str, str]:
    """SHA-256 + salt del embedding para verificación de integridad."""
    salt    = os.urandom(32).hex()
    payload = salt + emb.astype(np.float32).tobytes().hex()
    digest  = hashlib.sha256(payload.encode()).hexdigest()
    return digest, salt


def _verify_embedding(emb: np.ndarray, stored_hash: str, salt: str) -> bool:
    """Verifica integridad en tiempo constante."""
    import hmac  # Importamos la librería correcta aquí
    payload = salt + emb.astype(np.float32).tobytes().hex()
    digest  = hashlib.sha256(payload.encode()).hexdigest()
    return hmac.compare_digest(digest, stored_hash)


# ══════════════════════════════════════════════════════════════════════════════
# DDL — users.db
# ══════════════════════════════════════════════════════════════════════════════

_DDL_USERS = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode  = WAL;

-- ── Tabla raíz: identidad del usuario ────────────────────────────────────────
-- username_hash: SHA-256(nombre.lower()) → índice B-Tree → búsqueda O(log n).
-- El nombre real NUNCA se almacena (privacidad por diseño).
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username_hash   TEXT    NOT NULL UNIQUE,     -- SHA-256 del nombre → O(log n)
    display_name    TEXT,                        -- nombre mostrado (opcional)
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    last_login      TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_hash ON users(username_hash);  -- B-Tree explícito

-- ── FaceEmbedding 256-D cifrado ────────────────────────────────────────────
-- embedding_blob : AES-256-GCM(float32[256]) → BLOB (1024 + 16 bytes de tag)
-- nonce          : 12 bytes aleatorios de AESGCM
-- emb_hash/salt  : SHA-256 para verificar integridad sin descifrar
CREATE TABLE IF NOT EXISTS face_embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding_blob  BLOB    NOT NULL,    -- AES-256-GCM(float32[256])
    nonce           BLOB    NOT NULL,    -- 12 bytes GCM nonce
    emb_hash        TEXT    NOT NULL,    -- SHA-256 del embedding en claro
    salt            TEXT    NOT NULL,    -- salt del hash
    dim             INTEGER NOT NULL DEFAULT 256,
    liveness_score  REAL,               -- deepface confidence
    quality_score   REAL,               -- calidad de la imagen
    version         INTEGER NOT NULL DEFAULT 1,
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_face_user ON face_embeddings(user_id, is_active);

-- ── Perfil personal: características físicas fijas ────────────────────────
-- 1 fila por usuario (UNIQUE user_id). Se actualiza con upsert.
CREATE TABLE IF NOT EXISTS user_profile (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    -- Características personales
    edad            REAL,
    genero          TEXT,               -- 'hombre'|'mujer'|'otro'
    objetivo_fisico TEXT,               -- 'perder_peso'|'ganar_musculo'|'mantenimiento'|'salud'
    peso_kg         REAL,
    altura_cm       REAL,
    bmi             REAL,               -- calculado: peso/(altura_m^2)
    -- Contexto adicional
    nivel_actividad TEXT,               -- 'sedentario'|'ligero'|'moderado'|'activo'|'muy_activo'
    dias_gym        INTEGER,
    alergias        TEXT,               -- JSON: ["gluten","lactosa",...]
    dieta_tipo      TEXT,               -- 'omnivora'|'vegetariana'|'vegana'|'keto'
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- ── Hábitos diarios: 1 fila por día por usuario ───────────────────────────
-- Insertamos una nueva fila cada vez que el usuario actualiza sus hábitos.
-- La consulta siempre usa MAX(fecha) para obtener el estado actual.
CREATE TABLE IF NOT EXISTS habitos_diarios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fecha           TEXT    NOT NULL,   -- ISO date YYYY-MM-DD
    -- Hábitos de tiempo y pantalla
    t_meditacion    REAL,               -- minutos de meditación/día
    h_social        REAL,               -- horas de interacción social/día
    h_movil         REAL,               -- horas de uso de móvil/día
    t_pantalla      REAL,               -- horas totales de pantalla/día
    h_sol           REAL,               -- horas de exposición solar/día
    -- Hábitos alimentarios
    alim_enteros    REAL,               -- ración alimentos enteros (1-10)
    alim_procesados REAL,               -- ración alimentos procesados (1-10)
    g_proteina      REAL,               -- gramos de proteína/día
    g_fibra         REAL,               -- gramos de fibra/día
    l_agua          REAL,               -- litros de agua/día
    n_alcohol       REAL,               -- unidades de alcohol/semana
    -- Indicadores de estrés y ejercicio base
    base_estres     REAL,               -- nivel de estrés basal (1-10)
    base_ejercicio  REAL,               -- nivel base de actividad (1-5)
    -- Ejercicio del día
    m_ejercicio     REAL,               -- minutos de ejercicio/día
    i_ejercicio     REAL,               -- intensidad del ejercicio (1-5)
    pasos           REAL,               -- pasos diarios
    -- Sueño
    h_sueno         REAL,               -- horas de sueño/noche
    c_sueno         REAL,               -- calidad del sueño (1-10)
    created_at      TEXT NOT NULL,
    UNIQUE(user_id, fecha)              -- un registro por día por usuario
);
CREATE INDEX IF NOT EXISTS idx_habitos_user_fecha ON habitos_diarios(user_id, fecha DESC);

-- ── Historial de bienestar: puntuaciones diarias P1 ────────────────────
CREATE TABLE IF NOT EXISTS hist_bienestar (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fecha           TEXT    NOT NULL,   -- ISO date YYYY-MM-DD
    bf_score        REAL    NOT NULL,   -- bienestar físico [0-100]
    bm_score        REAL    NOT NULL,   -- bienestar mental [0-100]
    bg_score        REAL    NOT NULL,   -- bienestar global (media ponderada)
    modelo_usado    TEXT,               -- 'gradientboosting'|'randomforest'|'svr'
    habitos_json    TEXT,               -- snapshot de hábitos usados (JSON)
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bienestar_user_fecha ON hist_bienestar(user_id, fecha DESC);

-- ── Marcas de entrenamiento: 35 ejercicios (1RM / max peso) ──────────────
-- 1 fila por usuario (UNIQUE). Se actualiza cuando supera una marca.
CREATE TABLE IF NOT EXISTS marcas_entreno (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                 INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    -- Push: pecho
    press_banca             REAL,
    press_banca_inclinado   REAL,
    press_banca_declinado   REAL,
    chest_press             REAL,
    aperturas               REAL,
    flexiones               REAL,     -- repeticiones máximas (BW)
    -- Push: hombros
    press_militar           REAL,
    elevaciones_laterales   REAL,
    elevaciones_frontales   REAL,
    pajaros                 REAL,
    -- Pull: espalda alta
    dominadas               REAL,     -- repeticiones máximas (BW)
    remo_barra              REAL,
    remo_polea              REAL,
    pull_over               REAL,
    -- Pull: tríceps
    jalon_triceps           REAL,
    polea_triceps           REAL,
    fondos                  REAL,     -- repeticiones máximas (BW)
    extension_triceps       REAL,
    -- Pull: bíceps
    curl_biceps             REAL,
    curl_martillo           REAL,
    curl_concentrado        REAL,
    -- Legs: cuádriceps
    sentadilla              REAL,
    sentadilla_sumo         REAL,
    prensa                  REAL,
    extension_cuadriceps    REAL,
    -- Legs: isquiotibiales / glúteos
    peso_muerto             REAL,
    peso_muerto_rumano      REAL,
    curl_femoral            REAL,
    hip_thrust              REAL,
    zancadas                REAL,
    -- Legs: gemelos
    gemelo_maquina          REAL,
    elevacion_talon         REAL,
    -- Core
    plancha                 REAL,     -- segundos máximos
    crunch                  REAL,     -- repeticiones máximas
    rueda_abdominal         REAL,     -- repeticiones máximas
    updated_at              TEXT NOT NULL
);

-- ── Preferencia organoléptica: vector 19-D ─────────────────────────────
-- Aprende de las valoraciones de P4 con cada feedback del usuario.
-- Las 19 dimensiones reflejan: dulzor, acidez, amargor, salinidad,
-- umami, textura_crujiente, textura_cremosa, aroma_frutal, aroma_especiado,
-- temperatura_fria, temperatura_caliente, volumen_pequeño, volumen_grande,
-- color_vivo, presentacion_simple, presentacion_elaborada,
-- origen_vegetal, origen_animal, procesado_minimo
CREATE TABLE IF NOT EXISTS preferencia_organoleptica (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    -- 19 dimensiones explícitas (float [0.0-1.0])
    dim_dulzor              REAL DEFAULT 0.5,
    dim_acidez              REAL DEFAULT 0.5,
    dim_amargor             REAL DEFAULT 0.5,
    dim_salinidad           REAL DEFAULT 0.5,
    dim_umami               REAL DEFAULT 0.5,
    dim_textura_crujiente   REAL DEFAULT 0.5,
    dim_textura_cremosa     REAL DEFAULT 0.5,
    dim_aroma_frutal        REAL DEFAULT 0.5,
    dim_aroma_especiado     REAL DEFAULT 0.5,
    dim_temperatura_fria    REAL DEFAULT 0.5,
    dim_temperatura_caliente REAL DEFAULT 0.5,
    dim_volumen_pequeno     REAL DEFAULT 0.5,
    dim_volumen_grande      REAL DEFAULT 0.5,
    dim_color_vivo          REAL DEFAULT 0.5,
    dim_presentacion_simple REAL DEFAULT 0.5,
    dim_presentacion_elaborada REAL DEFAULT 0.5,
    dim_origen_vegetal      REAL DEFAULT 0.5,
    dim_origen_animal       REAL DEFAULT 0.5,
    dim_procesado_minimo    REAL DEFAULT 0.5,
    -- Metadatos de aprendizaje
    n_feedbacks     INTEGER DEFAULT 0,
    confianza       REAL    DEFAULT 0.0,  -- 0=sin datos, 1=muy ajustado
    updated_at      TEXT    NOT NULL
);

-- ── Despensa: inventario de alimentos del usuario ─────────────────────
-- alimento_id referencia foods.db::alimentos.id (FK lógica)
CREATE TABLE IF NOT EXISTS despensa (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    alimento_id     INTEGER NOT NULL,    -- FK lógica → foods.db::alimentos
    nombre_cache    TEXT    NOT NULL,    -- nombre cacheado para lecturas rápidas
    cantidad_g      REAL,               -- gramos disponibles
    unidades        REAL,               -- unidades si aplica
    unidad_medida   TEXT,               -- 'g'|'ml'|'unidades'|'raciones'
    fecha_caducidad TEXT,               -- ISO date
    updated_at      TEXT    NOT NULL,
    UNIQUE(user_id, alimento_id)
);
CREATE INDEX IF NOT EXISTS idx_despensa_user ON despensa(user_id);

-- ── Log de cambios: auditoría completa ───────────────────────────────
CREATE TABLE IF NOT EXISTS log_cambios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    modulo          TEXT    NOT NULL,    -- 'AUTH'|'P1'|'P2'|'P3'|'P4'|'PERFIL'
    accion          TEXT    NOT NULL,    -- 'INSERT'|'UPDATE'|'DELETE'|'LOGIN'|'LOGOUT'
    tabla           TEXT,               -- tabla afectada
    campo           TEXT,               -- campo específico (puede ser NULL)
    valor_anterior  TEXT,               -- valor antes del cambio (JSON serializado)
    valor_nuevo     TEXT,               -- valor después del cambio (JSON serializado)
    ip_address      TEXT,
    user_agent      TEXT,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_log_user  ON log_cambios(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_log_modulo ON log_cambios(modulo, created_at DESC);
"""


# ══════════════════════════════════════════════════════════════════════════════
# DDL — foods.db
# ══════════════════════════════════════════════════════════════════════════════

_DDL_FOODS = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode  = WAL;

-- ── Catálogo maestro de alimentos ────────────────────────────────────────
-- Crece con cada escaneo P3 (OCR de etiqueta) y con el catálogo estático P4.
-- v_nutricional  : JSON con todos los valores nutricionales por 100g
-- v_organoleptico: JSON con vector 19-D de propiedades sensoriales
CREATE TABLE IF NOT EXISTS alimentos (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre              TEXT    NOT NULL,
    marca               TEXT,
    categoria           TEXT,           -- 'lacteos'|'cereales'|'proteinas'|'verduras'...
    nom_supermercado    TEXT,           -- cadena donde se escaneó por primera vez
    -- Precio (último conocido)
    precio_eur          REAL,
    precio_por_100g     REAL,           -- normalizado para comparación
    -- Vector nutricional por 100g (también en JSON para flexibilidad)
    energia_kcal        REAL,
    energia_kj          REAL,
    proteinas_g         REAL,
    hidratos_g          REAL,
    azucares_g          REAL,
    grasas_g            REAL,
    grasas_saturadas_g  REAL,
    fibra_g             REAL,
    sal_g               REAL,
    -- Campos adicionales frecuentes en etiquetas españolas
    calcio_mg           REAL,
    hierro_mg           REAL,
    vitamina_c_mg       REAL,
    -- Vector nutricional completo serializado (fuente de verdad para P4)
    v_nutricional       TEXT,           -- JSON: {"energia_kcal": 110, "proteinas_g": 24, ...}
    -- Vector organoléptico 19-D (mismo espacio que preferencia_organoleptica)
    v_organoleptico     TEXT,           -- JSON: [0.2, 0.8, 0.1, ...] (19 valores)
    -- Descriptor textual frontal (extracción OCR P3)
    descriptor_frontal  TEXT,
    -- Metadatos de calidad
    completitud         REAL DEFAULT 0.0,  -- 0.0-1.0 fracción campos nutri no-NULL
    fuente              TEXT DEFAULT 'escaneo_p3',  -- 'escaneo_p3'|'catalogo_p4'|'manual'
    confianza_ocr       REAL,               -- confianza media del OCR (0-1)
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    UNIQUE(nombre, marca)
);
CREATE INDEX IF NOT EXISTS idx_alimentos_nombre ON alimentos(nombre);
CREATE INDEX IF NOT EXISTS idx_alimentos_categoria ON alimentos(categoria);
CREATE INDEX IF NOT EXISTS idx_alimentos_supermercado ON alimentos(nom_supermercado);

-- ── Escaneos nutricionales: trazabilidad completa de P3 ──────────────────
CREATE TABLE IF NOT EXISTS escaneos_nutricionales (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alimento_id     INTEGER REFERENCES alimentos(id) ON DELETE SET NULL,
    user_id         INTEGER,            -- FK lógica → users.db::users
    nom_supermercado TEXT,
    texto_ocr       TEXT,               -- texto crudo OCR (para re-parseo futuro)
    roi_coords      TEXT,               -- "(x,y,w,h)" área de la etiqueta
    completitud     REAL,
    confianza_ocr   REAL,
    imagen_hash     TEXT,               -- SHA-256 imagen → detectar duplicados
    pipeline_usado  TEXT,               -- 'easyocr'|'tesseract'|'claude_vision'
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_escaneos_alimento ON escaneos_nutricionales(alimento_id);
CREATE INDEX IF NOT EXISTS idx_escaneos_hash     ON escaneos_nutricionales(imagen_hash);

-- ── Histórico de precios por supermercado ──────────────────────────────
CREATE TABLE IF NOT EXISTS precios_supermercado (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alimento_id     INTEGER REFERENCES alimentos(id) ON DELETE SET NULL,
    nom_supermercado TEXT   NOT NULL,
    precio_texto    TEXT,               -- "3,49 €" tal como aparece en la balda
    precio_eur      REAL,
    precio_por_100g REAL,
    peso_neto_g     REAL,               -- peso del envase (para calcular precio/100g)
    user_id         INTEGER,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_precios_alimento ON precios_supermercado(alimento_id, nom_supermercado);
"""


# ══════════════════════════════════════════════════════════════════════════════
# Contexto de conexión
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def _conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")  # WAL + NORMAL es seguro y rápido
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now().date().isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# UsersDB — árbol completo del usuario
# ══════════════════════════════════════════════════════════════════════════════

class UsersDB:
    """
    CRUD completo sobre users.db.

    Paradigma:
      - Toda búsqueda de usuario usa username_hash (O(log n)).
      - Los embeddings viajan cifrados AES-256-GCM.
      - Los logs se escriben automáticamente en log_cambios.
    """

    def __init__(self, db_path: Path = USERS_DB_PATH):
        self.db_path = db_path

    def init_schema(self) -> None:
        with _conn(self.db_path) as c:
            c.executescript(_DDL_USERS)
            c.commit()

    # ── Usuarios ──────────────────────────────────────────────────────────────

    def create_user(self, nombre: str, display_name: Optional[str] = None) -> int:
        """
        Crea un nuevo usuario. Lanza ValueError si ya existe.
        Devuelve user_id.
        """
        uhash = hash_username(nombre)
        now   = _now()
        with _conn(self.db_path) as c:
            if c.execute("SELECT id FROM users WHERE username_hash=?", (uhash,)).fetchone():
                raise ValueError(f"Usuario ya registrado (hash={uhash[:8]}…)")
            cur = c.execute(
                "INSERT INTO users (username_hash, display_name, created_at, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (uhash, display_name, now, now),
            )
            c.commit()
            return cur.lastrowid

    def get_user_by_name(self, nombre: str) -> Optional[dict]:
        """
        Búsqueda O(log n) por hash del nombre.
        Devuelve el dict del usuario o None.
        """
        uhash = hash_username(nombre)
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM users WHERE username_hash=?", (uhash,)
            ).fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        with _conn(self.db_path) as c:
            row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            return dict(row) if row else None

    def update_last_login(self, user_id: int) -> None:
        with _conn(self.db_path) as c:
            c.execute("UPDATE users SET last_login=?, updated_at=? WHERE id=?",
                      (_now(), _now(), user_id))
            c.commit()

    def deactivate_user(self, user_id: int) -> None:
        with _conn(self.db_path) as c:
            c.execute("UPDATE users SET is_active=0, updated_at=? WHERE id=?",
                      (_now(), user_id))
            c.commit()

    # ── FaceEmbedding 256-D ───────────────────────────────────────────────────

    def save_face_embedding(
        self,
        user_id:        int,
        embedding:      np.ndarray,        # float32[256]
        liveness_score: Optional[float] = None,
        quality_score:  Optional[float] = None,
    ) -> int:
        """
        Cifra el embedding (AES-256-GCM) y lo guarda.
        Desactiva versiones anteriores. Devuelve el id del nuevo registro.
        """
        assert embedding.shape == (256,), f"Embedding debe ser 256-D, got {embedding.shape}"
        emb_f32 = embedding.astype(np.float32)
        ct, nonce   = _encrypt_embedding(emb_f32)
        emb_hash, salt = _hash_embedding(emb_f32)
        now = _now()

        with _conn(self.db_path) as c:
            # Desactivar versiones anteriores
            c.execute("UPDATE face_embeddings SET is_active=0 WHERE user_id=?", (user_id,))
            cur = c.execute(
                "INSERT INTO face_embeddings "
                "(user_id, embedding_blob, nonce, emb_hash, salt, dim, "
                " liveness_score, quality_score, version, is_active, created_at) "
                "VALUES (?, ?, ?, ?, ?, 256, ?, ?, "
                "  COALESCE((SELECT MAX(version)+1 FROM face_embeddings WHERE user_id=?), 1)"
                ", 1, ?)",
                (user_id, ct, nonce, emb_hash, salt,
                 liveness_score, quality_score, user_id, now),
            )
            c.commit()
            return cur.lastrowid

    def get_face_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Recupera y descifra el embedding activo del usuario.
        Devuelve np.ndarray float32[256] o None.
        """
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT embedding_blob, nonce, emb_hash, salt "
                "FROM face_embeddings WHERE user_id=? AND is_active=1",
                (user_id,),
            ).fetchone()
            if not row:
                return None
        emb = _decrypt_embedding(bytes(row["embedding_blob"]), bytes(row["nonce"]))
        # Verificar integridad
        if not _verify_embedding(emb, row["emb_hash"], row["salt"]):
            raise RuntimeError(f"Integridad del embedding comprometida para user_id={user_id}")
        return emb

    def compare_face(self, user_id: int, live_emb: np.ndarray) -> tuple[float, bool]:
        """
        Compara el embedding live con el almacenado.
        Devuelve (distancia_euclidiana, is_match).
        Umbral ajustado a 10.0 (estándar para Facenet).
        """
        stored = self.get_face_embedding(user_id)
        if stored is None:
            return float("inf"), False
        dist   = float(np.linalg.norm(live_emb.astype(np.float32) - stored))
        
        # Imprimimos la distancia en consola para que veas tus propios números al entrar
        print(f"🕵️ Distancia facial calculada: {dist:.2f}")
        
        return round(dist, 6), dist < 10.0

    # ── Perfil personal ───────────────────────────────────────────────────────

    def upsert_profile(
        self,
        user_id: int,
        edad: Optional[float]  = None,
        genero: Optional[str]  = None,
        objetivo_fisico: Optional[str] = None,
        peso_kg: Optional[float] = None,
        altura_cm: Optional[float] = None,
        nivel_actividad: Optional[str] = None,
        dias_gym: Optional[int]  = None,
        alergias: Optional[list] = None,
        dieta_tipo: Optional[str] = None,
    ) -> int:
        """Crea o actualiza el perfil. Solo sobreescribe los campos proporcionados."""
        now  = _now()
        bmi  = round(peso_kg / (altura_cm / 100) ** 2, 2) if peso_kg and altura_cm else None
        datos: dict = {}
        for k, v in {
            "edad": edad, "genero": genero, "objetivo_fisico": objetivo_fisico,
            "peso_kg": peso_kg, "altura_cm": altura_cm, "bmi": bmi,
            "nivel_actividad": nivel_actividad, "dias_gym": dias_gym,
            "alergias": json.dumps(alergias) if alergias else None,
            "dieta_tipo": dieta_tipo,
        }.items():
            if v is not None:
                datos[k] = v

        with _conn(self.db_path) as c:
            ex = c.execute("SELECT id FROM user_profile WHERE user_id=?", (user_id,)).fetchone()
            if ex:
                if datos:
                    sets = ", ".join(f"{k}=?" for k in datos)
                    c.execute(f"UPDATE user_profile SET {sets}, updated_at=? WHERE user_id=?",
                              list(datos.values()) + [now, user_id])
                c.commit()
                return ex["id"]
            cols = ["user_id", "created_at", "updated_at"] + list(datos.keys())
            cur  = c.execute(
                f"INSERT INTO user_profile ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
                [user_id, now, now] + list(datos.values()),
            )
            c.commit()
            return cur.lastrowid

    def get_profile(self, user_id: int) -> Optional[dict]:
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM user_profile WHERE user_id=?", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    # ── Hábitos diarios ───────────────────────────────────────────────────────

    def upsert_habitos(self, user_id: int, fecha: Optional[str] = None, **kwargs) -> int:
        """
        Inserta o actualiza los hábitos del día indicado (default: hoy).
        Acepta cualquier subconjunto de los 18 campos.
        """
        fecha = fecha or _today()
        now   = _now()
        campos_validos = {
            "t_meditacion", "h_social", "h_movil", "t_pantalla", "h_sol",
            "alim_enteros", "alim_procesados", "g_proteina", "g_fibra",
            "l_agua", "n_alcohol", "base_estres", "base_ejercicio",
            "m_ejercicio", "i_ejercicio", "pasos", "h_sueno", "c_sueno",
        }
        datos = {k: v for k, v in kwargs.items() if k in campos_validos and v is not None}

        with _conn(self.db_path) as c:
            ex = c.execute(
                "SELECT id FROM habitos_diarios WHERE user_id=? AND fecha=?",
                (user_id, fecha),
            ).fetchone()
            if ex:
                if datos:
                    sets = ", ".join(f"{k}=?" for k in datos)
                    c.execute(f"UPDATE habitos_diarios SET {sets} WHERE user_id=? AND fecha=?",
                              list(datos.values()) + [user_id, fecha])
                c.commit()
                return ex["id"]
            cols = ["user_id", "fecha", "created_at"] + list(datos.keys())
            cur  = c.execute(
                f"INSERT INTO habitos_diarios ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
                [user_id, fecha, now] + list(datos.values()),
            )
            c.commit()
            return cur.lastrowid

    def get_habitos_hoy(self, user_id: int) -> Optional[dict]:
        """Devuelve los hábitos del día de hoy. None si no hay registro."""
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM habitos_diarios WHERE user_id=? AND fecha=?",
                (user_id, _today()),
            ).fetchone()
            return dict(row) if row else None

    def get_habitos_recientes(self, user_id: int, dias: int = 7) -> list[dict]:
        """Devuelve los últimos N días de hábitos."""
        with _conn(self.db_path) as c:
            rows = c.execute(
                "SELECT * FROM habitos_diarios WHERE user_id=? "
                "ORDER BY fecha DESC LIMIT ?",
                (user_id, dias),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_habitos_como_vector_p1(self, user_id: int) -> dict:
        """
        Devuelve el último registro de hábitos en el formato exacto de features de P1.
        Útil para llamar directamente a WellnessService.procesar().
        """
        from p1_wellness.service import _DEFAULTS
        habitos = self.get_habitos_hoy(user_id) or {}
        perfil  = self.get_profile(user_id) or {}
        # Merge: perfil fijo + hábitos diarios, con defaults para los que falten
        vector = dict(_DEFAULTS)
        for campo in ("h_sueno", "c_sueno", "h_movil", "t_pantalla", "pasos",
                      "m_ejercicio", "i_ejercicio", "alim_enteros", "alim_procesados",
                      "g_proteina", "g_fibra", "h_social", "h_sol", "l_agua",
                      "n_alcohol", "t_meditacion"):
            if habitos.get(campo) is not None:
                vector[campo] = habitos[campo]
        if perfil.get("edad"):      vector["edad"]   = perfil["edad"]
        if perfil.get("peso_kg"):   vector["peso"]   = perfil["peso_kg"]
        if perfil.get("altura_cm"): vector["altura"] = perfil["altura_cm"] / 100
        if habitos.get("base_estres"): vector["n_estres"] = habitos["base_estres"]
        return vector

    # ── Historial de bienestar ─────────────────────────────────────────────────

    def log_bienestar(
        self,
        user_id:  int,
        bf_score: float,
        bm_score: float,
        bg_score: Optional[float] = None,
        modelo:   Optional[str]   = None,
        habitos:  Optional[dict]  = None,
        fecha:    Optional[str]   = None,
    ) -> int:
        now   = _now()
        fecha = fecha or _today()
        bg    = bg_score if bg_score is not None else round((bf_score + bm_score) / 2, 2)
        with _conn(self.db_path) as c:
            cur = c.execute(
                "INSERT OR REPLACE INTO hist_bienestar "
                "(user_id, fecha, bf_score, bm_score, bg_score, modelo_usado, habitos_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, fecha, bf_score, bm_score, bg,
                 modelo, json.dumps(habitos) if habitos else None, now),
            )
            c.commit()
            return cur.lastrowid

    def get_hist_bienestar(self, user_id: int, dias: int = 30) -> list[dict]:
        with _conn(self.db_path) as c:
            rows = c.execute(
                "SELECT fecha, bf_score, bm_score, bg_score, modelo_usado FROM hist_bienestar "
                "WHERE user_id=? ORDER BY fecha DESC LIMIT ?",
                (user_id, dias),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Marcas de entrenamiento ────────────────────────────────────────────────

    _MARCAS_COLS = [
        "press_banca", "press_banca_inclinado", "press_banca_declinado",
        "chest_press", "aperturas", "flexiones",
        "press_militar", "elevaciones_laterales", "elevaciones_frontales", "pajaros",
        "dominadas", "remo_barra", "remo_polea", "pull_over",
        "jalon_triceps", "polea_triceps", "fondos", "extension_triceps",
        "curl_biceps", "curl_martillo", "curl_concentrado",
        "sentadilla", "sentadilla_sumo", "prensa", "extension_cuadriceps",
        "peso_muerto", "peso_muerto_rumano", "curl_femoral", "hip_thrust", "zancadas",
        "gemelo_maquina", "elevacion_talon",
        "plancha", "crunch", "rueda_abdominal",
    ]

    def upsert_marca(self, user_id: int, ejercicio: str, valor: float) -> None:
        """
        Actualiza la marca personal de un ejercicio solo si el nuevo valor es mayor.
        ejercicio: nombre exacto de la columna (snake_case).
        """
        if ejercicio not in self._MARCAS_COLS:
            raise ValueError(f"Ejercicio desconocido: {ejercicio}. "
                             f"Válidos: {self._MARCAS_COLS}")
        now = _now()
        with _conn(self.db_path) as c:
            ex = c.execute("SELECT id FROM marcas_entreno WHERE user_id=?", (user_id,)).fetchone()
            if ex:
                c.execute(
                    f"UPDATE marcas_entreno SET {ejercicio}=MAX(COALESCE({ejercicio},0),?), "
                    f"updated_at=? WHERE user_id=?",
                    (valor, now, user_id),
                )
            else:
                c.execute(
                    f"INSERT INTO marcas_entreno (user_id, {ejercicio}, updated_at) VALUES (?,?,?)",
                    (user_id, valor, now),
                )
            c.commit()

    def upsert_marcas_batch(self, user_id: int, marcas: dict[str, float]) -> None:
        """Actualiza varias marcas de una vez (solo mejoras)."""
        for ej, val in marcas.items():
            self.upsert_marca(user_id, ej, val)

    def get_marcas(self, user_id: int) -> Optional[dict]:
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM marcas_entreno WHERE user_id=?", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    # ── Preferencia organoléptica 19-D ─────────────────────────────────────────

    _PREF_DIMS = [
        "sweet",   "salty",   "sour",    "bitter",  "umami",    # Sabor
    "fatty",   "spicy",   "fresh",   "fruity",  "earthy",   # Aroma
    "toasty",  "hard",    "elastic", "viscous", "crunchy",  # Textura
    "grainy",  "creamy",  "juicy",   "dry",
    ]  # 19 dimensiones exactas

    def upsert_preferencia(
        self,
        user_id:    int,
        vector:     Optional[list[float]] = None,  # lista de 19 valores [0-1]
        **kwargs,                                    # o campos individuales
    ) -> int:
        """
        Actualiza el vector de preferencias organolépticas.
        Acepta un vector [19] completo O campos individuales.
        """
        now = _now()
        datos: dict = {}
        if vector is not None:
            assert len(vector) == 19, f"Vector debe tener 19 dims, got {len(vector)}"
            datos = {dim: float(v) for dim, v in zip(self._PREF_DIMS, vector)}
        else:
            datos = {k: v for k, v in kwargs.items() if k in self._PREF_DIMS}

        with _conn(self.db_path) as c:
            ex = c.execute("SELECT id FROM preferencia_organoleptica WHERE user_id=?",
                           (user_id,)).fetchone()
            if ex:
                if datos:
                    sets = ", ".join(f"{k}=?" for k in datos)
                    c.execute(
                        f"UPDATE preferencia_organoleptica "
                        f"SET {sets}, n_feedbacks=n_feedbacks+1, updated_at=? WHERE user_id=?",
                        list(datos.values()) + [now, user_id],
                    )
                c.commit()
                return ex["id"]
            cols = ["user_id", "updated_at"] + list(datos.keys())
            cur  = c.execute(
                f"INSERT INTO preferencia_organoleptica ({','.join(cols)})"
                f" VALUES ({','.join('?'*len(cols))})",
                [user_id, now] + list(datos.values()),
            )
            c.commit()
            return cur.lastrowid

    def get_preferencia_vector(self, user_id: int) -> Optional[list[float]]:
        """Devuelve el vector 19-D o None si no existe."""
        with _conn(self.db_path) as c:
            row = c.execute(
                f"SELECT {','.join(self._PREF_DIMS)} FROM preferencia_organoleptica WHERE user_id=?",
                (user_id,),
            ).fetchone()
            if not row:
                return None
            return [row[d] for d in self._PREF_DIMS]

    # ── Despensa ───────────────────────────────────────────────────────────────

    def upsert_despensa(
        self,
        user_id:       int,
        alimento_id:   int,
        nombre_cache:  str,
        cantidad_g:    Optional[float] = None,
        unidades:      Optional[float] = None,
        unidad_medida: Optional[str]   = "g",
        fecha_caducidad: Optional[str] = None,
    ) -> int:
        now = _now()
        with _conn(self.db_path) as c:
            ex = c.execute(
                "SELECT id FROM despensa WHERE user_id=? AND alimento_id=?",
                (user_id, alimento_id),
            ).fetchone()
            if ex:
                c.execute(
                    "UPDATE despensa SET nombre_cache=?, cantidad_g=COALESCE(?,cantidad_g), "
                    "unidades=COALESCE(?,unidades), unidad_medida=COALESCE(?,unidad_medida), "
                    "fecha_caducidad=COALESCE(?,fecha_caducidad), updated_at=? "
                    "WHERE user_id=? AND alimento_id=?",
                    (nombre_cache, cantidad_g, unidades, unidad_medida,
                     fecha_caducidad, now, user_id, alimento_id),
                )
                c.commit()
                return ex["id"]
            cur = c.execute(
                "INSERT INTO despensa (user_id, alimento_id, nombre_cache, cantidad_g, "
                "unidades, unidad_medida, fecha_caducidad, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (user_id, alimento_id, nombre_cache, cantidad_g,
                 unidades, unidad_medida, fecha_caducidad, now),
            )
            c.commit()
            return cur.lastrowid

    def get_despensa(self, user_id: int) -> list[dict]:
        with _conn(self.db_path) as c:
            rows = c.execute(
                "SELECT * FROM despensa WHERE user_id=? ORDER BY nombre_cache",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def remove_de_despensa(self, user_id: int, alimento_id: int) -> None:
        with _conn(self.db_path) as c:
            c.execute("DELETE FROM despensa WHERE user_id=? AND alimento_id=?",
                      (user_id, alimento_id))
            c.commit()

    # ── Log de cambios ─────────────────────────────────────────────────────────

    def log_cambio(
        self,
        user_id:        int,
        modulo:         str,
        accion:         str,
        tabla:          Optional[str] = None,
        campo:          Optional[str] = None,
        valor_anterior: Optional[any] = None,
        valor_nuevo:    Optional[any] = None,
        ip_address:     Optional[str] = None,
        user_agent:     Optional[str] = None,
    ) -> int:
        now = _now()
        with _conn(self.db_path) as c:
            cur = c.execute(
                "INSERT INTO log_cambios (user_id, modulo, accion, tabla, campo, "
                "valor_anterior, valor_nuevo, ip_address, user_agent, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    user_id, modulo, accion, tabla, campo,
                    json.dumps(valor_anterior) if valor_anterior is not None else None,
                    json.dumps(valor_nuevo)    if valor_nuevo    is not None else None,
                    ip_address, user_agent, now,
                ),
            )
            c.commit()
            return cur.lastrowid

    def get_log_cambios(self, user_id: int, modulo: Optional[str] = None,
                        limit: int = 50) -> list[dict]:
        with _conn(self.db_path) as c:
            if modulo:
                rows = c.execute(
                    "SELECT * FROM log_cambios WHERE user_id=? AND modulo=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (user_id, modulo, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM log_cambios WHERE user_id=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Vista completa del árbol de usuario ────────────────────────────────────

    def get_full_context(self, user_id: int) -> dict:
        """
        Devuelve el contexto completo del usuario para inyectar en el grafo LangGraph.
        Este es el 'perfil_user' que consume el router.
        """
        perfil    = self.get_profile(user_id) or {}
        habitos   = self.get_habitos_hoy(user_id) or {}
        marcas    = self.get_marcas(user_id) or {}
        pref      = self.get_preferencia_vector(user_id)
        despensa  = self.get_despensa(user_id)
        bienestar = self.get_hist_bienestar(user_id, dias=7)

        return {
            "user_id":         user_id,
            "perfil":          perfil,
            "habitos_hoy":     habitos,
            "marcas_entreno":  marcas,
            "pref_organoleptica": pref,
            "despensa":        despensa,
            "hist_bienestar":  bienestar,
        }


# ══════════════════════════════════════════════════════════════════════════════
# FoodsDB — catálogo global de alimentos
# ══════════════════════════════════════════════════════════════════════════════

class FoodsDB:
    """
    CRUD sobre foods.db.
    Se alimenta principalmente desde P3 (OCR) y P4 (catálogo estático).
    """

    def __init__(self, db_path: Path = FOODS_DB_PATH):
        self.db_path = db_path

    def init_schema(self) -> None:
        with _conn(self.db_path) as c:
            c.executescript(_DDL_FOODS)
            c.commit()

    # ── Alimentos ─────────────────────────────────────────────────────────────

    def upsert_alimento(
        self,
        nombre:             str,
        marca:              Optional[str]   = None,
        categoria:          Optional[str]   = None,
        nom_supermercado:   Optional[str]   = None,
        precio_eur:         Optional[float] = None,
        peso_neto_g:        Optional[float] = None,
        v_nutricional:      Optional[dict]  = None,   # dict con campos nutricionales
        v_organoleptico:    Optional[list]  = None,   # lista 19-D
        descriptor_frontal: Optional[str]   = None,
        fuente:             str             = "escaneo_p3",
        completitud:        Optional[float] = None,
        confianza_ocr:      Optional[float] = None,
    ) -> int:
        """
        Inserta o actualiza un alimento.
        Si ya existe (nombre+marca), solo sobreescribe si mejora la completitud.
        Devuelve el id del alimento.
        """
        now = _now()
        vn  = v_nutricional or {}

        # Calcular precio/100g si se dispone del dato
        p100 = None
        if precio_eur and peso_neto_g and peso_neto_g > 0:
            p100 = round(precio_eur / peso_neto_g * 100, 4)

        campos = {
            "categoria":          categoria,
            "nom_supermercado":   nom_supermercado,
            "precio_eur":         precio_eur,
            "precio_por_100g":    p100,
            "energia_kcal":       vn.get("energia_kcal"),
            "energia_kj":         vn.get("energia_kj"),
            "proteinas_g":        vn.get("proteinas_g"),
            "hidratos_g":         vn.get("hidratos_g"),
            "azucares_g":         vn.get("azucares_g"),
            "grasas_g":           vn.get("grasas_g"),
            "grasas_saturadas_g": vn.get("grasas_saturadas_g"),
            "fibra_g":            vn.get("fibra_g"),
            "sal_g":              vn.get("sal_g"),
            "calcio_mg":          vn.get("calcio_mg"),
            "hierro_mg":          vn.get("hierro_mg"),
            "vitamina_c_mg":      vn.get("vitamina_c_mg"),
            "v_nutricional":      json.dumps(vn) if vn else None,
            "v_organoleptico":    json.dumps(v_organoleptico) if v_organoleptico else None,
            "descriptor_frontal": descriptor_frontal,
            "fuente":             fuente,
            "completitud":        completitud,
            "confianza_ocr":      confianza_ocr,
        }
        campos = {k: v for k, v in campos.items() if v is not None}

        with _conn(self.db_path) as c:
            ex = c.execute(
                "SELECT id, completitud FROM alimentos WHERE nombre=? AND marca IS ?",
                (nombre, marca),
            ).fetchone()

            if ex:
                nueva_c  = completitud or 0.0
                actual_c = ex["completitud"] or 0.0
                if nueva_c >= actual_c and campos:
                    sets = ", ".join(f"{k}=COALESCE(?,{k})" for k in campos)
                    c.execute(
                        f"UPDATE alimentos SET {sets}, updated_at=? WHERE id=?",
                        list(campos.values()) + [now, ex["id"]],
                    )
                c.commit()
                return ex["id"]

            cols = ["nombre", "marca", "created_at", "updated_at"] + list(campos.keys())
            cur  = c.execute(
                f"INSERT INTO alimentos ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
                [nombre, marca, now, now] + list(campos.values()),
            )
            c.commit()
            return cur.lastrowid

    def get_alimento(self, alimento_id: int) -> Optional[dict]:
        with _conn(self.db_path) as c:
            row = c.execute("SELECT * FROM alimentos WHERE id=?", (alimento_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            # Deserializar JSON
            if d.get("v_nutricional"):
                d["v_nutricional"] = json.loads(d["v_nutricional"])
            if d.get("v_organoleptico"):
                d["v_organoleptico"] = json.loads(d["v_organoleptico"])
            return d

    def buscar_alimento(self, nombre: str, marca: Optional[str] = None) -> list[dict]:
        with _conn(self.db_path) as c:
            if marca:
                rows = c.execute(
                    "SELECT * FROM alimentos WHERE nombre LIKE ? AND marca LIKE ? "
                    "ORDER BY completitud DESC LIMIT 20",
                    (f"%{nombre}%", f"%{marca}%"),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM alimentos WHERE nombre LIKE ? "
                    "ORDER BY completitud DESC LIMIT 20",
                    (f"%{nombre}%",),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_alimentos_despensa(self, alimento_ids: list[int]) -> list[dict]:
        """Devuelve los alimentos de la despensa del usuario (para P4)."""
        if not alimento_ids:
            return []
        placeholders = ",".join("?" * len(alimento_ids))
        with _conn(self.db_path) as c:
            rows = c.execute(
                f"SELECT * FROM alimentos WHERE id IN ({placeholders}) "
                "ORDER BY completitud DESC",
                alimento_ids,
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                if d.get("v_nutricional"):
                    d["v_nutricional"] = json.loads(d["v_nutricional"])
                if d.get("v_organoleptico"):
                    d["v_organoleptico"] = json.loads(d["v_organoleptico"])
                result.append(d)
            return result

    # ── Registro de escaneo P3 ────────────────────────────────────────────────

    def registrar_escaneo(
        self,
        nombre_producto:  str,
        v_nutricional:    dict,
        v_organoleptico:  Optional[list]  = None,
        texto_ocr:        str             = "",
        nom_supermercado: Optional[str]   = None,
        roi_coords:       Optional[str]   = None,
        completitud:      float           = 0.0,
        confianza_ocr:    Optional[float] = None,
        imagen_hash:      Optional[str]   = None,
        pipeline_usado:   Optional[str]   = None,
        user_id:          Optional[int]   = None,
        precio_eur:       Optional[float] = None,
        peso_neto_g:      Optional[float] = None,
        descriptor_frontal: Optional[str] = None,
    ) -> tuple[int, int]:
        """
        Flujo completo al recibir resultado de P3:
          1. Upsert en alimentos
          2. Insert en escaneos_nutricionales
        Devuelve (alimento_id, escaneo_id).
        """
        alimento_id = self.upsert_alimento(
            nombre            = nombre_producto,
            nom_supermercado  = nom_supermercado,
            v_nutricional     = v_nutricional,
            v_organoleptico   = v_organoleptico,
            descriptor_frontal= descriptor_frontal,
            precio_eur        = precio_eur,
            peso_neto_g       = peso_neto_g,
            fuente            = "escaneo_p3",
            completitud       = completitud,
            confianza_ocr     = confianza_ocr,
        )

        now = _now()
        with _conn(self.db_path) as c:
            cur = c.execute(
                "INSERT INTO escaneos_nutricionales "
                "(alimento_id, user_id, nom_supermercado, texto_ocr, roi_coords, "
                "completitud, confianza_ocr, imagen_hash, pipeline_usado, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (alimento_id, user_id, nom_supermercado, texto_ocr, roi_coords,
                 completitud, confianza_ocr, imagen_hash, pipeline_usado, now),
            )
            c.commit()
            return alimento_id, cur.lastrowid

    # ── Precios ───────────────────────────────────────────────────────────────

    def registrar_precio(
        self,
        alimento_id:     int,
        nom_supermercado: str,
        precio_eur:      Optional[float] = None,
        precio_texto:    Optional[str]   = None,
        peso_neto_g:     Optional[float] = None,
        user_id:         Optional[int]   = None,
    ) -> int:
        p100 = round(precio_eur / peso_neto_g * 100, 4) if precio_eur and peso_neto_g else None
        now  = _now()
        with _conn(self.db_path) as c:
            cur = c.execute(
                "INSERT INTO precios_supermercado "
                "(alimento_id, nom_supermercado, precio_texto, precio_eur, "
                "precio_por_100g, peso_neto_g, user_id, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (alimento_id, nom_supermercado, precio_texto, precio_eur,
                 p100, peso_neto_g, user_id, now),
            )
            # Actualizar precio_eur en alimentos si mejora el dato
            if precio_eur:
                c.execute(
                    "UPDATE alimentos SET precio_eur=?, precio_por_100g=COALESCE(?,precio_por_100g),"
                    " updated_at=? WHERE id=?",
                    (precio_eur, p100, now, alimento_id),
                )
            c.commit()
            return cur.lastrowid

    def get_precios(
        self,
        alimento_id:      Optional[int] = None,
        nom_supermercado: Optional[str] = None,
        limit:            int           = 50,
    ) -> list[dict]:
        with _conn(self.db_path) as c:
            q, p = "SELECT * FROM precios_supermercado WHERE 1=1", []
            if alimento_id:
                q += " AND alimento_id=?";      p.append(alimento_id)
            if nom_supermercado:
                q += " AND nom_supermercado=?"; p.append(nom_supermercado)
            rows = c.execute(q + " ORDER BY created_at DESC LIMIT ?", p + [limit]).fetchall()
            return [dict(r) for r in rows]

    def comparar_precios(self, alimento_id: int) -> list[dict]:
        """Devuelve el precio más reciente por supermercado para un alimento."""
        with _conn(self.db_path) as c:
            rows = c.execute(
                "SELECT nom_supermercado, precio_eur, precio_por_100g, created_at "
                "FROM precios_supermercado WHERE alimento_id=? "
                "GROUP BY nom_supermercado HAVING created_at=MAX(created_at) "
                "ORDER BY precio_por_100g ASC NULLS LAST",
                (alimento_id,),
            ).fetchall()
            return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# Singletons de acceso global
# ══════════════════════════════════════════════════════════════════════════════

_users_db: Optional[UsersDB] = None
_foods_db: Optional[FoodsDB] = None


def get_users_db() -> UsersDB:
    global _users_db
    if _users_db is None:
        _users_db = UsersDB()
        _users_db.init_schema()
    return _users_db


def get_foods_db() -> FoodsDB:
    global _foods_db
    if _foods_db is None:
        _foods_db = FoodsDB()
        _foods_db.init_schema()
    return _foods_db