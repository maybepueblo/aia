# vision/service.py
# ─────────────────────────────────────────────────────────────
# Punto de entrada del módulo de visión para la app web.
# Envuelve el grafo LangGraph y gestiona la persistencia.
# ─────────────────────────────────────────────────────────────

import time
import numpy as np
import cv2
from typing import Optional

from vision.domain.schemas import (
    VisionState, TipoImagen, ClasificacionImagen,
    ResultadoPrecios, ResultadoNutricional
)
from vision.graph.router import grafo_vision
from vision.infrastructure.db import SupermercadoDB


class VisionService:
    """
    Fachada principal del módulo de visión.
    Uso desde la app web:

        service = VisionService()
        estado  = service.procesar_bytes(image_bytes, supermercado="Mercadona")
        resumen = service.resumen_chat(estado)
    """

    def __init__(self, db_path: str = "supermercado.db"):
        self.db = SupermercadoDB(db_path)
        self.db.init_schema()

    # ── API principal ─────────────────────────────────────────

    def procesar_bytes(
        self,
        datos:           bytes,
        supermercado:    str           = "Desconocido",
        nombre_producto: str           = "Producto desconocido",
        user_id:         Optional[int] = None,
        imagen_path:     str           = "upload",
    ) -> VisionState:
        """
        Decodifica bytes de imagen y ejecuta el grafo LangGraph completo.
        Persiste los resultados en BBDD automáticamente.
        Devuelve el VisionState final para que el router web pueda leerlo.
        """
        img = self._bytes_a_bgr(datos)
        if img is None:
            return {"error": "No se pudo decodificar la imagen."}

        estado_inicial: VisionState = {
            "imagen_path":     imagen_path,
            "img_bgr":         img,
            "supermercado":    supermercado,
            "nombre_producto": nombre_producto,
        }

        t0     = time.time()
        estado = grafo_vision.invoke(estado_inicial)
        print(f"⏱  Tiempo total: {time.time() - t0:.1f}s")

        # Persistir si no hubo error
        if not estado.get("error"):
            self._persistir(estado, user_id)

        return estado

    def procesar_ruta(
        self,
        ruta:            str,
        supermercado:    str           = "Desconocido",
        nombre_producto: str           = "Producto desconocido",
        user_id:         Optional[int] = None,
    ) -> VisionState:
        """Atajo para usar desde CLI o tests."""
        datos = open(ruta, "rb").read()
        return self.procesar_bytes(datos, supermercado, nombre_producto, user_id, ruta)

    # ── Resumen para chat ─────────────────────────────────────

    @staticmethod
    def resumen_chat(estado: VisionState) -> str:
        """Genera el texto que el chat mostrará al usuario."""
        if estado.get("error"):
            return f"Error procesando imagen: {estado['error']}"

        clf: ClasificacionImagen = estado.get("clasificacion")
        if clf is None:
            return "No se pudo analizar la imagen."

        if clf.tipo == TipoImagen.BALDA_PRECIOS:
            r: ResultadoPrecios = estado.get("resultado_precios")
            if not r or not r.ok:
                return f"Error en pipeline de precios: {r.error if r else 'desconocido'}"
            if r.n_precios == 0:
                return "No detecté etiquetas de precio en la imagen."
            lineas = [f"Detecté {r.n_precios} precio(s) en {r.supermercado}:"]
            for p in r.precios:
                lineas.append(f"  • {p.valor_texto} €")
            return "\n".join(lineas)

        if clf.tipo == TipoImagen.ETIQUETA_NUTRI:
            r2: ResultadoNutricional = estado.get("resultado_nutricional")
            if not r2 or not r2.ok:
                return f"Error en pipeline nutricional: {r2.error if r2 else 'desconocido'}"
            v = r2.valores
            lineas = [f"Información nutricional de '{r2.nombre_producto}' (por 100g):"]
            mapa = {
                "Energía":   f"{v.energia_kcal} kcal"  if v.energia_kcal  else None,
                "Grasas":    f"{v.grasas_g} g"          if v.grasas_g      else None,
                "Hidratos":  f"{v.hidratos_g} g"        if v.hidratos_g    else None,
                "Azúcares":  f"{v.azucares_g} g"        if v.azucares_g    else None,
                "Fibra":     f"{v.fibra_g} g"           if v.fibra_g       else None,
                "Proteínas": f"{v.proteinas_g} g"       if v.proteinas_g   else None,
                "Sal":       f"{v.sal_g} g"             if v.sal_g         else None,
            }
            for k, val in mapa.items():
                if val:
                    lineas.append(f"  • {k}: {val}")
            lineas.append(f"\n(Completitud: {v.completitud():.0%})")
            return "\n".join(lineas)

        return "Imagen no clasificada con confianza suficiente."

    # ── Consultas BBDD ────────────────────────────────────────

    def listar_productos(self, supermercado: Optional[str] = None) -> list:
        return self.db.listar_productos(supermercado)

    def listar_nutricional(self) -> list:
        return self.db.listar_nutricional()

    def get_despensa(self, user_id: int) -> list:
        return self.db.get_despensa_usuario(user_id)

    def agregar_despensa(self, user_id: int, producto_id: int,
                         cantidad: float = 1.0, unidad: str = "ud") -> None:
        self.db.upsert_despensa(user_id, producto_id, cantidad, unidad)

    # ── Helpers privados ──────────────────────────────────────

    @staticmethod
    def _bytes_a_bgr(datos: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(datos, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _persistir(self, estado: VisionState, user_id: Optional[int]) -> None:
        clf: ClasificacionImagen = estado.get("clasificacion")
        if clf is None:
            return
        try:
            if clf.tipo == TipoImagen.BALDA_PRECIOS:
                r = estado.get("resultado_precios")
                if r and r.ok:
                    self.db.guardar_precios(r, user_id)
            elif clf.tipo == TipoImagen.ETIQUETA_NUTRI:
                r2 = estado.get("resultado_nutricional")
                if r2 and r2.ok:
                    self.db.guardar_nutricional(r2, user_id)
        except Exception as e:
            print(f"[VisionService] Advertencia al persistir: {e}")
