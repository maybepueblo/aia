# vision/api_router.py
# ─────────────────────────────────────────────────────────────
# Router FastAPI del módulo de visión.
# Se monta en la app principal con:
#   from vision.api_router import router as vision_router
#   app.include_router(vision_router)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Annotated, Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from vision.service import VisionService
from vision.domain.schemas import TipoImagen

router = APIRouter(prefix="/vision", tags=["Visión"])

# ── Singleton del servicio ────────────────────────────────────
_svc: Optional[VisionService] = None

def get_service() -> VisionService:
    global _svc
    if _svc is None:
        _svc = VisionService()
    return _svc

Svc = Annotated[VisionService, Depends(get_service)]


# ── Endpoint principal ────────────────────────────────────────

@router.post("/escanear", summary="Analiza imagen (precios o etiqueta nutricional)")
async def escanear(
    imagen:          UploadFile       = File(...),
    supermercado:    str              = Form("Desconocido"),
    nombre_producto: str              = Form("Producto desconocido"),
    user_id:         Optional[int]    = Form(None),
    svc:             Svc              = Depends(get_service),
) -> dict:
    datos = await imagen.read()
    if not datos:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Imagen vacía")

    estado = svc.procesar_bytes(
        datos=datos, supermercado=supermercado,
        nombre_producto=nombre_producto, user_id=user_id,
        imagen_path=imagen.filename or "upload",
    )

    if estado.get("error"):
        raise HTTPException(status_code=422, detail=estado["error"])

    clf = estado.get("clasificacion")
    resp = {
        "tipo":         clf.tipo.value if clf else "desconocido",
        "confianza":    clf.confianza  if clf else 0,
        "mensaje_chat": svc.resumen_chat(estado),
        "precios":      [],
        "nutricional":  {},
    }

    if clf and clf.tipo == TipoImagen.BALDA_PRECIOS:
        r = estado.get("resultado_precios")
        if r and r.ok:
            resp["precios"] = [
                {"valor": p.valor_texto, "float": p.valor_float, "cx": p.cx, "cy": p.cy}
                for p in r.precios
            ]

    elif clf and clf.tipo == TipoImagen.ETIQUETA_NUTRI:
        r2 = estado.get("resultado_nutricional")
        if r2 and r2.ok:
            resp["nutricional"]   = r2.valores.to_dict()
            resp["completitud"]   = r2.valores.completitud()
            resp["texto_ocr"]     = r2.texto_ocr

    return resp


@router.get("/productos")
async def productos(supermercado: Optional[str] = None, svc: Svc = Depends(get_service)):
    return svc.listar_productos(supermercado)


@router.get("/nutricional")
async def nutricional(svc: Svc = Depends(get_service)):
    return svc.listar_nutricional()


@router.get("/despensa/{user_id}")
async def despensa(user_id: int, svc: Svc = Depends(get_service)):
    return svc.get_despensa(user_id)


@router.post("/despensa/{user_id}")
async def add_despensa(
    user_id: int,
    producto_id: int   = Form(...),
    cantidad:    float = Form(1.0),
    unidad:      str   = Form("ud"),
    svc: Svc           = Depends(get_service),
):
    svc.agregar_despensa(user_id, producto_id, cantidad, unidad)
    return {"ok": True}
