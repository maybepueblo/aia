# app/api/main.py
import os
import time
import cv2
import tempfile
import numpy as np
import logging
from typing import Optional
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.db.database import get_users_db
from app.security.cyber import cyber_manager

app = FastAPI(title="WellnessAI", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_FE = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_FE):
    app.mount("/static", StaticFiles(directory=_FE), name="static")
    @app.get("/")
    async def root(): return FileResponse(os.path.join(_FE, "index.html"))

# ====================================================
#   CONFIGURACIÓN DE LOGS (logCambios)
# ====================================================
log_cambios = logging.getLogger("logCambios")
log_cambios.setLevel(logging.INFO)

# Evitamos duplicar handlers si Uvicorn recarga el servidor
if not log_cambios.handlers:
    # Crea el archivo logCambios.log con soporte para tildes y emojis (utf-8)
    file_handler = logging.FileHandler("logCambios.log", encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s | USER:%(user_id)s | %(accion)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    log_cambios.addHandler(file_handler)

def escribir_log(user_id: int, accion: str, mensaje: str):
    log_cambios.info(mensaje, extra={"user_id": user_id, "accion": accion})

# ====================================================
#   AUTENTICACIÓN BIOMÉTRICA
# ====================================================
@app.post("/auth/register")
async def register_biometrics(request: Request, nombre: str = Form(...), foto: UploadFile = File(...)):
    db = get_users_db()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await foto.read())
            tmp_path = tmp.name
        
        embedding = cyber_manager.extract_embedding(tmp_path)
        os.remove(tmp_path)

        user_id = db.create_user(nombre)
        db.save_face_embedding(user_id, embedding)
        db.log_cambio(user_id, "AUTH", "REGISTER", ip_address=request.client.host)
        
        return {"ok": True, "user_id": user_id, "mensaje": "Rostro registrado y encriptado con éxito"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login_biometrics(request: Request, nombre: str = Form(...), foto: UploadFile = File(...)):
    db = get_users_db()
    user = db.get_user_by_name(nombre)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    user_id = user["id"]
    ip = request.client.host

    if not cyber_manager.check_rate_limit(user_id):
        db.log_cambio(user_id, "AUTH", "LOCKED_OUT", ip_address=ip)
        raise HTTPException(status_code=429, detail="Demasiados intentos. Cuenta bloqueada temporalmente.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await foto.read())
            tmp_path = tmp.name
        
        live_embedding = cyber_manager.extract_embedding(tmp_path)
        os.remove(tmp_path)

        distance, is_match = db.compare_face(user_id, live_embedding)
        failed_count = len(cyber_manager.failed_attempts.get(user_id, []))
        anomaly_score = cyber_manager.get_anomaly_score(distance, failed_count)

        if anomaly_score > 0.85:
            is_match = False

        if not is_match:
            cyber_manager.record_failed_attempt(user_id)
            db.log_cambio(user_id, "AUTH", "LOGIN_FAILED", 
                          valor_nuevo={"distance": distance, "anomaly": anomaly_score}, ip_address=ip)
            raise HTTPException(status_code=401, detail="Rostro no coincide o patrón anómalo detectado")

        db.log_cambio(user_id, "AUTH", "LOGIN_SUCCESS", 
                      valor_nuevo={"distance": distance, "anomaly": anomaly_score}, ip_address=ip)
        db.update_last_login(user_id)
        
        return {"ok": True, "user_id": user_id, "mensaje": "Acceso concedido", "distancia": distance}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================
#   GESTIÓN DE PERFIL
# ====================================================
class ProfileData(BaseModel):
    user_id: int
    edad: Optional[int] = None
    genero: Optional[str] = None
    peso_kg: Optional[float] = None
    altura_cm: Optional[float] = None

@app.post("/api/profile")
async def update_profile(data: ProfileData):
    db = get_users_db()
    db.upsert_profile(
        user_id=data.user_id,
        edad=data.edad,
        genero=data.genero,
        peso_kg=data.peso_kg,
        altura_cm=data.altura_cm
    )
    return {"ok": True, "mensaje": "Perfil actualizado"}

@app.get("/api/profile/{user_id}")
async def get_profile(user_id: int):
    db = get_users_db()
    prof = db.get_profile(user_id)
    return {"ok": True, "profile": prof or {}}

# ====================================================
#   CHAT UNIFICADO
# ====================================================
@app.post("/chat")
async def chat_unificado(
    request: Request,
    user_id: int = Form(0),
    mensaje: str = Form(""),
    imagen: UploadFile = File(None)
):
    from app.graph.router import grafo_app, AppState
    t0 = time.time()
    
    imagen_bytes = await imagen.read() if imagen else None
    
    estado_dict = {
        "user_id": user_id,
        "mensaje": mensaje,
        "historial": [],
        "perfil_user": {}
    }
    
    if imagen_bytes:
        estado_dict["imagen_bytes"] = imagen_bytes
        estado_dict["entidades"] = {"supermercado": "Desconocido", "nombre_producto": "Producto"}
        if not mensaje:
            estado_dict["mensaje"] = "Analiza esta imagen"
            
    estado = AppState(**estado_dict)
    
    # Invocamos al grafo IA
    r = grafo_app.invoke(estado)
    
    # --- PREPARAMOS LAS VARIABLES DE RESPUESTA ---
    respuesta_sistema = r.get("respuesta", "")
    intencion = r.get("intencion", "GENERAL")
    confianza = r.get("confianza", 0.0)
    tiempo_ms = (time.time() - t0) * 1000
    
    # --- 📝 GUARDAR EN LOGCAMBIOS ---
    # 1. En el archivo físico del servidor (logCambios.log)
    tiene_img = "SI" if imagen_bytes else "NO"
    texto_log = f"INTENCION: {intencion} | IMG: {tiene_img} | MSG_USER: {mensaje!r} | RESP_SISTEMA: {respuesta_sistema!r} | TIEMPO: {tiempo_ms:.0f}ms"
    escribir_log(user_id=user_id, accion="CHAT_CALL", mensaje=texto_log)
    
    # 2. En la Base de Datos (usando el método existente)
    db = get_users_db()
    db.log_cambio(
        user_id=user_id, 
        modulo="CHAT", 
        accion=intencion, 
        valor_nuevo={"msg_user": mensaje, "resp_sistema": respuesta_sistema, "tiempo_ms": round(tiempo_ms, 2)},
        ip_address=request.client.host
    )

    # --- RESPONDER AL FRONTEND ---
    resultado_dict = r.get("resultado", {})
    return {
        "respuesta": respuesta_sistema,
        "intencion": intencion,
        "confianza": confianza,
        "imagen_b64": resultado_dict.get("imagen_b64"), 
        "tiempo_ms": tiempo_ms
    }

@app.get("/health")
async def health(): 
    return {"status": "ok", "modules": ["P1", "P2", "P3", "P4"]}