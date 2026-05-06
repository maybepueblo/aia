# app/graph/router.py
# Grafo LangGraph principal — clasifica intención y despacha módulos.
from __future__ import annotations
import re, unicodedata, sys, time
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

class AppState(TypedDict, total=False):
    user_id:      int
    mensaje:      str
    imagen_bytes: Optional[bytes]
    historial:    list[dict]
    perfil_user:  dict
    intencion:    str
    confianza:    float
    entidades:    dict
    resultado:    dict
    respuesta:    str
    error:        Optional[str]
    tiempo_ms:    float

_KW = {
    "WELLNESS": [
        "bienestar","habitos","sueno","dormir","estres","ansiedad",
        "energia","cansancio","meditacion","pasos","pantalla","movil",
        "social","alcohol","agua","hidratacion","relajado","agobiado",
        "progreso","mejora","como estoy","mis habitos","levanto","acuesto",
        "dia", "hoy", "siento", "animo"
    ],
    "TRAINING": [
        "entrenamiento","ejercicio","rutina","sesion","pecho","espalda",
        "biceps","triceps","piernas","hombros","core","abdomen","gluteos",
        "press","jalon","sentadilla","peso muerto","dominadas",
        "series","repeticiones","rir","musculo","fatiga",
        "gym","gimnasio","fuerza","entrene","hice","levante","toca",
        "entreno", "registrar", "apuntar", "pesas", "entrenar"
    ],
    "FOOD": [
        "dieta","menu","receta","despensa","comer","comida",
        "alimentos","cesta","compra","ingredientes","plan",
        "que como","recomendacion","nutricion",
        "grasa", "peso", "definicion", "volumen", "muscular", "kilos", "adelgazar"
    ],
    "VISION": [
        "etiqueta", "nutricional", "escanear", "balda", "precio", 
        "analiza", "foto", "imagen", "supermercado", "producto"
    ]
}

def _norm(t):
    t=unicodedata.normalize("NFD",t)
    return "".join(c for c in t if unicodedata.category(c)!="Mn").lower()

def clasificar(mensaje, imagen):
    if imagen: return "VISION", 0.97, {}
    texto=_norm(mensaje)
    if re.search(r"\d+\s*[xX]\s*\d+", mensaje):
        return "TRAINING", 0.98, {}
    scores={k:sum(1. for kw in kws if kw in texto) for k,kws in _KW.items()}
    best=max(scores,key=scores.get)
    if scores[best]<1.: best="GENERAL"
    total=sum(scores.values()) or 1.
    conf=round(min(scores.get(best,0)/total,1.),3)
    ent={}
    for sm in ("mercadona","lidl","aldi","carrefour","eroski","dia"):
        if sm in texto: ent["supermercado"]=sm.capitalize(); break
    return best, conf, ent

def nodo_clasificar(state):
    t0=time.time()
    i,c,e=clasificar(state.get("mensaje",""),state.get("imagen_bytes"))
    print(f"\n[Router] {i} ({c:.0%}) | {state.get('mensaje','')[:60]}")
    return {"intencion":i,"confianza":c,"entidades":e,"tiempo_ms":(time.time()-t0)*1000}

def nodo_wellness(state):
    try:
        from p1_wellness.service import WellnessService
        r=WellnessService().procesar(state.get("mensaje",""),state.get("historial",[]),state.get("user_id"))
        return {"resultado":r,"respuesta":r["texto"]}
    except Exception as e: return {"resultado":{},"respuesta":f"⚠️ P1: {e}","error":str(e)}

def nodo_training(state):
    try:
        from p2_training.service import TrainingService
        r=TrainingService().procesar(state.get("mensaje",""),state.get("user_id"),state.get("perfil_user",{}))
        return {"resultado":r,"respuesta":r["texto"]}
    except Exception as e: return {"resultado":{},"respuesta":f"⚠️ P2: {e}","error":str(e)}

def nodo_vision(state):
    if not state.get("imagen_bytes"):
        return {
            "resultado": {}, 
            "respuesta": "📷 ¡Perfecto! Por favor, pulsa el botón **+** (abajo a la izquierda) para subir la foto."
        }
        
    import sys
    import cv2
    import base64
    import numpy as np
    
    try:
        import p3_vision
        sys.modules['vision'] = p3_vision 
        from p3_vision.service import VisionService
        
        svc = VisionService()
        ent = state.get("entidades", {})
        
        # Ejecutamos el módulo de visión (devuelve un dict con el estado de LangGraph)
        ev = svc.procesar_bytes(
            state["imagen_bytes"],
            supermercado=ent.get("supermercado", "Desconocido"),
            nombre_producto=ent.get("nombre_producto", "Producto"),
            user_id=state.get("user_id")
        )
        
        respuesta_texto = svc.resumen_chat(ev)
        img_procesada = None
        
        # --- EXTRACCIÓN CORRECTA DE LA IMAGEN GENERADA ---
        if isinstance(ev, dict):
            # Si es el diccionario nativo de LangGraph
            res_precios = ev.get("resultado_precios")
            res_nutri = ev.get("resultado_nutricional")
            
            # Sacamos 'imagen_anotada' (cajas verdes) o 'img_procesada' (tabla recortada)
            if res_precios and hasattr(res_precios, "imagen_anotada"):
                img_procesada = res_precios.imagen_anotada
            elif res_nutri and hasattr(res_nutri, "img_procesada"):
                img_procesada = res_nutri.img_procesada
        else:
            # Por si tu VisionService lo envuelve en un objeto personalizado
            res_precios = getattr(ev, "resultado_precios", None)
            res_nutri = getattr(ev, "resultado_nutricional", None)
            
            if res_precios and hasattr(res_precios, "imagen_anotada"):
                img_procesada = res_precios.imagen_anotada
            elif res_nutri and hasattr(res_nutri, "img_procesada"):
                img_procesada = res_nutri.img_procesada

        # Convertimos la matriz de OpenCV a texto (Base64) para el chat
        imagen_b64 = None
        if img_procesada is not None:
            _, buffer = cv2.imencode('.jpg', img_procesada)
            imagen_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "resultado": {"imagen_b64": imagen_b64}, 
            "respuesta": respuesta_texto
        }
    except Exception as e: 
        return {"resultado": {}, "respuesta": f"⚠️ P3: {e}", "error": str(e)}

def nodo_food(state):
    import sys
    import re
    import os
    from pathlib import Path
    from app.db.database import get_users_db
    
    os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"  # Asegúrate de configurar tu clave de API de Groq aquí

    p4_path = str(Path(__file__).parent.parent.parent / "p4_food")
    if p4_path not in sys.path:
        sys.path.insert(0, p4_path)
        
    try:
        user_id = state.get("user_id", 0)
        db = get_users_db()
        
        # 1. Recuperar perfil real del usuario desde la base de datos
        pu = {}
        if user_id:
            pu = db.get_profile(user_id) or {}
            
        mensaje = state.get("mensaje", "").lower()
        
        # 2. Extraer datos si el usuario los acaba de dar (Ej: "peso 80kg, mido 1.80...")
        nuevos_datos = {}
        
        m_peso = re.search(r"(?:peso|pesando)\s*(\d+[\.,]?\d*)", mensaje)
        if m_peso: nuevos_datos["peso_kg"] = float(m_peso.group(1).replace(",","."))
        elif re.search(r"(\d+[\.,]?\d*)\s*(?:kg|kilos)", mensaje):
            nuevos_datos["peso_kg"] = float(re.search(r"(\d+[\.,]?\d*)\s*(?:kg|kilos)", mensaje).group(1).replace(",","."))
        
        m_alt = re.search(r"(?:mido|altura)\s*(\d+[\.,]?\d*)", mensaje)
        if m_alt: 
            v = float(m_alt.group(1).replace(",","."))
            nuevos_datos["altura_cm"] = v if v > 3 else v * 100
        elif re.search(r"(\d+[\.,]?\d*)\s*(?:cm|centimetros)", mensaje):
            nuevos_datos["altura_cm"] = float(re.search(r"(\d+[\.,]?\d*)\s*(?:cm|centimetros)", mensaje).group(1).replace(",","."))

        m_edad = re.search(r"(\d+)\s*a[ñn]os", mensaje)
        if m_edad: nuevos_datos["edad"] = float(m_edad.group(1))

        if re.search(r"\b(hombre|chico|var[oó]n)\b", mensaje): nuevos_datos["genero"] = "hombre"
        elif re.search(r"\b(mujer|chica|hembra)\b", mensaje): nuevos_datos["genero"] = "mujer"
        
        # Guardar en Base de Datos si cazamos nuevos datos
        if nuevos_datos and user_id:
            db.upsert_profile(user_id, **nuevos_datos)
            pu.update(nuevos_datos)
        elif nuevos_datos: # Por si es un usuario invitado/sin login
            pu.update(nuevos_datos)

        # 3. Comprobar qué datos faltan en el perfil
        faltan = []
        peso = pu.get("peso_kg") or pu.get("peso")
        altura = pu.get("altura_cm")
        edad = pu.get("edad")
        genero = pu.get("genero") or pu.get("sexo")

        if not peso: faltan.append("peso")
        if not altura: faltan.append("altura")
        if not edad: faltan.append("edad")
        if not genero: faltan.append("sexo")

        # 4. Si falta algo, pausamos y le preguntamos amablemente
        if faltan:
            return {
                "resultado": {},
                "respuesta": (
                    f"¡Me encantaría diseñarte ese menú! 🍽️\n\n"
                    f"Pero para poder calcular tus necesidades calóricas exactas, me falta saber tu **{', '.join(faltan)}**.\n\n"
                    f"*(Dímelo por aquí, por ejemplo: \"Peso 75kg, mido 180, tengo 25 años y soy hombre\")*"
                )
            }

        # 5. Si tenemos todo el perfil completo, ¡generamos el menú con P4!
        from cesta_inteligente.pipeline import Pipeline
        
        sexo_en = "male" if genero in ["hombre", "male", 1, "varon"] else "female"
        
        pipe = Pipeline(respuesta_demo=None)
        perfil, cesta, menu = pipe.ejecutar(
            texto_usuario=mensaje,
            user_id=str(user_id),
            peso_kg=float(peso), 
            altura_cm=float(altura),
            edad=int(edad), 
            sexo=sexo_en
        )
        
        lines = [
            "🍽️ **Plan nutricional personalizado**",
            f"  alpha (hedonista↔fitness): {perfil.alpha:.2f}",
            "",
            "**Menú:**",
            menu
        ]
        return {"resultado": {"menu": menu}, "respuesta": "\n".join(lines)}
    except Exception as e: 
        return {"resultado": {}, "respuesta": f"⚠️ P4: {e}", "error": str(e)}

def nodo_general(state):
    t=_norm(state.get("mensaje",""))
    if any(s in t for s in ["hola","buenas","hey","buenos dias"]):
        resp=("¡Hola! 👋 Soy tu asistente de bienestar integral.\n\n"
              "  🧘 **P1 Bienestar** — cuéntame tus hábitos\n"
              "  💪 **P2 Entrenamiento** — registra sesión o pide rutina\n"
              "  📷 **P3 Visión** — foto de etiqueta nutricional o balda\n"
              "  🍽️ **P4 Dieta** — cuéntame tu objetivo nutricional")
    else:
        resp="No entendí del todo. Puedo ayudarte con bienestar, entrenamiento, nutrición o dieta."
    return {"resultado":{},"respuesta":resp}

def _ruta(state):
    return {"WELLNESS":"nodo_wellness","TRAINING":"nodo_training",
            "VISION":"nodo_vision","FOOD":"nodo_food"
            }.get(state.get("intencion","GENERAL"),"nodo_general")

def build_graph():
    g=StateGraph(AppState)
    for n,f in [("nodo_clasificar",nodo_clasificar),("nodo_wellness",nodo_wellness),
                ("nodo_training",nodo_training),("nodo_vision",nodo_vision),
                ("nodo_food",nodo_food),("nodo_general",nodo_general)]:
        g.add_node(n,f)
    g.set_entry_point("nodo_clasificar")
    g.add_conditional_edges("nodo_clasificar",_ruta,{
        "nodo_wellness":"nodo_wellness","nodo_training":"nodo_training",
        "nodo_vision":"nodo_vision","nodo_food":"nodo_food","nodo_general":"nodo_general"})
    for n in ("nodo_wellness","nodo_training","nodo_vision","nodo_food","nodo_general"):
        g.add_edge(n,END)
    return g

grafo_app=build_graph().compile()
