# p1_wellness/service.py
# Adapta el modelo entrenado por app.py (P1) al chat.
# Features y targets EXACTOS de wellness_dataset.py.
from __future__ import annotations
import pickle, re, unicodedata
from pathlib import Path
from typing import Optional
import numpy as np

FEATURES = [
    "edad","sexo","altura","bmi","peso",
    "h_sueno","c_sueno","n_estres","h_movil","t_pantalla",
    "pasos","i_ejercicio","m_ejercicio",
    "alim_enteros","alim_procesados","g_proteina","g_fibra",
    "h_social","h_sol","l_agua","n_alcohol","t_meditacion",
]
_DEFAULTS = {
    "edad":35.0,"sexo":0.5,"altura":1.70,"bmi":24.0,"peso":70.0,
    "h_sueno":7.0,"c_sueno":6.5,"n_estres":5.0,"h_movil":4.0,
    "t_pantalla":6.0,"pasos":7000.0,"i_ejercicio":3.0,"m_ejercicio":30.0,
    "alim_enteros":6.0,"alim_procesados":4.0,"g_proteina":90.0,
    "g_fibra":25.0,"h_social":5.0,"h_sol":1.5,"l_agua":2.0,
    "n_alcohol":1.0,"t_meditacion":10.0,
}
_NEG = {"n_estres","h_movil","t_pantalla","alim_procesados","n_alcohol"}
_REC = {
    "h_sueno":"Intenta dormir 30 min más esta semana",
    "c_sueno":"Evita pantallas 1h antes de acostarte",
    "n_estres":"Practica 5 min de respiración diafragmática al despertar",
    "h_movil":"Reduce el móvil 30 min al día",
    "t_pantalla":"Aplica la regla 20-20-20 para tus ojos",
    "pasos":"Añade un paseo de 10 min después de comer",
    "m_ejercicio":"Suma 10 min de movimiento moderado al día",
    "i_ejercicio":"Sube ligeramente la intensidad en tu próxima sesión",
    "alim_enteros":"Añade una ración de verdura más a tu próxima comida",
    "alim_procesados":"Sustituye un snack procesado por fruta o frutos secos",
    "g_proteina":"Incluye una fuente de proteína en el desayuno",
    "g_fibra":"Come legumbres al menos 2 días esta semana",
    "h_social":"Queda con alguien esta semana, aunque sea 30 min",
    "h_sol":"Sal 15 min al exterior a primera hora de la mañana",
    "l_agua":"Pon un vaso de agua en tu mesa como recordatorio",
    "n_alcohol":"Sustituye una copa por agua con gas y limón",
    "t_meditacion":"Prueba 5 min de meditación guiada antes de dormir",
}
_ESTRES = [
    ("muy estresado",8.5),("bastante estresado",7.0),("agobiado",7.5),
    ("ansioso",7.0),("algo estresado",6.0),("poco estresado",3.0),
    ("sin estres",1.5),("nada estresado",1.5),("relajado",2.5),("tranquilo",3.0),
]

def _norm(t):
    t = unicodedata.normalize("NFD",t)
    return "".join(c for c in t if unicodedata.category(c)!="Mn").lower()

def extraer_habitos(mensaje, historial):
    ctx = _norm(mensaje+" "+" ".join(h.get("content","") for h in historial[-6:]))
    h = dict(_DEFAULTS)
    m = re.search(r"(?:duermo|dormi|sueno|horas? (?:de )?sueno)[^\d]*(\d+(?:[.,]\d+)?)",ctx)
    if m: h["h_sueno"]=float(m.group(1).replace(",","."))
    m = re.search(r"(\d+)\s*min(?:utos?)?\s*(?:de\s+)?ejercicio",ctx)
    if m: h["m_ejercicio"]=float(m.group(1))
    m = re.search(r"(\d{3,5})\s*pasos",ctx)
    if m: h["pasos"]=float(m.group(1))
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*l(?:itros?)?\s*(?:de\s+)?agua",ctx)
    if m: h["l_agua"]=float(m.group(1).replace(",","."))
    m = re.search(r"(\d+)\s*min(?:utos?)?\s*(?:de\s+)?meditac",ctx)
    if m: h["t_meditacion"]=float(m.group(1))
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*h(?:oras?)?\s*(?:de\s+)?(?:movil|telefono|pantalla)",ctx)
    if m: h["h_movil"]=float(m.group(1).replace(",","."))
    for kw,val in _ESTRES:
        if kw in ctx: h["n_estres"]=val; break
    return h

class WellnessService:
    _pipe = None
    _CANDIDATES = ["randomforest_wellness.pkl","gradientboosting_wellness.pkl","svr_wellness.pkl","best_model.pkl"]
    _DIR = Path(__file__).parent/"models"

    def _load(self):
        if WellnessService._pipe: return WellnessService._pipe
        for n in self._CANDIDATES:
            p = self._DIR/n
            if p.exists():
                with open(p,"rb") as f: bundle=pickle.load(f)
                WellnessService._pipe = bundle["pipeline"]
                print(f"[WellnessService] {n}")
                return WellnessService._pipe
        raise FileNotFoundError(f"Modelos no encontrados en {self._DIR}. Ejecuta p1_wellness/train_models.py")

    def procesar(self, mensaje, historial=[], user_id=None):
        import time; t0=time.time()
        hab  = extraer_habitos(mensaje, historial)
        pipe = self._load()
        X    = np.array([[hab[f] for f in FEATURES]])
        pred = pipe.predict(X)[0]
        bf   = round(float(pred[0]),1)
        bm   = round(float(pred[1]),1)

        nudge={}
        for feat in FEATURES:
            if feat not in _REC: continue
            h2=dict(hab)
            h2[feat]=max(hab[feat]*0.85,0.) if feat in _NEG else hab[feat]*1.12
            p2=pipe.predict(np.array([[h2[f] for f in FEATURES]]))[0]
            nudge[feat]=float((p2[0]-pred[0])+(p2[1]-pred[1]))
        top=[f for f in sorted(nudge,key=nudge.get,reverse=True)[:3] if nudge[f]>0.01]
        rec=[_REC[f] for f in top] or ["¡Lo estás haciendo bien! Mantén tus hábitos."]

        lines=["📊 **Análisis de bienestar**",
               f"  • Bienestar físico: **{bf}/100**",
               f"  • Bienestar mental: **{bm}/100**","",
               "💡 **Recomendaciones para esta semana:**"]
        for i,r in enumerate(rec,1): lines.append(f"  {i}. {r}")

        return {"texto":"\n".join(lines),"bienestar_fisico":bf,"bienestar_mental":bm,
                "recomendaciones":rec,"habitos":hab,"tiempo_ms":round((time.time()-t0)*1000,1)}
