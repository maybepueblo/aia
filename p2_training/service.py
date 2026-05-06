# p2_training/service.py
# Adapta los módulos P2 (parser, esfuerzo_notas, metodos_musculos,
# recuperacion, rutina) exactamente como main.py de P2.
from __future__ import annotations
import sys, re, unicodedata
from pathlib import Path
from typing import Optional

_DIR = Path(__file__).parent
if str(_DIR) not in sys.path: sys.path.insert(0, str(_DIR))

def _norm(t):
    t=unicodedata.normalize("NFD",t)
    return "".join(c for c in t if unicodedata.category(c)!="Mn").lower()

def es_registro(texto): 
    return bool(re.search(r"\d+\s*[xX]\s*\d+", texto))

def es_consulta(texto):
    t=_norm(texto)
    return any(k in t for k in ["siguiente rutina","rutina para manana","que entreno","toca manana",
                                  "entreno manana","sugiereme rutina","que hago manana"])

def tipo_sesion(texto):
    t=_norm(texto)
    if any(k in t for k in ["pecho","triceps","press banca","fondos","aperturas"]): return "push"
    if any(k in t for k in ["espalda","biceps","remo","dominadas","jalon","trapecio"]): return "pull"
    if any(k in t for k in ["piernas","sentadilla","peso muerto","gluteos","gemelo"]): return "legs"
    if any(k in t for k in ["abdomen","core","planchas","crunch"]): return "core"
    return "full"

def siguiente_sesion(texto):
    t=_norm(texto)
    if any(k in t for k in ["pecho","triceps","press","fondos"]): return "pull"
    if any(k in t for k in ["espalda","biceps","remo"]): return "legs"
    if any(k in t for k in ["piernas","sentadilla","peso muerto"]): return "push"
    return "full"

class TrainingService:
    _analizador=None
    def _anlz(self):
        if not TrainingService._analizador:
            from esfuerzo_notas import cargar_analizador
            TrainingService._analizador=cargar_analizador()
        return TrainingService._analizador

    def procesar(self, mensaje, user_id=None, perfil=None):
        perfil=perfil or {}
        if es_registro(mensaje):   return self._registro(mensaje, perfil)
        if es_consulta(mensaje):   return self._consulta(mensaje, perfil)
        return self._ayuda()

    def _registro(self, texto, perfil):
        import time, numpy as np; t0=time.time()
        try:
            from parser import parsear_entrenamiento
            from esfuerzo_notas import analizar_nota
            from metodos_musculos import (cargar_lexico, procesar_sesion,
                cargar_lexicon_personal, Musculo, cuentaMusculo, CAPACIDAD_MAX, User)
            from recuperacion import daily_recovery
            from rutina import sugerir_rutina

            logs = parsear_entrenamiento(texto)
            if not logs:
                return {"ok":False,"texto":"No detecté ejercicios. Formato: * Nombre NxM Xkg RIR, nota"}

            anlz = self._anlz()
            pf   = {e.nombre: analizar_nota(e.nota_raw, anlz) for e in logs}

            lexico  = cargar_lexico(_DIR/"lexico_ejercicios.json")
            lp      = cargar_lexicon_personal(_DIR/"lexicon_personal.json")
            usuario = User(edad=int(perfil.get("edad",25)),
                           experiencia=int(perfil.get("experiencia",2)),
                           masa_magra=float(perfil.get("masa_magra",70.)))

            fv = perfil.get("f_vector")
            f_prev = np.array(fv,dtype=np.float32) if fv and len(fv)==cuentaMusculo                      else np.full(cuentaMusculo, CAPACIDAD_MAX, dtype=np.float32)

            sesion=[{"nombre":e.nombre,"series":e.series,"reps":e.reps,"rir":e.rir,
                     "pf":pf[e.nombre],"peso_kg":e.peso_kg,"es_bw":e.es_bw} for e in logs]
            f_nueva, alertas, no_enc = procesar_sesion(sesion,lexico,f_prev,
                usuario=usuario, ruta_lexicon_personal=_DIR/"lexicon_personal.json")

            f_man = daily_recovery(f_nueva,
                sleep_score=float(perfil.get("sleep_score",0.85)),
                diet_score=float(perfil.get("diet_score",0.90)))

            tipo_sig = siguiente_sesion(texto)
            rutina   = sugerir_rutina(f_man,lexico,lp,max_ejercicios=7,
                                      tipo_sesion=tipo_sig,wear_factor=usuario.exp_wear_factor)

            fatiga={Musculo(i).name:round(float(f_nueva[i]),1)
                    for i in range(cuentaMusculo) if f_nueva[i]<CAPACIDAD_MAX-0.5}

            lines=[f"💪 **Sesión registrada** — {len(logs)} ejercicio(s)"]
            if no_enc: lines.append(f"  ⚠️ Sin léxico: {no_enc}")
            if fatiga:
                lines.append("\n🔋 **Fatiga muscular:**")
                for m,v in sorted(fatiga.items(),key=lambda x:x[1]):
                    b="█"*int(v)+"░"*(10-int(v))
                    lines.append(f"  {m:<24}[{b}] {v:.1f}/10")
            if alertas: lines.append(f"\n⚠️ Sobrecarga: {', '.join(alertas)}")
            lines.append(f"\n📋 **Rutina mañana ({tipo_sig.upper()}):**")
            for r in rutina:
                pw="BW" if r.es_bw else (f"{r.peso_kg}kg" if r.peso_kg else "—")
                lines.append(f"  • {r.nombre} {r.series}×{r.reps} {pw}")

            return {"ok":True,"tipo":"registro","texto":"\n".join(lines),
                    "fatiga":fatiga,"alertas":alertas,"f_vector_nuevo":f_nueva.tolist(),
                    "f_vector_manana":f_man.tolist(),"tipo_sig":tipo_sig,
                    "tiempo_ms":round((time.time()-t0)*1000,1)}
        except ImportError as e: return {"ok":False,"texto":f"⚠️ P2 no instalado: {e}"}
        except Exception as e:   return {"ok":False,"texto":f"⚠️ Error: {e}"}

    def _consulta(self, texto, perfil):
        import time, numpy as np; t0=time.time()
        try:
            from metodos_musculos import (cargar_lexico, cargar_lexicon_personal,
                cuentaMusculo, CAPACIDAD_MAX, User)
            from rutina import sugerir_rutina
            tipo    = tipo_sesion(texto)
            lexico  = cargar_lexico(_DIR/"lexico_ejercicios.json")
            lp      = cargar_lexicon_personal(_DIR/"lexicon_personal.json")
            usuario = User(edad=int(perfil.get("edad",25)),
                           experiencia=int(perfil.get("experiencia",2)),
                           masa_magra=float(perfil.get("masa_magra",70.)))
            fv = perfil.get("f_vector")
            f  = np.array(fv,dtype=np.float32) if fv and len(fv)==cuentaMusculo                  else np.full(cuentaMusculo, CAPACIDAD_MAX*0.7, dtype=np.float32)
            rutina = sugerir_rutina(f,lexico,lp,max_ejercicios=7,tipo_sesion=tipo,
                                    wear_factor=usuario.exp_wear_factor)
            lines=[f"📋 **Rutina sugerida — {tipo.upper()}**"]
            for r in rutina:
                pw="BW" if r.es_bw else (f"{r.peso_kg}kg" if r.peso_kg else "—")
                lines.append(f"  • {r.nombre} {r.series}×{r.reps} {pw}")
            return {"ok":True,"tipo":"consulta","texto":"\n".join(lines),
                    "tiempo_ms":round((time.time()-t0)*1000,1)}
        except Exception as e: return {"ok":False,"texto":f"⚠️ Error: {e}"}

    def _ayuda(self):
        return {"ok":False,"tipo":"ayuda","texto":(
            "💪 Puedo ayudarte de dos formas:\n\n"
            "**Registrar sesión** (formato P2):\n"
            "```\n* Press banca 4x10 60kg 3RIR, me costó más de lo normal\n"
            "* Jalón 3x12 50kg 2RIR, fue bien\n* Fondos 3x15 BW\n```\n\n"
            "**Pedir rutina:** «¿Qué entreno mañana?» o «Rutina pull»")}
