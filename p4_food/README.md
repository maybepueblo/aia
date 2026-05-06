# Cesta Inteligente — Sistema de Recomendación

**Arquitectura:** LangGraph + Llama 3.3 70B (Groq) + DEAP (Algoritmo Genético)  
**Universidad Rey Juan Carlos — AIA Práctica 4 → Modularizado para aplicación web**

---

## Estructura del proyecto

```
cesta_inteligente/
│
├── __init__.py              ← Punto de entrada público (Pipeline + contratos)
├── pipeline.py              ← Orquestador de las 4 fases
├── demo.py                  ← Equivalente a la celda 13 del notebook original
│
├── modelos/
│   └── contratos.py         ← DIMS, MACROS, FuzzyMacro, UserProfile, ArticuloCompra
│
├── nucleo/
│   ├── llm_utils.py         ← Parser JSON robusto + llm_call con reintentos
│   └── calculadora_nutricional.py  ← Mifflin-St Jeor (puro, sin LLM)
│
├── agentes/
│   ├── agente_intencion.py  ← Grafo 1 LangGraph: texto → intent JSON
│   └── agente_organoleptico.py  ← Grafo 3 LangGraph: intent → vector 19D
│
├── optimizacion/
│   └── optimizador.py       ← EvaluadorCesta (función J) + BuscadorCesta (DEAP)
│
├── generacion/
│   └── generador_menu.py    ← Fase 3: cesta → menú diario (LLM creativo)
│
└── datos/
    └── catalogo.py          ← CatalogoAlimentos: carga CSV, genera demo
```

---

## Flujo global (4 fases)

| Fase | Módulo | Entrada → Salida |
|------|--------|-----------------|
| 0 ETL | `datos/catalogo.py` | CSV → `List[ArticuloCompra]` |
| 1A Cognitiva | `agentes/agente_intencion.py` | Texto libre → `dict` intención |
| 1B Nutricional | `nucleo/calculadora_nutricional.py` | `dict` biométrico → `FuzzyMacro` x8 |
| 1C Organoléptica | `agentes/agente_organoleptico.py` | `dict` intención → vector 19D |
| 2 Optimización | `optimizacion/optimizador.py` | `UserProfile` + pool → cesta óptima |
| 3 Generativa | `generacion/generador_menu.py` | Cesta → menú Markdown |

---

## Instalación

```bash
pip install groq langchain-groq langgraph langchain langchain-core \
            pydantic numpy pandas deap tqdm
```

## Uso rápido

```python
import os
os.environ["GROQ_API_KEY"] = "gsk_..."

from cesta_inteligente import Pipeline

pipe = Pipeline()
perfil, cesta, menu = pipe.ejecutar(
    texto_usuario = "Quiero perder tripa, voy al gym 3 veces...",
    peso_kg=88, altura_cm=182, edad=34, sexo="male",
)
```

## Uso desde web (Flask/FastAPI)

```python
# Instanciar una vez al arrancar la app
pipe = Pipeline(csv_catalogo="ruta/supermercado.csv", respuesta_demo=None)

# Por petición: solo construir perfil y optimizar
@app.post("/recomendar")
def recomendar(datos: DatosUsuario):
    perfil = pipe.construir_perfil(
        texto_usuario = datos.texto,
        peso_kg=datos.peso, altura_cm=datos.altura,
        edad=datos.edad, sexo=datos.sexo,
    )
    cesta, j = pipe.optimizar_cesta(perfil)
    menu = pipe.generar_menu(cesta)
    return {"perfil": perfil.model_dump(), "cesta": [...], "menu": menu}
```

---

## Módulos futuros previstos

El sistema está diseñado para crecer con módulos adicionales que comparten
los mismos contratos (`UserProfile`, `ArticuloCompra`):

- **`despensa/`** — Inventario personal del usuario (referencias a BBDD común + cantidades)
- **`entrenamiento/`** — Registro de sesiones de entrenamiento
- **`exploracion/`** — Combinaciones organolépticas entre alimentos seleccionados
- **`recetario/`** — Recomendación de recetas desde alimentos de la despensa

Todos comparten `modelos/contratos.py` como API de datos.

---

## Decisiones arquitectónicas clave

| Decisión | Justificación |
|----------|---------------|
| `CalculadoraNutricional` es clase pura (sin LLM) | Mifflin-St Jeor es aritmética determinista. LangGraph añadiría latencia y puntos de fallo sin beneficio. |
| Penalización lineal (no exponencial) en `FuzzyMacro` | El GA siempre recibe gradiente positivo. Con exponenciales, cestas alejadas del óptimo obtienen score ≈ 0 y DEAP no sabe qué dirección mejorar. |
| Castigo −1000 (no −inf) para alergias | DEAP puede medir la distancia entre cestas malas. Con −inf no hay gradiente. |
| CLIPPING post-mutación en el GA | Evita gramos negativos y cantidades absurdas que sesgarían la función de coste. |
| Merge seguro en Grafo 1 | Evita que la amnesia del LLM machaque datos previos (allergens, flavor_mentions). |
| Parámetros biométricos con prioridad sobre LLM | En producción, la app conoce el peso/altura exactos. El LLM se reserva para lo que hace bien: interpretar objetivos y preferencias en lenguaje natural. |
