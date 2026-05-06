# WellnessAI - Asistente Integral de Bienestar

WellnessAI es una aplicación web impulsada por Inteligencia Artificial diseñada para actuar como un asistente personal de salud, nutrición y entrenamiento. Utiliza una arquitectura avanzada basada en agentes (LangGraph), Modelos de Lenguaje Grandes (LLMs) y Visión Artificial para ofrecer una experiencia personalizada.

## Características Principales

* **Autenticación Biométrica (Face ID):** Inicio de sesión y registro mediante reconocimiento facial (DeepFace). Los vectores faciales (*embeddings*) se almacenan encriptados en la base de datos (AES-256 GCM) e incluyen detección de anomalías y limitación de tasa (*rate limit*).
* **Enrutamiento Inteligente (LangGraph):** El sistema analiza la intención del usuario y deriva la consulta al módulo especializado correspondiente de forma automática.
* **Módulo de Entrenamiento (M2):** Análisis de sentimiento y esfuerzo en los entrenamientos (usando `transformers` y `pysentimiento`) para registrar rutinas, peso levantado, RIR y fatiga.
* **Módulo de Visión Artificial (M3):** 
  * **Análisis de Precios:** Detecta precios en baldas de supermercado, dibujando *Bounding Boxes* (cajas delimitadoras) sobre la imagen original usando OpenCV y EasyOCR.
  * **Etiquetas Nutricionales:** Detecta el ROI (Región de Interés), recorta, mejora la imagen y extrae macronutrientes de forma estructurada.
* **Módulo de Nutrición (M4):** Generación de menús y dietas adaptadas a las características físicas del usuario (peso, altura, género, edad).
* **Auditoría y Logs:** Registro detallado de la actividad del sistema (prompts, respuestas, tiempos de ejecución y errores) en un archivo local (`logCambios.log`) de forma invisible para el usuario final.

---

## Tecnologías usadas

* **Backend:** FastAPI (Python)
* **Gestor de Paquetes/Entorno:** `uv`
* **Inteligencia Artificial & NLP:** LangGraph, Groq API (LLMs), HuggingFace (Transformers, PySentimiento)
* **Visión Artificial:** OpenCV, EasyOCR, DeepFace
* **Base de Datos:** SQLite (Local)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Integrado sin frameworks).

---

## Cómo Ejecutar la Aplicación

### 1. Requisitos Previos
* **Python 3.10 o superior** instalado en el sistema.
* El gestor de paquetes ultrarrápido **`uv`** instalado (opcional pero recomendado) o en su defecto `pip`.

### 2. Instalación de Dependencias
Abre la terminal en la carpeta raíz del proyecto (`wellnessAPP/`). 

Si usas `uv` (recomendado para sincronizar el entorno virtual ultra rápido):
```bash
uv pip install -r requirements.txt
```

(Si usas pip tradicional: pip install -r requirements.txt)

### 3. Levantar el Servidor

Para iniciar el backend de FastAPI con recarga en vivo, ejecuta el siguiente comando desde la raíz del proyecto:
Bash

uv run uvicorn app.api.main:app --reload

(Si no usas uv, simplemente ejecuta: uvicorn app.api.main:app --reload)

### 4. Acceder a la Interfaz Web

Una vez que el servidor esté corriendo (verás Application startup complete en la terminal), abre tu navegador web favorito y dirígete a la dirección local que te ha adjuntado uvicorn

## Estructura del Proyecto

```text
wellnessAPP/
├── .venv/                     # Entorno virtual de Python principal
├── app/                       # Core de la aplicación web
│   ├── api/                   # Endpoints de FastAPI y configuración del servidor
│   ├── db/                    # Lógica de base de datos y encriptación
│   │   ├── data/              
│   │   │   └── users.db       # Base de datos SQLite (usuarios y biometría)
│   │   └── database.py        
│   ├── frontend/              # Interfaz de usuario (HTML/CSS/JS)
│   ├── graph/                 # Nodos y enrutador principal de LangGraph
│   └── security/              # Ciberseguridad, rate limit y validación
├── p1_wellness/               # Módulo 1: Bienestar y Hábitos
│   ├── models/                
│   └── service.py             # Lógica de análisis de sueño, estrés y actividad
├── p2_training/               # Módulo 2: Entrenamiento y Esfuerzo
│   ├── esfuerzo_notas.py      # NLP para notas de entrenamiento
│   ├── parser.py              # Extracción de RIR, series y repeticiones
│   ├── rutina.py              # Generación de rutinas
│   └── service.py             
├── p3_vision/                 # Módulo 3: Visión Artificial
│   ├── domain/                # Esquemas Pydantic para resultados
│   ├── graph/                 # Nodos de LangGraph específicos para visión
│   ├── ocr/                   # Lógica de EasyOCR y extracción de texto
│   ├── pipelines/             # Pipelines de precios y etiquetas nutricionales
│   └── service.py             
├── p4_food/                   # Módulo 4: Nutrición y Dietas
│   ├── agentes/               # Agentes IA generadores de menú
│   ├── cesta_inteligente/     # Generador de lista de la compra
│   ├── optimizacion/          # Ajuste de macros y calorías
│   └── pipeline.py            
├── logCambios.log             # Registro de auditoría y chat (Auto-generado)
├── main.py                    # Script de entrada secundario/pruebas
├── pyproject.toml             # Configuración del proyecto y dependencias (uv)
├── uv.lock                    # Archivo de bloqueo de dependencias ultra-rápido
├── supermercado.csv / .db     # Bases de datos de productos y precios
└── README.md                  # Este archivo