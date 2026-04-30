# Predicción de Bienestar Físico y Mental mediante Aprendizaje Automático

Este proyecto implementa un sistema de Machine Learning capaz de predecir el nivel de bienestar físico y mental de un usuario (escala 0-100) basándose en sus métricas diarias (sueño, nutrición, ejercicio, estrés, interacciones sociales y uso de pantallas).

En su iteración actual, el proyecto ha evolucionado de un modelo estadístico plano a un **enfoque basado en Grafos de Conocimiento (Knowledge Graphs)**, simulando dinámicas biológicas y psicológicas reales (como ciclos viciosos de estrés-insomnio y cuellos de botella de recuperación muscular) para lograr predicciones de alta precisión ($R^2 > 0.90$).

---

## 📂 Estructura del Proyecto

El proyecto está modularizado para separar el código fuente de los artefactos generados por el pipeline de datos:
```text
modulo1/
│
├── wellness_dataset.py      # Generador de datos sintéticos causales (Knowledge Graph)
├── app.py                   # Pipeline de entrenamiento, Feature Engineering y evaluación
├── README.md                # Documentación del proyecto
│
└── research/                # Directorio de artefactos (generado automáticamente)
    ├── data/
    │   └── wellness_dataset.csv             # Dataset final generado por wellness_dataset.py
    │
    ├── models/
    │   ├── gradientboosting_wellness.pkl    # Modelo entrenado Gradient Boosting
    │   ├── randomforest_wellness.pkl        # Modelo entrenado Random Forest
    │   └── svr_wellness.pkl                 # Modelo entrenado Support Vector Regressor
    │
    └── graficos/
        ├── comparacion_metricas.png         # Comparativa de R² y MAE entre modelos
        ├── cross_validation.png             # Validación cruzada (5-fold)
        ├── importancia_features_bienestar_fisico.png  # Permutation importance (Físico)
        ├── importancia_features_bienestar_mental.png  # Permutation importance (Mental)
        ├── real_vs_predicho.png             # Dispersión de predicciones vs valores reales
        └── residuos.png                     # Distribución de errores del modelo