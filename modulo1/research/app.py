import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import os 
import pickle
os.makedirs('modulo1/research/models', exist_ok=True)

df = pd.read_csv('modulo1/research/data/wellness_dataset.csv')

features = ['edad', 'sexo', 'altura', 'bmi', 'peso', 'h_sueno', 'c_sueno', 'n_estres', 'h_movil', 't_pantalla', 'pasos', 'i_ejercicio', 'm_ejercicio', 'alim_enteros', 'alim_procesados', 'g_proteina', 'g_fibra', 'h_social', 'h_sol', 
            'l_agua', 'n_alcohol', 't_meditacion']

targets = ['bienestar_fisico', 'bienestar_mental']

x = df[features]
y = df[targets]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

modelos = {
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
    'GradientBoosting':MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=4)),
    'SVR': MultiOutputRegressor(SVR(kernel='rbf', C=10, epsilon=0.2))
}

# entrenamos, evaluamos y guardamos
os.makedirs('modulo1/research/models', exist_ok=True)

resultados = {}

for nombre, modelo in modelos.items():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   modelo)
    ])

    # Cross-validation
    scores = cross_val_score(pipe, x_train, y_train, cv=5, scoring='r2')

    # Entrenar sobre todo el train
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)

    # Métricas por target
    metricas = {}
    for i, target in enumerate(targets):
        metricas[target] = {
            'r2':  r2_score(y_test[target], y_pred[:, i]),
            'mae': mean_absolute_error(y_test[target], y_pred[:, i]),
            'y_pred': y_pred[:, i],
        }

    resultados[nombre] = {
        'pipe':    pipe,
        'r2_cv':   scores.mean(),
        'metricas': metricas,
    }

    # Guardar modelo
    bundle = {
        'pipeline':      pipe,
        'modelo_nombre': nombre,
        'features':      features,
        'targets':       targets,
    }
    ruta = f'modulo1/research/models/{nombre.lower()}_wellness.pkl'
    with open(ruta, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'{nombre} guardado en {ruta}')

# graficar pa comparar
os.makedirs('modulo1/research/graficos', exist_ok=True)

COLORES = {
    'RandomForest':    '#2ECC71',
    'GradientBoosting':'#3498DB',
    'SVR':             '#E74C3C',
}

nombres   = list(resultados.keys())
y_test_np = y_test.to_numpy()

# r2 y MAE
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Comparación de modelos — Métricas por target', fontsize=14, fontweight='bold')

x      = np.arange(len(targets))
width  = 0.25
offset = [-width, 0, width]

for ax, metrica in zip(axes, ['r2', 'mae']):
    for i, nombre in enumerate(nombres):
        valores = [resultados[nombre]['metricas'][t][metrica] for t in targets]
        ax.bar(x + offset[i], valores, width, label=nombre,
               color=COLORES[nombre], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(['Bienestar Físico', 'Bienestar Mental'])
    ax.set_title('R² (mayor es mejor)' if metrica == 'r2' else 'MAE (menor es mejor)')
    ax.set_ylabel('R²' if metrica == 'r2' else 'Error absoluto medio')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    if metrica == 'r2':
        ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('modulo1/research/graficos/comparacion_metricas.png', dpi=130, bbox_inches='tight')
print('Guardado: modulo1/research/graficos/comparacion_metricas.png')

# real vs predicho
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Real vs Predicho por modelo y target', fontsize=14, fontweight='bold')

for col, nombre in enumerate(nombres):
    for row, target in enumerate(targets):
        ax     = axes[row, col]
        y_real = y_test[target].to_numpy()
        y_pred = resultados[nombre]['metricas'][target]['y_pred']
        r2     = resultados[nombre]['metricas'][target]['r2']

        ax.scatter(y_real, y_pred, alpha=0.4, s=16,
                   color=COLORES[nombre], edgecolors='none')
        lims = [min(y_real.min(), y_pred.min()) - 2,
                max(y_real.max(), y_pred.max()) + 2]
        ax.plot(lims, lims, 'w--', lw=1.2, alpha=0.6)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f'{nombre}\n{target}  (R²={r2:.3f})', fontsize=10)
        ax.set_xlabel('Real');  ax.set_ylabel('Predicho')
        ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('modulo1/research/graficos/real_vs_predicho.png', dpi=130, bbox_inches='tight')
print('Guardado: modulo1/research/graficos/real_vs_predicho.png')

# residuos
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Distribución de residuos por modelo y target', fontsize=14, fontweight='bold')

for col, nombre in enumerate(nombres):
    for row, target in enumerate(targets):
        ax       = axes[row, col]
        y_real   = y_test[target].to_numpy()
        y_pred   = resultados[nombre]['metricas'][target]['y_pred']
        residuos = y_real - y_pred
        mae      = resultados[nombre]['metricas'][target]['mae']

        ax.scatter(y_pred, residuos, alpha=0.4, s=16,
                   color=COLORES[nombre], edgecolors='none')
        ax.axhline(0, color='white', lw=1.2, ls='--', alpha=0.6)
        ax.set_title(f'{nombre}\n{target}  (MAE={mae:.2f})', fontsize=10)
        ax.set_xlabel('Predicho'); ax.set_ylabel('Residuo')
        ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('modulo1/research/graficos/residuos.png', dpi=130, bbox_inches='tight')
print('Guardado: modulo1/research/graficos/residuos.png')

# r^2 cross-validation
fig, ax = plt.subplots(figsize=(7, 5))
r2_cv_vals = [resultados[n]['r2_cv'] for n in nombres]
bars = ax.bar(nombres, r2_cv_vals,
              color=[COLORES[n] for n in nombres], alpha=0.85, width=0.5)

for bar, val in zip(bars, r2_cv_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

ax.set_title('R² medio — Cross-validation (5-fold)', fontsize=13, fontweight='bold')
ax.set_ylabel('R²')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('modulo1/research/graficos/cross_validation.png', dpi=130, bbox_inches='tight')
print('Guardado: modulo1/research/graficos/cross_validation.png')

# iteramos por target
for i, target in enumerate(targets):
    
    # grafica por target
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    titulo = 'Bienestar Físico' if target == 'bienestar_fisico' else 'Bienestar Mental'
    fig.suptitle(f'Importancia de Características para: {titulo}', fontsize=16, fontweight='bold')

    def custom_scorer(estimator, X, y_true, index=i):
        y_pred = estimator.predict(X)
        if isinstance(y_true, pd.DataFrame):
            y_true_target = y_true.iloc[:, index]
        else:
            y_true_target = y_true[:, index]
            
        return r2_score(y_true_target, y_pred[:, index])
    
    for ax, nombre in zip(axes, nombres):
        pipe = resultados[nombre]['pipe']
        
        resultado_importancia = permutation_importance(
            pipe, x_test, y_test, scoring=custom_scorer, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        importancias = resultado_importancia.importances_mean
        indices_ordenados = np.argsort(importancias)
        
        features_ordenadas = [features[j] for j in indices_ordenados]
        valores_ordenados = importancias[indices_ordenados]
        
        ax.barh(range(len(indices_ordenados)), valores_ordenados, color=COLORES[nombre], alpha=0.85)
        ax.set_yticks(range(len(indices_ordenados)))
        ax.set_yticklabels(features_ordenadas, fontsize=9)
        ax.set_title(f'{nombre}', fontsize=12)
        ax.set_xlabel('Caída en R² (Importancia)')
        
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardamos cada gráfico con un nombre distinto
    ruta_guardado = f'modulo1/research/graficos/importancia_features_{target}.png'
    plt.savefig(ruta_guardado, dpi=130, bbox_inches='tight')
    print(f'Guardado: {ruta_guardado}')