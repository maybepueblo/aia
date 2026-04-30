import pandas as pd
import numpy as np

def generar_dataset_causal(num_samples=1500, random_state=42):
    np.random.seed(random_state)
    
    # ---------------------------------------------------------
    # 1. CAPA BASE: Demografía y Hábitos Independientes
    # ---------------------------------------------------------
    edad = np.random.uniform(18, 65, num_samples)
    sexo = np.random.binomial(1, 0.5, num_samples) # 0 o 1
    altura = np.random.normal(1.70, 0.08, num_samples)
    peso = np.random.normal(70, 15, num_samples)
    bmi = peso / (altura ** 2)
    
    # Hábitos sueltos
    t_meditacion = np.random.exponential(15, num_samples).clip(0, 60)
    h_social = np.random.normal(5, 2, num_samples).clip(0, 15)
    h_movil = np.random.normal(4, 1.5, num_samples).clip(0.5, 10)
    t_pantalla = (h_movil + np.random.normal(2, 1, num_samples)).clip(1, 12)
    h_sol = np.random.normal(1.5, 1, num_samples).clip(0, 5)
    
    # Nutrición
    alim_enteros = np.random.normal(6, 2, num_samples).clip(1, 10)
    alim_procesados = (10 - alim_enteros + np.random.normal(0, 1, num_samples)).clip(0, 10)
    g_proteina = np.random.normal(90, 20, num_samples).clip(40, 200)
    g_fibra = np.random.normal(25, 8, num_samples).clip(5, 50)
    l_agua = np.random.normal(2.0, 0.6, num_samples).clip(0.5, 5)
    n_alcohol = np.random.exponential(2, num_samples).clip(0, 10)

    # ---------------------------------------------------------
    # 2. CAPA MEDIADORA: Variables Causales (Dependen de la Capa Base)
    # ---------------------------------------------------------
    # Estrés: Aumenta con pantallas, disminuye con socialización y meditación
    base_estres = np.random.normal(5, 1.5, num_samples)
    n_estres = base_estres + (t_pantalla * 0.3) - (h_social * 0.2) - (t_meditacion * 0.05)
    n_estres = n_estres.clip(1, 10)

    # Ejercicio: Penalizado si hay muchísimo estrés o BMI muy alto
    base_ejercicio = np.random.normal(45, 30, num_samples)
    penalizacion_bmi = np.where(bmi > 30, 15, 0)
    m_ejercicio = (base_ejercicio - (n_estres * 2) - penalizacion_bmi).clip(0, 150)
    i_ejercicio = np.where(m_ejercicio > 0, np.random.normal(3, 1, num_samples), 1).clip(1, 5)
    
    pasos = 3000 + (m_ejercicio * 60) + np.random.normal(1000, 500, num_samples)
    pasos = pasos.clip(1000, 25000)

    # Sueño: Afectado por estrés, alcohol y pantallas. Mejorado levemente por ejercicio
    h_sueno = 8.5 - (n_estres * 0.25) - (t_pantalla * 0.15) - (n_alcohol * 0.2) + (m_ejercicio * 0.01)
    h_sueno = (h_sueno + np.random.normal(0, 0.5, num_samples)).clip(3, 10)
    c_sueno = (h_sueno * 0.8 - n_alcohol * 0.3 + np.random.normal(1, 0.5, num_samples)).clip(1, 10)

    # ---------------------------------------------------------
    # 3. CAPA DE RESULTADOS (TARGETS): Con no-linealidades y ciclos
    # ---------------------------------------------------------
    
    # Bienestar Físico
    # Fórmula base
    b_fisico = 40 + (m_ejercicio * 0.2) + (c_sueno * 3) + (alim_enteros * 2) - (alim_procesados * 1.5) + (l_agua * 2) - (n_alcohol * 2)
    # Interacciones No-Lineales (Penalización severa por mal sueño y obesidad)
    b_fisico = np.where(h_sueno < 5.5, b_fisico - 15, b_fisico) # Bottleneck del sueño
    b_fisico = np.where(bmi > 30, b_fisico - 10, b_fisico)
    b_fisico = (b_fisico + np.random.normal(0, 3, num_samples)).clip(0, 100) # Ruido natural

    # Bienestar Mental
    # Fórmula base
    b_mental = 50 - (n_estres * 4.5) + (c_sueno * 3.5) + (h_social * 2) + (t_meditacion * 0.3) + (h_sol * 2) - (t_pantalla * 1.5)
    # Vicious Cycle: Alto estrés + Poco sueño = Hundimiento del bienestar mental
    ciclo_vicioso = (n_estres > 7.5) & (h_sueno < 6.0)
    b_mental = np.where(ciclo_vicioso, b_mental - 20, b_mental)
    b_mental = (b_mental + np.random.normal(0, 4, num_samples)).clip(0, 100) # Ruido natural

    # ---------------------------------------------------------
    # 4. CREAR Y GUARDAR EL DATAFRAME
    # ---------------------------------------------------------
    # IMPORTANTE: Respetar el orden exacto de las features de tu app.py
    features = ['edad', 'sexo', 'altura', 'bmi', 'peso', 'h_sueno', 'c_sueno', 'n_estres', 'h_movil', 
                't_pantalla', 'pasos', 'i_ejercicio', 'm_ejercicio', 'alim_enteros', 'alim_procesados', 
                'g_proteina', 'g_fibra', 'h_social', 'h_sol', 'l_agua', 'n_alcohol', 't_meditacion']
    
    df = pd.DataFrame({
        'edad': np.round(edad),
        'sexo': sexo,
        'altura': np.round(altura, 2),
        'bmi': np.round(bmi, 2),
        'peso': np.round(peso, 2),
        'h_sueno': np.round(h_sueno, 1),
        'c_sueno': np.round(c_sueno, 1),
        'n_estres': np.round(n_estres, 1),
        'h_movil': np.round(h_movil, 1),
        't_pantalla': np.round(t_pantalla, 1),
        'pasos': np.round(pasos),
        'i_ejercicio': np.round(i_ejercicio, 1),
        'm_ejercicio': np.round(m_ejercicio),
        'alim_enteros': np.round(alim_enteros, 1),
        'alim_procesados': np.round(alim_procesados, 1),
        'g_proteina': np.round(g_proteina, 1),
        'g_fibra': np.round(g_fibra, 1),
        'h_social': np.round(h_social, 1),
        'h_sol': np.round(h_sol, 1),
        'l_agua': np.round(l_agua, 1),
        'n_alcohol': np.round(n_alcohol, 1),
        't_meditacion': np.round(t_meditacion),
        'bienestar_fisico': np.round(b_fisico, 2),
        'bienestar_mental': np.round(b_mental, 2)
    })
    
    # Reordenar para que encaje perfecto con tu app.py
    df = df[features + ['bienestar_fisico', 'bienestar_mental']]
    
    df.to_csv('modulo1/research/data/wellness_dataset.csv', index=False)
    print("¡Dataset causal generado con éxito! Guardado como 'wellness_dataset.csv'")
    print(f"Dimensiones: {df.shape}")
    return df

# Ejecutar la generación
if __name__ == "__main__":
    df_sintetico = generar_dataset_causal(1500)
    print("\nCorrelaciones con Bienestar Mental (deberían tener sentido lógico ahora):")
    print(df_sintetico.corr()['bienestar_mental'].sort_values(ascending=False).head(5))
    print(df_sintetico.corr()['bienestar_mental'].sort_values(ascending=True).head(5))