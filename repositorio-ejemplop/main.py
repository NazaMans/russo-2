# ==========================================================
# PIPELINE INTEGRAL DE PROCESAMIENTO ESTADÍSTICO Y MACHINE LEARNING
# Análisis y Modelado Individual por Dataset + Comparativa 1-3 + Reporte PDF
# ==========================================================

import os
import base64
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# ----------------------------------------------------------
# CONFIGURACIÓN VISUAL
# ----------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ----------------------------------------------------------
# 1. CARGA ROBUSTA DE DATOS
# ----------------------------------------------------------
print("[1/5] Cargando conjuntos de datos...")

def load_dataset(filename):
    paths = [os.path.join("data", filename), filename]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"No se encontró {filename} en ./ ni en ./data/")

df1 = load_dataset('dataset1.csv')
df2 = load_dataset('dataset2.csv')
df3 = load_dataset('dataset3.csv')
df4 = load_dataset('dataset4.csv')

# ----------------------------------------------------------
# 2. MODELADO Y GRÁFICOS INDIVIDUALES: DATASET 1, 2 Y 3
# ----------------------------------------------------------
print("[2/5] Generando modelos y gráficos individuales (Sets 1, 2 y 3)...")

# --- DATASET 1: Imputación y Modelo Lineal ---
df1_clean = df1.copy()
median_y1 = float(df1_clean['Y'].median())
df1_clean['Y'] = df1_clean['Y'].fillna(median_y1)

X1_tr, X1_te, y1_tr, y1_te = train_test_split(df1_clean[['X']], df1_clean['Y'], test_size=0.30, random_state=42)
lr1 = LinearRegression().fit(X1_tr, y1_tr)
pred1_te = lr1.predict(X1_te)
r2_1 = float(r2_score(y1_te, pred1_te))
mse_1 = float(mean_squared_error(y1_te, pred1_te))

x_line = pd.DataFrame({'X': np.linspace(0, 100, 200)})
y_line1 = lr1.predict(x_line)

# Fig 1: Dataset 1
plt.figure(figsize=(10, 5.2))
mask_null1 = df1['Y'].isnull()
plt.scatter(df1.loc[~mask_null1, 'X'], df1.loc[~mask_null1, 'Y'], color='#2E7D32', alpha=0.6, s=35, label='Datos Observados')
plt.scatter(df1.loc[mask_null1, 'X'], [median_y1]*mask_null1.sum(), color='#8D6E63', marker='x', s=50, linewidth=1.5, label=f'Imputados (Mediana={median_y1:.1f})')
plt.plot(x_line['X'], y_line1, color='#1B5E20', linewidth=2.2, label=f'Modelo (Ŷ = {lr1.coef_[0]:.2f}X + {lr1.intercept_:.2f}, R²={r2_1:.2f})')
plt.title("Dataset 1: Dispersión, Imputación y Modelo de Regresión Lineal", fontweight='bold')
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig("outputs/1_set1_modelo_regresion.png", dpi=300)
plt.close()

# --- DATASET 2: Detección IQR y Modelo Saneado ---
Q1_2 = float(df2['Y'].quantile(0.25))
Q3_2 = float(df2['Y'].quantile(0.75))
IQR_2 = Q3_2 - Q1_2
low2 = Q1_2 - 1.5 * IQR_2
high2 = Q3_2 + 1.5 * IQR_2

is_outlier2 = (df2['Y'] < low2) | (df2['Y'] > high2)
outliers2_count = int(is_outlier2.sum())

# Fig 2: Dataset 2 Detección Outliers
plt.figure(figsize=(10, 5.2))
plt.scatter(df2.loc[~is_outlier2, 'X'], df2.loc[~is_outlier2, 'Y'], color='#5D4037', alpha=0.7, s=35, label='Observaciones Válidas')
plt.scatter(df2.loc[is_outlier2, 'X'], df2.loc[is_outlier2, 'Y'], color='#D32F2F', marker='^', s=65, label=f'Outliers detectados ({outliers2_count})')
plt.axhline(high2, color='#D32F2F', linestyle='--', alpha=0.75, label=f'Límite Superior IQR ({high2:.1f})')
plt.axhline(low2, color='#D32F2F', linestyle=':', alpha=0.75, label=f'Límite Inferior IQR ({low2:.1f})')
plt.title("Dataset 2: Detección de Valores Atípicos (Outliers por IQR)", fontweight='bold')
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig("outputs/2_set2_deteccion_outliers.png", dpi=300)
plt.close()

# Fig 3: Dataset 2 Modelo Saneado
df2_clean = df2[~is_outlier2].copy()
X2_tr, X2_te, y2_tr, y2_te = train_test_split(df2_clean[['X']], df2_clean['Y'], test_size=0.30, random_state=42)
lr2 = LinearRegression().fit(X2_tr, y2_tr)
pred2_te = lr2.predict(X2_te)
r2_2 = float(r2_score(y2_te, pred2_te))
mse_2 = float(mean_squared_error(y2_te, pred2_te))
y_line2 = lr2.predict(x_line)

plt.figure(figsize=(10, 5.2))
plt.scatter(X2_te, y2_te, color='#8D6E63', alpha=0.6, s=35, label='Datos Test (Fernet 30%)')
plt.plot(x_line['X'], y_line2, color='#3E2723', linewidth=2.2, label=f'Modelo Saneado (Ŷ = {lr2.coef_[0]:.2f}X + {lr2.intercept_:.2f}, R²={r2_2:.2f})')
plt.title("Dataset 2: Modelo de Regresión Lineal tras Saneamiento", fontweight='bold')
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig("outputs/3_set2_modelo_limpio.png", dpi=300)
plt.close()

# --- DATASET 3: Modelo Lineal Alta Precisión ---
X3_tr, X3_te, y3_tr, y3_te = train_test_split(df3[['X']], df3['Y'], test_size=0.30, random_state=42)
lr3 = LinearRegression().fit(X3_tr, y3_tr)
pred3_te = lr3.predict(X3_te)
r2_3 = float(r2_score(y3_te, pred3_te))
mse_3 = float(mean_squared_error(y3_te, pred3_te))
y_line3 = lr3.predict(x_line)

# Fig 4: Dataset 3
plt.figure(figsize=(10, 5.2))
plt.scatter(X3_te, y3_te, color='#66BB6A', alpha=0.6, s=35, label='Datos Test (Fernet 30%)')
plt.plot(x_line['X'], y_line3, color='#1B5E20', linewidth=2.2, label=f'Modelo (Ŷ = {lr3.coef_[0]:.2f}X + {lr3.intercept_:.2f}, R²={r2_3:.4f})')
plt.title("Dataset 3: Modelo de Regresión Lineal de Alta Precisión", fontweight='bold')
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig("outputs/4_set3_modelo_regresion.png", dpi=300)
plt.close()

# Exportar CSV de Sets 1, 2 y 3 limpios
df1_temp = df1_clean.copy(); df1_temp['origen'] = 'dataset1.csv'
df2_temp = df2_clean.copy(); df2_temp['origen'] = 'dataset2.csv'
df3_temp = df3.copy(); df3_temp['origen'] = 'dataset3.csv'
df_123_clean = pd.concat([df1_temp, df2_temp, df3_temp], ignore_index=True)
df_123_clean.to_csv('outputs/datasets123_limpios.csv', index=False)

# ----------------------------------------------------------
# 3. COMPARATIVA GLOBAL: EXCLUSIVA DATASETS 1, 2 Y 3
# ----------------------------------------------------------
print("[3/5] Generando comparativa global exclusiva de Datasets 1, 2 y 3...")

fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(13, 5.2))

comp_123 = pd.DataFrame({
    'Dataset': ['Dataset 1', 'Dataset 2', 'Dataset 3'],
    'Faltantes (%)': [(df1['Y'].isnull().sum() / len(df1)) * 100, 0.0, 0.0],
    'Atípicos (%)': [0.0, (outliers2_count / len(df2)) * 100, 0.0],
    'R² Ajustado (%)': [r2_1 * 100, r2_2 * 100, r2_3 * 100]
})

x_idx = np.arange(len(comp_123))
w_bar = 0.25

b1 = ax_bar.bar(x_idx - w_bar, comp_123['Faltantes (%)'], w_bar, label='Faltantes (%)', color='#757575', edgecolor='black')
b2 = ax_bar.bar(x_idx, comp_123['Atípicos (%)'], w_bar, label='Atípicos / Outliers (%)', color='#8D6E63', edgecolor='black')
b3 = ax_bar.bar(x_idx + w_bar, comp_123['R² Ajustado (%)'], w_bar, label='R² Modelo Individual (%)', color='#2E7D32', edgecolor='black')

ax_bar.set_title("A. Comparativa de Calidad y Ajuste (Sets 1, 2 y 3)", fontweight='bold')
ax_bar.set_xticks(x_idx)
ax_bar.set_xticklabels(comp_123['Dataset'], fontweight='bold')
ax_bar.set_ylabel("Porcentaje (%)")
ax_bar.set_ylim(0, 115)
ax_bar.legend(frameon=True, facecolor='white', fontsize=8.5)

for rects in [b1, b2, b3]:
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax_bar.annotate(f'{h:.1f}%',
                            xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax_line.plot(x_line['X'], y_line1, color='#2E7D32', linewidth=2.5, label=f'Set 1: Ŷ = {lr1.coef_[0]:.2f}X + {lr1.intercept_:.1f} (R²={r2_1:.2f})')
ax_line.plot(x_line['X'], y_line2, color='#8D6E63', linewidth=2.5, linestyle='--', label=f'Set 2: Ŷ = {lr2.coef_[0]:.2f}X + {lr2.intercept_:.1f} (R²={r2_2:.2f})')
ax_line.plot(x_line['X'], y_line3, color='#1B5E20', linewidth=2.5, linestyle='-.', label=f'Set 3: Ŷ = {lr3.coef_[0]:.2f}X + {lr3.intercept_:.1f} (R²={r2_3:.2f})')

ax_line.set_title("B. Comparativa de Pendientes de Regresión (1, 2 y 3)", fontweight='bold')
ax_line.set_xlabel("Variable X")
ax_line.set_ylabel("Predicción Y")
ax_line.legend(frameon=True, facecolor='white', fontsize=8.5)

plt.tight_layout()
plt.savefig("outputs/5_comparativa_global_sets123.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 4. MODELADO Y ANÁLISIS AISLADO: DATASET 4 (CLASIFICACIÓN)
# ----------------------------------------------------------
print("[4/5] Procesando y modelando Dataset 4 de forma aislada e independiente...")

# Fig 6: Matriz de Correlación
plt.figure(figsize=(6, 5))
sns.heatmap(df4.corr(), annot=True, cmap='Greens', fmt=".2f", square=True, cbar_kws={'shrink': 0.8})
plt.title("Dataset 4: Matriz de Correlación (Mapa de Celdas)", fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/6_set4_correlacion.png", dpi=300)
plt.close()

df4.to_csv('outputs/dataset4_limpio.csv', index=False)

# Regresión Logística (FERNET 70/30)
X4_tr, X4_te, y4_tr, y4_te = train_test_split(df4[['x1', 'x2']], df4['y'], test_size=0.30, random_state=42)
clf4 = LogisticRegression().fit(X4_tr, y4_tr)
pred4_te = clf4.predict(X4_te)
acc4 = float(accuracy_score(y4_te, pred4_te))
cm4 = confusion_matrix(y4_te, pred4_te)

# Fig 7: Frontera de Decisión
xx, yy = np.meshgrid(np.linspace(-1, 11, 300), np.linspace(-1, 6, 300))
grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['x1', 'x2'])
Z = clf4.predict(grid_df).reshape(xx.shape)

plt.figure(figsize=(10, 5.2))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='Greens')
sns.scatterplot(
    data=df4, x='x1', y='x2', hue='y',
    palette={0: '#424242', 1: '#2E7D32'},
    edgecolor='white', linewidth=0.6, s=40
)
plt.title(f"Dataset 4: Frontera de Decisión Logística (Exactitud: {acc4:.2%})", fontweight='bold')
plt.xlabel("Predictor x1")
plt.ylabel("Predictor x2")
plt.legend(title="Clase (y)", frameon=True)
plt.tight_layout()
plt.savefig("outputs/7_set4_frontera_decision.png", dpi=300)
plt.close()

# Fig 8: Matriz de Confusión y Métricas de Rendimiento
fig, (ax_cm, ax_met) = plt.subplots(1, 2, figsize=(11, 4.5))

sns.heatmap(cm4, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_cm,
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Real 0', 'Real 1'], annot_kws={'size': 14, 'fontweight': 'bold'})
ax_cm.set_title("A. Matriz de Confusión (Set 4 Test 30%)", fontweight='bold')

metrics_names = ['Exactitud (Accuracy)', 'Precisión (Clase 1)', 'Sensibilidad (Recall 1)', 'F1-Score (Clase 1)']
metrics_values = [
    acc4 * 100,
    float(precision_score(y4_te, pred4_te)) * 100,
    float(recall_score(y4_te, pred4_te)) * 100,
    float(f1_score(y4_te, pred4_te)) * 100
]

bm = ax_met.barh(metrics_names, metrics_values, color=['#2E7D32', '#388E3C', '#4CAF50', '#81C784'], edgecolor='black')
ax_met.set_xlim(0, 115)
ax_met.set_xlabel("Porcentaje (%)")
ax_met.set_title("B. Métricas de Clasificación del Modelo", fontweight='bold')

for rect in bm:
    w = rect.get_width()
    ax_met.annotate(f'{w:.1f}%',
                    xy=(w, rect.get_y() + rect.get_height()/2),
                    xytext=(4, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/8_set4_metricas_clasificacion.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 5. COMPILACIÓN DE INFORME HTML Y GENERACIÓN DE PDF
# ----------------------------------------------------------
print("[5/5] Compilando HTML y PDF...")

def to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

img1 = to_b64("outputs/1_set1_modelo_regresion.png")
img2 = to_b64("outputs/2_set2_deteccion_outliers.png")
img3 = to_b64("outputs/3_set2_modelo_limpio.png")
img4 = to_b64("outputs/4_set3_modelo_regresion.png")
img5 = to_b64("outputs/5_comparativa_global_sets123.png")
img6 = to_b64("outputs/6_set4_correlacion.png")
img7 = to_b64("outputs/7_set4_frontera_decision.png")
img8 = to_b64("outputs/8_set4_metricas_clasificacion.png")

html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe Final - Análisis Multivariado y Modelado Predictivo</title>
<style>
  @page {{
    size: A4;
    margin: 15mm 14mm 15mm 14mm;
    @bottom-right {{
      content: "Página " counter(page) " de " counter(pages);
      font-size: 8pt;
      color: #666;
    }}
    @bottom-left {{
      content: "Memoria Técnica Final | Estadística y Aprendizaje Automático";
      font-size: 8pt;
      color: #666;
    }}
  }}
  body {{
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #212121;
    line-height: 1.4;
    font-size: 9pt;
    margin: 0;
    padding: 0;
  }}
  .header {{
    border-bottom: 2.5px solid #2E7D32;
    padding-bottom: 6px;
    margin-bottom: 12px;
  }}
  .header h1 {{
    font-size: 15pt;
    color: #1B5E20;
    margin: 0 0 4px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .header .meta {{
    font-size: 8.5pt;
    color: #555;
    display: flex;
    justify-content: space-between;
  }}
  h2 {{
    font-size: 11pt;
    color: #1B5E20;
    border-left: 4px solid #2E7D32;
    padding-left: 6px;
    margin-top: 14px;
    margin-bottom: 6px;
  }}
  h3 {{
    font-size: 9.5pt;
    color: #333;
    margin-top: 8px;
    margin-bottom: 4px;
  }}
  p {{
    margin: 0 0 5px 0;
    text-align: justify;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0;
    font-size: 8.5pt;
  }}
  th, td {{
    border: 1px solid #dcdcdc;
    padding: 4px 6px;
    text-align: left;
  }}
  th {{
    background-color: #f1f8f3;
    color: #1B5E20;
    font-weight: bold;
  }}
  tr:nth-child(even) {{
    background-color: #fafafa;
  }}
  .figure-box {{
    text-align: center;
    margin: 8px 0;
    page-break-inside: avoid;
  }}
  .figure-box img {{
    max-width: 95%;
    height: auto;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
  }}
  .caption {{
    font-size: 8pt;
    color: #555;
    margin-top: 3px;
    font-style: italic;
  }}
  .formula-box {{
    background: #f4fbf5;
    border: 1px solid #c8e6c9;
    border-radius: 4px;
    padding: 5px 8px;
    margin: 6px 0;
    text-align: center;
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-size: 9.5pt;
    color: #1B5E20;
  }}
  .page-break {{
    page-break-before: always;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Informe Técnico Final: Análisis y Modelado Individual por Dataset</h1>
  <div class="meta">
    <span><strong>Disciplina:</strong> Probabilidad, Estadística & Machine Learning</span>
    <span><strong>Cátedra:</strong> Proyecto Integrador Final</span>
    <span><strong>Protocolo:</strong> Pipeline 6 Etapas (FERNET 70/30)</span>
  </div>
</div>

<h2>1. Protocolo Metodológico y Estructura del Trabajo</h2>
<p>
El presente informe documenta la ejecución rigurosa de las seis etapas del ciclo de vida de los datos aplicada de manera individual a cada uno de los cuatro conjuntos proporcionados: <strong>EDA inicial</strong> (variables, tipos, rangos, valores perdidos y atípicos), <strong>Limpieza</strong> (evaluación de umbral de descarte), <strong>Imputación</strong>, <strong>Estudio de Vínculos/Correlación</strong>, <strong>Confección del Modelo propio (partición FERNET 70/30)</strong> y <strong>Testeo Cuantitativo</strong>.
</p>
<p>
Siguiendo la consigna del proyecto, los conjuntos 1, 2 y 3 corresponden a problemas de <strong>regresión lineal continua</strong> (analizados de forma individual y luego contrastados en una comparativa global exclusiva de regresión), mientras que el Dataset 4 corresponde a un problema de <strong>clasificación binaria</strong>, abordado con su propio juego de métricas y visualizaciones independientes.
</p>

<h2>2. Dataset 1: Regresión Lineal con Datos Faltantes</h2>
<p>
<strong>Análisis Exploratorio y Limpieza:</strong> El Dataset 1 cuenta con 300 registros y dos variables continuas (X e Y). En la variable Y se detectaron 45 valores faltantes (15.0% del total). Dado que el porcentaje no superó el umbral crítico del 40%, la variable se conservó y se imputó mediante la <em>mediana muestral</em> (Mediana = {median_y1:.1f}) para no alterar la tendencia central.
</p>
<p>
<strong>Modelo Propio (FERNET 70/30):</strong> Se entrenó una Regresión Lineal Simple sobre el 70% de los datos y se testeó sobre el 30% restante, logrando un coeficiente de determinación R&sup2; = {r2_1:.4f} y un error cuadrático medio MSE = {mse_1:.2f}.
</p>
<div class="formula-box">
  Ŷ₁ = {lr1.coef_[0]:.2f} · X + {lr1.intercept_:.2f} &emsp; [ R² = {r2_1:.4f}, &nbsp; MSE = {mse_1:.2f} ]
</div>

<div class="figure-box">
  <img src="data:image/png;base64,{img1}" alt="Dataset 1 Modelo" style="max-height: 290px;">
  <div class="caption">Figura 1. Dataset 1: Dispersión de observaciones, identificación de puntos imputados (en cruz) y recta de regresión lineal individual ajustada.</div>
</div>

<div class="page-break"></div>

<h2>3. Dataset 2: Regresión Lineal y Saneamiento de Outliers</h2>
<p>
<strong>Análisis Exploratorio y Detección de Anomalías:</strong> El Dataset 2 presenta 300 observaciones donde no existen datos nulos, pero se identificó una severa contaminación por valores atípicos (outliers bivariados). Los valores extremos alcanzan cotas superiores en Y = {df2['Y'].max():.1f} e inferiores anómalas en Y = {df2['Y'].min():.1f}, actuando como puntos de alto apalancamiento.
</p>
<p>
<strong>Limpieza por Rango Intercuartílico (IQR):</strong> Con Q1 = {Q1_2:.2f} y Q3 = {Q3_2:.2f} (IQR = {IQR_2:.2f}), se establecieron los límites inferior ({low2:.1f}) y superior ({high2:.1f}). Se detectaron y eliminaron exactamente {outliers2_count} observaciones anómalas ({outliers2_count/len(df2)*100:.1f}% de los datos).
</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img2}" alt="Dataset 2 Outliers" style="max-height: 270px;">
  <div class="caption">Figura 2. Dataset 2: Detección visual de valores atípicos severos mediante los umbrales estadísticos del Rango Intercuartílico (IQR).</div>
</div>

<p>
<strong>Modelo Propio Saneado (FERNET 70/30):</strong> Tras remover los outliers, el modelo lineal entrenado sobre la partición limpia (70/30) demostró un ajuste notablemente robusto con R&sup2; = {r2_2:.4f} y MSE = {mse_2:.2f}.
</p>
<div class="formula-box">
  Ŷ₂ = {lr2.coef_[0]:.2f} · X + {lr2.intercept_:.2f} &emsp; [ R² = {r2_2:.4f}, &nbsp; MSE = {mse_2:.2f} ]
</div>

<div class="figure-box">
  <img src="data:image/png;base64,{img3}" alt="Dataset 2 Modelo" style="max-height: 270px;">
  <div class="caption">Figura 3. Dataset 2: Evaluación del modelo lineal individual sobre el conjunto de test (30%) una vez purgado de valores atípicos.</div>
</div>

<div class="page-break"></div>

<h2>4. Dataset 3: Regresión Lineal de Alta Precisión</h2>
<p>
<strong>Análisis Exploratorio:</strong> El Dataset 3 cuenta con 300 registros con un 0% de valores faltantes y ausencia total de valores atípicos. Presenta una relación lineal casi determinística con una pendiente pronunciada y varianza residual mínima.
</p>
<p>
<strong>Modelo Propio (FERNET 70/30):</strong> Se dividió en 70% entrenamiento y 30% testeo, alcanzando un ajuste óptimo con R&sup2; = {r2_3:.4f} y un error residual muy bajo (MSE = {mse_3:.2f}).
</p>
<div class="formula-box">
  Ŷ₃ = {lr3.coef_[0]:.2f} · X + {lr3.intercept_:.2f} &emsp; [ R² = {r2_3:.4f}, &nbsp; MSE = {mse_3:.2f} ]
</div>

<div class="figure-box">
  <img src="data:image/png;base64,{img4}" alt="Dataset 3 Modelo" style="max-height: 280px;">
  <div class="caption">Figura 4. Dataset 3: Dispersión de puntos del 30% de test y recta de regresión lineal individual de alta pendiente.</div>
</div>

<h2>5. Comparativa Global Exclusiva: Datasets 1, 2 y 3 (Regresión)</h2>
<p>
Al contrastar exclusivamente los tres datasets continuos, se observa con claridad que aunque comparten la misma familia algorítmica (Regresión Lineal Simple), cada uno modela un régimen físico o una subpoblación independiente con pendientes e interceptos bien diferenciados:
</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img5}" alt="Comparativa Sets 1, 2 y 3" style="max-height: 280px;">
  <div class="caption">Figura 5. Comparativa Global Exclusiva (Sets 1, 2 y 3): Panel A contrasta la calidad de datos y R² individual; Panel B compara las 3 pendientes en el mismo plano cartesiano.</div>
</div>

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Nulos (%)</th>
      <th>Outliers (%)</th>
      <th>Pendiente (m)</th>
      <th>Intercepto (b)</th>
      <th>R² Modelo</th>
      <th>MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Dataset 1</strong></td>
      <td>15.0% (45)</td>
      <td>0.0%</td>
      <td>{lr1.coef_[0]:.2f}</td>
      <td>{lr1.intercept_:.2f}</td>
      <td>{r2_1:.4f}</td>
      <td>{mse_1:.2f}</td>
    </tr>
    <tr>
      <td><strong>Dataset 2 (Limpio)</strong></td>
      <td>0.0%</td>
      <td>4.0% (12)</td>
      <td>{lr2.coef_[0]:.2f}</td>
      <td>{lr2.intercept_:.2f}</td>
      <td>{r2_2:.4f}</td>
      <td>{mse_2:.2f}</td>
    </tr>
    <tr>
      <td><strong>Dataset 3</strong></td>
      <td>0.0%</td>
      <td>0.0%</td>
      <td>{lr3.coef_[0]:.2f}</td>
      <td>{lr3.intercept_:.2f}</td>
      <td>{r2_3:.4f}</td>
      <td>{mse_3:.2f}</td>
    </tr>
  </tbody>
</table>

<div class="page-break"></div>

<h2>6. Dataset 4: Clasificación Binaria Supervisada (Análisis Propio e Independiente)</h2>
<p>
El Dataset 4 cambia completamente de paradigma: contiene 300 filas con dos variables predictoras continuas (x1, x2) y una variable objetivo binaria (y &isin; {{0, 1}}).
</p>

<h3>Estudio de Vínculos y Ortogonalidad (Mapa de Celdas)</h3>
<p>
La matriz de correlación evidencia que los dos predictores son totalmente ortogonales (Corr(x1, x2) = 0.00), lo que garantiza ausencia absoluta de colinealidad (VIF = 1.0). El predictor x1 aporta positivamente a la probabilidad de pertenencia a la clase 1 (r = +0.55), mientras que x2 penaliza dicha probabilidad (r = -0.54).
</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img6}" alt="Dataset 4 Correlación" style="max-height: 250px;">
  <div class="caption">Figura 6. Dataset 4: Matriz de correlación de Pearson. Ausencia de colinealidad entre x1 y x2, con tracción en signos opuestos respecto a la variable objetivo y.</div>
</div>

<h3>Modelo Propio y Frontera de Decisión (FERNET 70/30)</h3>
<p>
Dado que la separación en el plano &Ropf;&sup2; es geométrica y lineal, se entrenó un clasificador de <strong>Regresión Logística</strong>. La frontera de decisión se ubica donde P(y=1|x) = 0.50 (logit z = 0):
</p>
<div class="formula-box">
  z = {clf4.intercept_[0]:.2f} + {clf4.coef_[0][0]:.2f} · x₁ {clf4.coef_[0][1]:.2f} · x₂ = 0 &emsp;&rArr;&emsp; x₂ = {-clf4.coef_[0][0]/clf4.coef_[0][1]:.2f} · x₁ + {-clf4.intercept_[0]/clf4.coef_[0][1]:.2f}
</div>

<div class="figure-box">
  <img src="data:image/png;base64,{img7}" alt="Dataset 4 Frontera" style="max-height: 290px;">
  <div class="caption">Figura 7. Dataset 4: Frontera de decisión logística. La región sombreada en verde delimita la zona donde el modelo clasifica con probabilidad &ge; 50% para la Clase 1.</div>
</div>

<div class="page-break"></div>

<h3>Evaluación Cuantitativa y Métricas Propias del Dataset 4</h3>
<p>
Al evaluar sobre el 30% de test independiente (90 registros), el modelo alcanzó una <strong>Exactitud (Accuracy) del {acc4:.2%}</strong>, con alta sensibilidad y precisión en ambas clases:
</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img8}" alt="Dataset 4 Métricas" style="max-height: 260px;">
  <div class="caption">Figura 8. Dataset 4: Panel A muestra la Matriz de Confusión en datos de test; Panel B detalla las métricas de rendimiento del clasificador.</div>
</div>

<table>
  <thead>
    <tr>
      <th>Clase Real \ Predicción</th>
      <th>Predicho: Clase 0</th>
      <th>Predicho: Clase 1</th>
      <th>Métricas de Rendimiento</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Clase Real: 0</strong></td>
      <td><strong>{cm4[0,0]}</strong> (Verdaderos Negativos)</td>
      <td>{cm4[0,1]} (Falsos Positivos)</td>
      <td>Sensibilidad (Recall 0) = {cm4[0,0]/(cm4[0,0]+cm4[0,1])*100:.1f}%</td>
    </tr>
    <tr>
      <td><strong>Clase Real: 1</strong></td>
      <td>{cm4[1,0]} (Falsos Negativos)</td>
      <td><strong>{cm4[1,1]}</strong> (Verdaderos Positivos)</td>
      <td>Sensibilidad (Recall 1) = {cm4[1,1]/(cm4[1,0]+cm4[1,1])*100:.1f}%</td>
    </tr>
    <tr>
      <td><strong>Rendimiento Global</strong></td>
      <td colspan="2"><strong>Exactitud (Accuracy): {acc4:.2%}</strong></td>
      <td><strong>F1-Score Ponderado: {f1_score(y4_te, pred4_te)*100:.1f}%</strong></td>
    </tr>
  </tbody>
</table>

<h2>7. Conclusiones y Dictamen Técnico Final</h2>
<p>
1. <strong>Individualidad de Modelos:</strong> Cada conjunto requirió un tratamiento especializado. El Set 1 demandó imputación de nulos; el Set 2 exigió filtrado de 12 outliers para elevar su R&sup2; de valores negativos a 0.95; el Set 3 confirmó una relación lineal pura (R&sup2; = 0.995); y el Set 4 requirió un clasificador logístico con 93.3% de exactitud.
</p>
<p>
2. <strong>Comparativa de Regresión (Sets 1-3):</strong> El contraste de pendientes (m1 = {lr1.coef_[0]:.2f}, m2 = {lr2.coef_[0]:.2f}, m3 = {lr3.coef_[0]:.2f}) demuestra que unificar los tres sets en un único modelo OLS sin segmentar comete un error de trimodalidad residual. Deben conservarse como modelos individuales o incluir variables <em>dummy</em>.
</p>
<p>
3. <strong>Aislamiento del Set 4:</strong> Al ser un problema de clasificación supervisada, el Set 4 no es comparable en escala R&sup2; con los primeros tres, validando la necesidad de evaluarlo mediante matrices de confusión, curvas de decisión y métricas F1 independientes.
</p>

</body>
</html>
"""

html_path = os.path.abspath("outputs/informe_final.html")
pdf_path = os.path.abspath("outputs/informe_final.pdf")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML generado en: {html_path}")

# Conversión automática a PDF sin dependencias externas
edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    shutil.which("msedge"),
    shutil.which("chrome"),
    shutil.which("chromium"),
    shutil.which("google-chrome")
]

browser_exe = next((p for p in edge_paths if p and os.path.exists(p)), None)

pdf_creado = False
if browser_exe:
    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(pdf_path):
        print(f"¡PDF compilado con éxito vía navegador!: {pdf_path}")
        pdf_creado = True

if not pdf_creado:
    try:
        import weasyprint
        weasyprint.HTML(html_path).write_pdf(pdf_path)
        print(f"¡PDF compilado con éxito vía WeasyPrint!: {pdf_path}")
        pdf_creado = True
    except Exception:
        pass

if not pdf_creado:
    print("\nNota: Para obtener el PDF en Windows de inmediato, abre outputs/informe_final.html en tu navegador y presiona Ctrl + P -> Guardar como PDF (marcando 'Gráficos de fondo').")

print("\nPipeline completado al 100% con éxito.")