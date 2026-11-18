"""Genera un informe HTML autocontenido con diez secciones y resultados recalculados."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import pandas as pd

from analizar_ventas import ROOT, clean_sales, export_analysis
from modelar_ventas import evaluate_returns


def number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def figure(folder: Path, name: str, caption: str) -> str:
    file = folder / "graficos" / name
    content = base64.b64encode(file.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{content}" alt="{html.escape(caption)}">'
        f'<figcaption>{html.escape(caption)}</figcaption></figure>'
    )


def table(data: pd.DataFrame, fields: list[str], labels: list[str],
          formats: dict[str, int] | None = None) -> str:
    selected = data[fields].copy()
    for column, digits in (formats or {}).items():
        selected[column] = selected[column].map(
            lambda value: "—" if pd.isna(value) else number(float(value), digits)
        )
    selected.columns = labels
    return selected.to_html(index=False, border=0, classes="data-table", escape=True)


def compose(source: Path, folder: Path, audit: dict, kpi: dict, model: dict) -> str:
    read = lambda name: pd.read_csv(folder / f"{name}.csv")
    product = read("por_producto").sort_values("ingresos_verificados", ascending=False)
    category = read("por_categoria").sort_values("ingresos_verificados", ascending=False)
    region = read("por_region").sort_values("ingresos_verificados", ascending=False)
    channel = read("por_canal").sort_values("ingresos_verificados", ascending=False)
    payment = read("por_medio_pago").sort_values("ingresos_verificados", ascending=False)
    monthly = read("por_mes").sort_values("mes")
    year_month = read("por_anio_mes")
    correlations = pd.read_csv(folder / "correlaciones_pearson.csv", index_col=0)
    best_month = monthly.nlargest(1, "ingresos_verificados").iloc[0]
    worst_month = monthly.nsmallest(1, "ingresos_verificados").iloc[0]
    best_product = product.iloc[0]
    best_units = product.sort_values("unidades_validas", ascending=False).iloc[0]
    best_region = region.iloc[0]
    top_category = category.iloc[0]
    month_2025 = monthly.loc[monthly["mes"].str.startswith("2025-")]
    all_years = sorted(year_month["anio"].unique().tolist())
    full_years = [
        year for year in all_years
        if year_month.loc[year_month["anio"] == year, "mes_calendario"].nunique() == 12
    ]
    check = pd.read_csv(folder / "ventas_preparadas.csv", low_memory=False)
    return_by_product = product.sort_values("tasa_devolucion_pct", ascending=False).iloc[0]
    corr_delivery = correlations.loc["dias_entrega", "satisfaccion"]
    corr_age = correlations.loc["edad_cliente", "total_venta_validado"]
    corr_discount = correlations.loc["descuento_pct", "total_venta_validado"]
    fraud = audit["importes_inconsistentes"] + audit["importes_no_verificables"]
    unknown_region = int((check["region"] == "Sin dato").sum())
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if full_years:
        years_text = ", ".join(map(str, full_years))
    else:
        years_text = "ninguno"
    kpi_cards = [
        ("Operaciones preparadas", number(kpi["operaciones_preparadas"], 0)),
        ("Ingresos verificados", number(kpi["ingresos_verificados"])),
        ("Cobertura de ingresos", number(kpi["cobertura_ingresos_pct"], 2) + " %"),
        ("Devoluciones observadas", number(kpi["tasa_devolucion_pct"], 2) + " %"),
    ]
    cards = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in kpi_cards
    )
    quality = pd.DataFrame([
        ("Duplicados completos eliminados", audit["duplicados_exactos_eliminados"], "No se usan en indicadores."),
        ("Registros con ID repetido", audit["filas_con_id_repetido_conservadas"], "Se conservan para análisis; se excluyen del modelo."),
        ("Fechas inválidas", audit["fechas_invalidas"], "Se excluyen solo de series temporales y modelado."),
        ("Regiones faltantes", unknown_region, "Se agrupan como «Sin dato»."),
        ("Precios originalmente vacíos", audit["faltantes_originales"]["precio_unitario"], "No se imputan."),
        ("Precios textuales inválidos", audit["conversiones_numericas_invalidas"]["precio_unitario"], "Se convierten a faltantes."),
        ("Cantidades no positivas", audit["valores_fuera_de_rango"]["cantidad"], "Se convierten a faltantes."),
        ("Descuentos fuera de 0–1", audit["valores_fuera_de_rango"]["descuento_pct"], "Se convierten a faltantes."),
        ("Importes contradictorios", audit["importes_inconsistentes"], "No se incluyen en el subtotal verificado."),
        ("Importes no verificables", audit["importes_no_verificables"], "No se incluyen en el subtotal verificado."),
    ], columns=["Problema", "Casos", "Tratamiento"])
    model_metrics = pd.DataFrame([
        (
            "Clase mayoritaria",
            model["referencia"]["exactitud"], model["referencia"]["exactitud_balanceada"],
            model["referencia"]["precision_clase_1"], model["referencia"]["sensibilidad_clase_1"],
            model["referencia"]["f1_clase_1"], model["referencia"]["auc_roc"],
        ),
        (
            "Regresión logística",
            model["regresion_logistica"]["exactitud"],
            model["regresion_logistica"]["exactitud_balanceada"],
            model["regresion_logistica"]["precision_clase_1"],
            model["regresion_logistica"]["sensibilidad_clase_1"],
            model["regresion_logistica"]["f1_clase_1"],
            model["regresion_logistica"]["auc_roc"],
        ),
    ], columns=["Método", "Exactitud", "Exactitud balanceada", "Precisión devolución",
                "Sensibilidad devolución", "F1 devolución", "AUC ROC"])
    model_cols = list(model_metrics)
    model_table = table(model_metrics, model_cols, model_cols,
                        {name: 3 for name in model_cols[1:]})
    fin = figure

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe técnico — Ventas comerciales</title>
<style>
  :root {{ --ink: #202b25; --green: #1d5d3c; --soft: #eff6f0; --line: #cfddd2;
           --muted: #52655a; --warning: #fff8e8; }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{ margin: 0; color: var(--ink); background: #f5f7f5;
          font: 16px/1.58 "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
  main {{ max-width: 1060px; margin: 30px auto; background: white;
          padding: 52px 68px 75px; box-shadow: 0 8px 34px #183c2814; }}
  .masthead {{ border-top: 7px solid var(--green); border-bottom: 1px solid var(--line);
               padding: 26px 0 30px; }}
  .masthead p {{ margin: 0 0 10px; color: var(--muted); font-size: .9rem; }}
  h1 {{ font: 600 2.65rem/1.13 Georgia, "Times New Roman", serif;
        color: var(--green); margin: 0 0 12px; max-width: 680px; }}
  .subtitle {{ font-size: 1.06rem; max-width: 710px; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 8px 16px; padding: 20px 0;
         border-bottom: 1px solid var(--line); font-size: .87rem; }}
  nav a {{ color: var(--green); text-decoration: none; }}
  nav a:hover, nav a:focus-visible {{ text-decoration: underline; }}
  section {{ margin-top: 43px; }}
  h2 {{ color: var(--green); font: 600 1.48rem/1.25 Georgia, serif;
        border-left: 4px solid var(--green); padding-left: 13px; margin: 0 0 16px; }}
  h3 {{ font-size: 1.08rem; color: #254f37; margin: 25px 0 8px; }}
  p, li {{ max-width: 80ch; }}
  p {{ margin: 8px 0 15px; }}
  ul {{ padding-left: 23px; }}
  .summary {{ background: var(--soft); border-left: 4px solid var(--green);
              padding: 17px 20px; margin: 25px 0; }}
  .caution {{ background: var(--warning); border-left: 4px solid #b38630;
              padding: 15px 18px; margin: 19px 0; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 25px 0; }}
  .metric {{ padding: 16px 14px; background: var(--soft); border-top: 2px solid var(--green); }}
  .metric span {{ display: block; font-size: .82rem; color: var(--muted); }}
  .metric strong {{ display: block; font-size: 1.22rem; color: var(--green); margin-top: 5px; }}
  .table-wrap {{ overflow-x: auto; margin: 15px 0 24px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line);
            vertical-align: top; text-align: left; }}
  th {{ background: var(--soft); color: var(--green); }}
  tr:nth-child(even) td {{ background: #fafcfb; }}
  figure {{ margin: 22px 0 30px; text-align: center; break-inside: avoid; }}
  figure img {{ width: 100%; max-width: 830px; height: auto; border: 1px solid var(--line); }}
  figcaption {{ color: var(--muted); font-size: .84rem; margin: 5px auto 0; max-width: 75ch; }}
  code {{ overflow-wrap: anywhere; font-size: .85em; }}
  .small {{ color: var(--muted); font-size: .85rem; }}
  :focus-visible {{ outline: 2px solid var(--green); outline-offset: 3px; }}
  @media(max-width: 760px) {{
    main {{ margin: 0; padding: 26px 18px 50px; box-shadow: none; }}
    h1 {{ font-size: 2rem; }}
    .metrics {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  @media print {{
    @page {{ size: A4; margin: 16mm; }}
    body {{ background: white; font-size: 10pt; }}
    main {{ margin: 0; padding: 0; max-width: none; box-shadow: none; }}
    nav {{ display: none; }}
    section {{ margin-top: 22px; }}
    figure {{ margin: 12px 0; }}
    figure img {{ max-height: 140mm; width: auto; max-width: 100%; }}
    h2, h3, .table-wrap, tr {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<main>
<header class="masthead">
  <p>Proyecto integrador · Análisis de datos comerciales · Fuente: ventas_4.csv</p>
  <h1>Informe técnico de ventas comerciales</h1>
  <p><strong>Integrantes del grupo:</strong> Lucas Selva, Nazareno Mansilla y Roman Moreno.</p>
  <div class="subtitle">Calidad del dato, desempeño de ventas y evaluación de un modelo
  exploratorio de devoluciones. Período con fechas válidas:
  {kpi["inicio_periodo"]} a {kpi["fin_periodo"]}.</div>
</header>
<nav aria-label="Contenido">
  {''.join(f'<a href="#s{i}">{i}. {title}</a>' for i, title in enumerate([
    "Introducción", "Problema y objetivos", "Fuente y datos", "Metodología",
    "Limpieza", "Exploración", "Modelado", "Resultados", "Conclusiones", "Limitaciones"
  ], start=1))}
</nav>
<div class="summary"><strong>Resumen ejecutivo.</strong> Se analizaron
{number(kpi["operaciones_preparadas"], 0)} registros preparados de ventas tecnológicas.
El subtotal de ingresos <em>verificados</em> es {number(kpi["ingresos_verificados"])},
con cobertura de {number(kpi["cobertura_ingresos_pct"])} % de operaciones.
La tasa observada de devoluciones es {number(kpi["tasa_devolucion_pct"])} %.
El modelo exploratorio de devoluciones no muestra una mejora consistente frente
a una referencia simple en el período futuro;
no se recomienda usarlo para decidir sobre clientes.</div>
<div class="metrics">{cards}</div>

<section id="s1"><h2>1. Introducción y contexto</h2>
<p>La empresa comercializa productos tecnológicos y necesita conocer qué productos,
categorías, regiones y canales aportan a las ventas, así como explorar satisfacción,
entregas y devoluciones. El trabajo convierte los registros disponibles en indicadores
descriptivos auditables; no pretende demostrar causas ni extrapolar a operaciones no registradas.</p>
</section>

<section id="s2"><h2>2. Problema y objetivos</h2>
<p><strong>Problema:</strong> existen dudas sobre el rendimiento relativo de productos,
regiones y canales, pero el archivo contiene errores de calidad que impiden sumar
indiscriminadamente todos los importes.</p>
<p><strong>Objetivo:</strong> preparar los datos y responder, con denominadores explícitos:
cuánto ingreso puede verificarse, cómo cambia por mes, qué grupos destacan,
qué proporción se devuelve y qué asociaciones aparecen con edad, descuento,
entrega y satisfacción. Una pregunta predictiva adicional es si las variables
disponibles al vender anticipan una devolución mejor que una referencia trivial.</p>
</section>

<section id="s3"><h2>3. Fuente y descripción de los datos</h2>
<p>Archivo original: <code>{html.escape(source.name)}</code> (CSV, SHA-256
<code>{checksum}</code>). Se recibieron {number(audit["filas_originales"], 0)}
filas y 15 columnas; una fila parece representar una operación de venta,
aunque no se dispone de documentación externa sobre cómo se recolectaron.
El productor, la moneda y el alcance geográfico exactos no constan en la fuente;
por ello los importes se expresan en <strong>unidades monetarias del archivo</strong>,
sin asignar una divisa.</p>
<div class="table-wrap"><table><thead><tr><th>Campos</th><th>Función</th></tr></thead><tbody>
<tr><td>id_registro, fecha_venta</td><td>Identificación de operación y tiempo.</td></tr>
<tr><td>producto, categoria, region, canal_venta, medio_pago</td><td>Segmentación comercial.</td></tr>
<tr><td>cantidad, precio_unitario, descuento_pct, total_venta</td><td>Unidades, precio, proporción de descuento e importe.</td></tr>
<tr><td>edad_cliente, devolucion, dias_entrega, satisfaccion</td><td>Cliente, retorno y calidad del servicio.</td></tr>
</tbody></table></div>
<p>Las fechas válidas van del {kpi["inicio_periodo"]} al {kpi["fin_periodo"]}.
La edad es información potencialmente sensible: el informe presenta agregados,
no filas individuales ni identificadores personales.</p>
</section>

<section id="s4"><h2>4. Metodología</h2>
<p>Se conserva intacto el CSV. Python con pandas y NumPy transforma y audita una copia;
Matplotlib y Seaborn producen gráficos; scikit-learn evalúa un clasificador.
Primero se perfilan tipos, ausencias y duplicados; luego se normalizan categorías,
convierten fechas/números, verifican rangos y contrastan importes con su fórmula.
Los resultados descriptivos se calculan sobre los registros que poseen las variables
necesarias; se indica su cobertura. Se comparan tablas por producto, región,
categoría, canal, medio de pago y mes, sin afirmar causalidad.</p>
<p>Para la pregunta predictiva se ordenan los registros fechados, se entrena con
ventas anteriores al {model["corte_test_desde"]} y se evalúa sobre ventas posteriores.
Los IDs conflictivos no entran al modelo. Imputación, escalado y codificación se ajustan
dentro del <em>pipeline</em> únicamente con entrenamiento. Se compara regresión logística
con un clasificador que siempre predice la clase mayoritaria.</p>
</section>

<section id="s5"><h2>5. Limpieza y preparación</h2>
<p>Se quitaron {audit["duplicados_exactos_eliminados"]} duplicados completos, no todos
los IDs repetidos: {audit["filas_con_id_repetido_conservadas"]} filas con ID compartido
siguen visibles y requieren investigación. Quedaron
{number(audit["filas_preparadas"], 0)} operaciones. Las regiones se estandarizaron
recortando espacios y mayúsculas; las ausentes se etiquetaron «Sin dato».
Las fechas o cifras no interpretables pasan a valores faltantes, sin reemplazo
automático por promedios.</p>
<div class="table-wrap">{table(quality, list(quality), list(quality), {"Casos": 0})}</div>
<p>Reglas de plausibilidad: cantidad y precio mayores que cero; descuento entre
0 y 1; edad entre 18 y 100; días de entrega no negativos; satisfacción de 1 a 5.
Un ingreso se acepta si cantidad × precio × (1 − descuento) difiere a lo sumo
0,05 del importe declarado. Los montos no verificables o contradictorios no se
reconstruyen. Los valores extremos que cumplen estas reglas <em>no</em> se eliminan
automáticamente.</p>
<div class="caution"><strong>Alcance:</strong> {number(fraud, 0)} operaciones no aportan
ingresos verificados ({audit["importes_no_verificables"]} sin información para comprobar
el cálculo y {audit["importes_inconsistentes"]} con cálculo contradictorio).
El subtotal nunca debe etiquetarse «ingresos totales» de la empresa.</div>
</section>

<section id="s6"><h2>6. Análisis exploratorio</h2>
<h3>Ventas y períodos</h3>
<p>El mes con mayor subtotal comprobado es {best_month["mes"]}
({number(best_month["ingresos_verificados"])}; {number(best_month["operaciones"], 0)}
operaciones con fecha e importe verificados). El menor es {worst_month["mes"]}
({number(worst_month["ingresos_verificados"])}; {number(worst_month["operaciones"], 0)}
operaciones). El período de 2025 llega solo hasta septiembre y no es comparable
como año completo con 2023 o 2024. Años con doce meses representados:
{years_text}; incluso allí, observar los doce meses no acredita cobertura exhaustiva.</p>
{fin(folder, "01_evolucion_mensual.png", "Figura 1. Subtotal mensual verificable; solo operaciones con fecha válida. La serie no demuestra estacionalidad por sí misma.")}
<h3>Productos, categorías y regiones</h3>
<div class="table-wrap">{table(product, ["producto","operaciones","operaciones_importe_verificado","ingresos_verificados","unidades_validas","tasa_devolucion_pct"],
["Producto","Operaciones","Importes verificados","Ingresos verificados","Unidades válidas","Devoluciones (%)"],
{"operaciones":0,"operaciones_importe_verificado":0,"ingresos_verificados":2,"unidades_validas":0,"tasa_devolucion_pct":1})}</div>
<p>{best_product["producto"]} registra el mayor ingreso verificado
({number(best_product["ingresos_verificados"])}); {best_units["producto"]} reúne más
unidades válidas ({number(best_units["unidades_validas"],0)}). Liderar en importe
no equivale necesariamente a vender más unidades.</p>
{fin(folder, "04_ingresos_producto.png", "Figura 2. Ingresos verificados por producto; las coberturas de cálculo figuran en la tabla anterior.")}
{fin(folder, "06_unidades_producto.png", "Figura 3. Unidades por producto, únicamente registros con cantidad válida; no implica ingreso verificado.")}
<div class="table-wrap">{table(category, ["categoria","operaciones","operaciones_importe_verificado","ingresos_verificados"],
["Categoría","Operaciones","Importes verificados","Ingresos verificados"],
{"operaciones":0,"operaciones_importe_verificado":0,"ingresos_verificados":2})}</div>
<p>La categoría con mayor subtotal es {top_category["categoria"]}
({number(top_category["ingresos_verificados"])}). La región con mayor subtotal
es {best_region["region"]} ({number(best_region["ingresos_verificados"])}).
La comparación requiere considerar operaciones y cobertura, no solo suma.</p>
{fin(folder, "02_ingresos_categoria.png", "Figura 4. Subtotal verificado por categoría, en unidades monetarias no especificadas por la fuente.")}
<div class="table-wrap">{table(region, ["region","operaciones","operaciones_importe_verificado","ingresos_verificados","tasa_devolucion_pct"],
["Región","Operaciones","Importes verificados","Ingresos verificados","Devoluciones (%)"],
{"operaciones":0,"operaciones_importe_verificado":0,"ingresos_verificados":2,"tasa_devolucion_pct":1})}</div>
{fin(folder, "03_ingresos_region.png", "Figura 5. Ingresos verificados por región, incluida la región «Sin dato» para no esconder faltantes.")}
<h3>Canales, pagos y servicio</h3>
<div class="table-wrap">{table(channel, ["canal_venta","operaciones","operaciones_importe_verificado","ingresos_verificados","tasa_devolucion_pct"],
["Canal","Operaciones","Importes verificados","Ingresos verificados","Devoluciones (%)"],
{"operaciones":0,"operaciones_importe_verificado":0,"ingresos_verificados":2,"tasa_devolucion_pct":1})}</div>
<div class="table-wrap">{table(payment, ["medio_pago","operaciones","operaciones_importe_verificado","ingresos_verificados"],
["Medio de pago","Operaciones","Importes verificados","Ingresos verificados"],
{"operaciones":0,"operaciones_importe_verificado":0,"ingresos_verificados":2})}</div>
{fin(folder, "05_ingresos_canal.png", "Figura 6. Subtotal verificado por canal de venta; no controla diferencias en cantidad de operaciones.")}
<p>La tasa global de devolución es {number(kpi["tasa_devolucion_pct"])} % sobre
{number(kpi["devoluciones_observadas"],0)} operaciones con etiqueta.
La mayor tasa por producto corresponde a {return_by_product["producto"]}
({number(return_by_product["tasa_devolucion_pct"])} % sobre
{number(return_by_product["devoluciones_observadas"],0)} registros etiquetados).
La satisfacción media es {number(kpi["satisfaccion_media"])} / 5
(n={number(kpi["satisfaccion_observada"],0)}) y la entrega media
{number(kpi["dias_entrega_medios"])} días
(n={number(kpi["dias_entrega_observados"],0)}).</p>
{fin(folder, "07b_devoluciones_producto.png", "Figura 7. Devoluciones como porcentaje de operaciones etiquetadas, por producto.")}
{fin(folder, "07_devoluciones_region.png", "Figura 8. Tasa de devolución por región, incluida la categoría sin dato.")}
{fin(folder, "07c_devoluciones_canal.png", "Figura 9. Tasa de devolución por canal, con denominadores en por_canal.csv.")}
{fin(folder, "08_entrega_satisfaccion.png", "Figura 10. Días de entrega según satisfacción; solo pares observados, sin inferencia causal.")}
<h3>Relaciones entre variables</h3>
<p>Las correlaciones de Pearson por pares disponibles son: entrega–satisfacción
{number(corr_delivery,3)}, edad–importe verificado {number(corr_age,3)}
y descuento–importe verificado {number(corr_discount,3)}. Estas asociaciones
descriptivas no aíslan el efecto de producto, precio ni otros factores.
Un descuento afecta mecánicamente la fórmula del importe, por lo que esa
correlación no mide un impacto causal de la política comercial.</p>
{fin(folder, "09_edad_importe.png", "Figura 11. Edad del cliente e importe verificado; registros con ambas variables válidas.")}
{fin(folder, "10_descuento_importe.png", "Figura 12. Descuento e importe verificado; parte de la relación es aritmética por definición.")}
{fin(folder, "11_correlaciones.png", "Figura 13. Matriz Pearson con pares disponibles; valores faltantes pueden cambiar el n de cada celda.")}
</section>

<section id="s7"><h2>7. Modelado, si corresponde</h2>
<p><strong>Pregunta predictiva:</strong> ¿se puede anticipar una devolución con datos
disponibles durante la venta? Se ensayó un clasificador logístico interpretable,
no una regresión del importe: predecir importe desde cantidad, precio y descuento
repetiría su definición. Predictores: producto, categoría, región, canal, medio
de pago, cantidad, precio, descuento y edad. Se excluyeron satisfacción y entrega,
que pueden conocerse después del evento; tampoco se usó el importe o el ID.</p>
<p>Se evaluaron {number(model["filas_elegibles"],0)} operaciones elegibles:
{number(model["entrenamiento"],0)} anteriores al {model["corte_test_desde"]} para
entrenar y {number(model["prueba"],0)} desde esa fecha para prueba, sin mezclar
orden temporal ni IDs repetidos. Prevalencia de devolución:
{number(model["prevalencia_entrenamiento"]*100,2)} % en entrenamiento frente a
{number(model["prevalencia_prueba"]*100,2)} % en prueba. Umbral de clase 1: 0,5.
Una referencia predice siempre la clase mayoritaria de entrenamiento.</p>
<div class="table-wrap">{model_table}</div>
{fin(folder, "12_modelo_confusion_comparacion.png", "Figura 14. Matriz de confusión del modelo (filas observadas, columnas predichas) y comparación con referencia.")}
{fin(folder, "13_modelo_curvas.png", "Figura 15. ROC y precisión–sensibilidad del clasificador; las líneas punteadas representan azar y prevalencia en prueba.")}
<p>La exactitud balanceada del modelo es
{number(model["regresion_logistica"]["exactitud_balanceada"],3)} frente a
{number(model["referencia"]["exactitud_balanceada"],3)} de la referencia;
el AUC ROC es {number(model["regresion_logistica"]["auc_roc"],3)} frente a 0,500.
En prueba hubo {model["regresion_logistica"]["matriz_confusion"][1][0]} devoluciones
no anticipadas y {model["regresion_logistica"]["matriz_confusion"][0][1]}
falsas alarmas. El F1 de la clase positiva sí supera al de la referencia
porque esta última nunca predice devoluciones, pero la exactitud balanceada
y el AUC ROC empeoran. <strong>No hay evidencia de valor predictivo útil</strong> en esta
partición; el modelo se muestra como evaluación negativa, no como herramienta operativa.</p>
</section>

<section id="s8"><h2>8. Resultados</h2>
<div class="table-wrap"><table><thead><tr><th>Pregunta</th><th>Hallazgo e interpretación</th><th>Límite</th></tr></thead><tbody>
<tr><td>¿Cuánto se vendió?</td><td>{number(kpi["ingresos_verificados"])} de ingreso verificable;
ticket medio verificable {number(kpi["ticket_medio_verificado"])}.</td>
<td>Solo {number(kpi["operaciones_con_importe_verificado"],0)} de
{number(kpi["operaciones_preparadas"],0)} operaciones; no representa el total empresarial.</td></tr>
<tr><td>¿Qué productos y regiones destacan?</td><td>{html.escape(str(best_product["producto"]))}
lidera en ingreso comprobado y {html.escape(str(best_region["region"]))} en
subtotal regional.</td><td>Las diferencias reflejan esta muestra y distintas coberturas.</td></tr>
<tr><td>¿Cómo evolucionaron las ventas?</td><td>La serie presenta variación mensual;
{best_month["mes"]} es el máximo observado.</td><td>2025 es parcial; sin cobertura
comprobada no cabe afirmar una estacionalidad general.</td></tr>
<tr><td>¿Qué ocurre con devoluciones y satisfacción?</td>
<td>Devolución observada {number(kpi["tasa_devolucion_pct"])} %;
satisfacción media {number(kpi["satisfaccion_media"])} / 5.</td>
<td>No se conoce el proceso de generación ni el motivo de cada devolución.</td></tr>
<tr><td>¿Podemos anticipar devoluciones?</td>
<td>La regresión logística no mejora de forma consistente a la referencia
en test temporal.</td>
<td>No usar el modelo para decisiones individuales.</td></tr>
</tbody></table></div>
</section>

<section id="s9"><h2>9. Conclusiones y recomendaciones</h2>
<ol>
<li>Usar el subtotal <strong>verificado</strong> y su cobertura cuando se comuniquen
ingresos; no presentar el importe bruto sin auditar como hecho establecido.</li>
<li>Investigar primero los {audit["importes_inconsistentes"]} importes discordantes,
los {audit["importes_no_verificables"]} no comprobables y los IDs repetidos;
corregirlos desde el sistema de origen, sin atribuir automáticamente error a
un campo particular.</li>
<li>Contrastar productos, regiones y canales con sus denominadores antes de
priorizar acciones. Las tasas de devolución elevadas merecen revisión de causas
con datos de motivo de devolución, no atribución causal por categoría.</li>
<li>Registrar períodos completos y definiciones de moneda, captación y descuentos;
sin ello no puede evaluarse con rigor una estacionalidad ni comparar desempeño externo.</li>
<li>No desplegar el clasificador de devoluciones: en la prueba futura la exactitud
balanceada y el AUC ROC no mejoran la referencia elemental.</li>
</ol>
</section>

<section id="s10"><h2>10. Limitaciones y trabajo futuro</h2>
<p>Procedencia, moneda, cobertura poblacional y forma de muestreo no documentadas;
{audit["fechas_invalidas"]} fechas inválidas; 2025 parcial; ID repetido no resuelto;
ausencias y rangos anómalos; y solo {number(kpi["cobertura_ingresos_pct"])} % de
operaciones con importe comprobable. El subtotal y las comparaciones pueden estar
sesgados si los errores no se distribuyen por igual entre grupos. No se conocen
costos, margen, motivo de devolución, promociones ni exposición de clientes;
las relaciones no prueban causalidad y las diferencias no son necesariamente
generalizables.</p>
<p>Trabajo futuro: obtener diccionario y trazabilidad desde el sistema de ventas,
reconciliar operaciones defectuosas, validar el importe contra comprobantes,
confirmar períodos cerrados, estudiar motivos de devolución, y solo entonces
replantear un objetivo predictivo con datos disponibles antes de decidir.</p>
</section>
</main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "ventas_4.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "resultados_ventas")
    args = parser.parse_args()
    raw = pd.read_csv(args.input, dtype={"id_registro": "string"}, low_memory=False)
    data, audit = clean_sales(raw)
    kpi = export_analysis(data, audit, args.output)
    model = evaluate_returns(data, args.output)
    document = compose(args.input, args.output, audit, kpi, model)
    path = args.output / "informe_ventas.html"
    path.write_text(document, encoding="utf-8")
    print(json.dumps({"informe": str(path.resolve()), "secciones": 10,
                      "cobertura_ingresos_pct": kpi["cobertura_ingresos_pct"],
                      "modelo_auc_roc": model["regresion_logistica"]["auc_roc"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
