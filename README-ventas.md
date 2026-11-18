# Análisis reproducible de `ventas_4.csv`

`analizar_ventas.py` procesa únicamente el CSV de ventas. No modifica el original
ni usa los datasets del repositorio de ejemplo. `generar_informe_ventas.py`
recalcula el análisis, evalúa un modelo exploratorio y genera el informe HTML
autocontenido de diez secciones.

## Ejecutar (PowerShell)

```powershell
cd C:\Users\nazar\Downloads\examen-russo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ventas.txt
.\.venv\Scripts\python.exe analizar_ventas.py
.\.venv\Scripts\python.exe generar_informe_ventas.py
.\.venv\Scripts\python.exe -m unittest -v test_analizar_ventas.py
```

Se puede elegir otro archivo o carpeta con `--input RUTA` y `--output RUTA`.
Por defecto produce `resultados_ventas/`: CSV preparado, auditoría de calidad
en JSON y CSV, indicadores en JSON, tablas por mes/año/producto/categoría/región/
canal/medio de pago/devolución, estadísticos descriptivos, correlaciones y gráficos.
El informe final queda en `resultados_ventas/informe_ventas.html`; integra sus
gráficos dentro del propio archivo y puede abrirse sin conexión.

## Decisiones de limpieza e interpretación

- Elimina solo duplicados **completos**. Conserva identificadores repetidos si sus
  registros difieren, y los marca para revisión.
- Normaliza espacios y mayúsculas de la región; la región ausente se etiqueta
  «Sin dato». Una fecha inválida se conserva como fecha faltante y no entra en
  el análisis temporal.
- Convierte errores de tipo a faltantes. Excluye cantidades y precios no positivos,
  descuentos fuera de `[0,1]`, edades fuera de `[18,100]`, días de entrega negativos
  y satisfacciones fuera de `1–5`. Los registros parciales se conservan.
- Un importe solo se contabiliza como ingreso si es verificable con cantidad,
  precio y descuento válidos, y la diferencia con su fórmula no supera 0,05.
  **Los ingresos son un subtotal verificado, no el total real de toda la empresa.**
  Se reporta su cobertura y no se imputan importes ni se inventan fechas.
- Cada métrica usa los registros con los datos requeridos; se exportan sus
  denominadores. Correlaciones observacionales no implican causalidad.
- El script descriptivo no entrena modelos. El generador del informe evalúa
  devoluciones con regresión logística frente a una referencia de clase
  mayoritaria. Usa un corte temporal (anteriores vs. posteriores), descarta
  IDs repetidos del modelo y aprende imputaciones/transformaciones solo
  del conjunto de entrenamiento. No usa `total_venta` como objetivo porque
  predecirlo desde sus componentes sería una tautología.
- El modelo tuvo peor exactitud balanceada y AUC ROC que la referencia
  en el período futuro, aunque sí detectó algunas devoluciones que la
  referencia nunca detectaba. **No se recomienda su uso operativo**.

El informe sigue las diez secciones de `estructura.md`; las decisiones de
interpretación y las limitaciones siguen `proceso.md` y `ventas.md`.
