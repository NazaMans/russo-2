"""Limpieza auditable y análisis descriptivo de ventas_4.csv.

No genera un informe ni entrena un modelo: la prioridad es obtener indicadores
defendibles, sin presentar como ingresos confiables importes contradictorios.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
EXPECTED = [
    "id_registro", "fecha_venta", "producto", "categoria", "region",
    "cantidad", "precio_unitario", "descuento_pct", "total_venta",
    "edad_cliente", "canal_venta", "medio_pago", "devolucion",
    "dias_entrega", "satisfaccion",
]
NUMERIC = [
    "cantidad", "precio_unitario", "descuento_pct", "total_venta",
    "edad_cliente", "dias_entrega", "satisfaccion",
]


def clean_sales(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Conserva registros parciales y marca explícitamente toda exclusión de KPI."""
    missing = set(EXPECTED) - set(raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(sorted(missing))}")
    if raw.empty:
        raise ValueError("El archivo CSV está vacío.")

    data = raw[EXPECTED].copy()
    initial_nulls = data.isna().sum().astype(int).to_dict()
    exact_duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    duplicate_ids = int(data["id_registro"].duplicated(keep=False).sum())

    for column in ["producto", "categoria", "region", "canal_venta", "medio_pago"]:
        data[column] = data[column].astype("string").str.strip()
        data[column] = data[column].replace("", pd.NA)
    # Unificar diferencias de mayúsculas/espacios sin inventar regiones faltantes.
    data["region"] = data["region"].str.title()
    invalid_dates = data["fecha_venta"].notna() & pd.to_datetime(
        data["fecha_venta"], errors="coerce"
    ).isna()
    data["fecha_venta"] = pd.to_datetime(data["fecha_venta"], errors="coerce")
    data["region"] = data["region"].fillna("Sin dato")

    invalid_numbers = {}
    for column in NUMERIC:
        before = data[column].notna()
        data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid_numbers[column] = int((before & data[column].isna()).sum())

    invalid_ranges = {}
    rules = {
        "cantidad": lambda s: s <= 0,
        "precio_unitario": lambda s: s <= 0,
        "descuento_pct": lambda s: (s < 0) | (s > 1),
        "total_venta": lambda s: s < 0,
        "edad_cliente": lambda s: (s < 18) | (s > 100),
        "dias_entrega": lambda s: s < 0,
        "satisfaccion": lambda s: ~s.isin([1, 2, 3, 4, 5]),
    }
    for column, invalid in rules.items():
        mask = data[column].notna() & invalid(data[column])
        invalid_ranges[column] = int(mask.sum())
        data.loc[mask, column] = np.nan

    returns = data["devolucion"].astype("string").str.strip().str.lower()
    bad_returns = ~returns.isin(["true", "false"]) & returns.notna()
    data["devolucion"] = returns.map({"true": True, "false": False}).astype("boolean")

    # No sustituimos total_venta por una fórmula: puede haber errores en cualquiera
    # de los cuatro campos. Una diferencia > 5 centavos supera el redondeo a 2 decimales.
    comparable = data[["cantidad", "precio_unitario", "descuento_pct", "total_venta"]].notna().all(axis=1)
    expected_total = data["cantidad"] * data["precio_unitario"] * (1 - data["descuento_pct"])
    inconsistent = comparable & ((data["total_venta"] - expected_total).abs() > 0.05)
    unverifiable = ~comparable
    data["importe_verificado"] = comparable & ~inconsistent
    data["total_venta_validado"] = data["total_venta"].where(data["importe_verificado"])
    data["fecha_valida"] = data["fecha_venta"].notna()
    data["id_duplicado"] = data["id_registro"].duplicated(keep=False)
    data["mes"] = data["fecha_venta"].dt.to_period("M").astype("string")
    data["mes_calendario"] = data["fecha_venta"].dt.month

    audit = {
        "filas_originales": int(len(raw)),
        "duplicados_exactos_eliminados": exact_duplicates,
        "filas_con_id_repetido_conservadas": duplicate_ids,
        "filas_preparadas": int(len(data)),
        "faltantes_originales": initial_nulls,
        "fechas_invalidas": int(invalid_dates.sum()),
        "conversiones_numericas_invalidas": invalid_numbers,
        "valores_fuera_de_rango": invalid_ranges,
        "devoluciones_invalidas": int(bad_returns.sum()),
        "importes_inconsistentes": int(inconsistent.sum()),
        "importes_no_verificables": int(unverifiable.sum()),
        "importes_verificados": int(data["importe_verificado"].sum()),
        "faltantes_finales": data[EXPECTED].isna().sum().astype(int).to_dict(),
        "criterio_ingresos": (
            "Solo operaciones con cantidad > 0, precio > 0, descuento entre 0 y 1 "
            "e importe consistente con cantidad * precio * (1 - descuento), "
            "tolerancia absoluta 0.05. No se imputan valores monetarios."
        ),
    }
    return data.reset_index(drop=True), audit


def group_sales(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Los denominadores se muestran para no ocultar el alcance de cada indicador."""
    result = (
        data.groupby(column, dropna=False, observed=True)
        .agg(
            operaciones=("id_registro", "size"),
            operaciones_importe_verificado=("importe_verificado", "sum"),
            ingresos_verificados=("total_venta_validado", "sum"),
            unidades_validas=("cantidad", "sum"),
            observaciones_cantidad=("cantidad", "count"),
            devoluciones=("devolucion", lambda x: int(x.sum())),
            devoluciones_observadas=("devolucion", "count"),
            satisfaccion_media=("satisfaccion", "mean"),
            observaciones_satisfaccion=("satisfaccion", "count"),
        )
        .reset_index()
    )
    result["ticket_medio_verificado"] = (
        result["ingresos_verificados"] /
        result["operaciones_importe_verificado"].replace(0, np.nan)
    )
    result["tasa_devolucion_pct"] = (
        100 * result["devoluciones"] /
        result["devoluciones_observadas"].replace(0, np.nan)
    )
    return result


def export_analysis(data: pd.DataFrame, audit: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    charts = output / "graficos"
    charts.mkdir(exist_ok=True)
    data.to_csv(output / "ventas_preparadas.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "columna": EXPECTED,
        "faltantes_originales": [audit["faltantes_originales"][k] for k in EXPECTED],
        "faltantes_preparados": [audit["faltantes_finales"][k] for k in EXPECTED],
    }).to_csv(output / "calidad_por_columna.csv", index=False, encoding="utf-8-sig")

    tables = {}
    for name, column in [
        ("por_producto", "producto"), ("por_categoria", "categoria"),
        ("por_region", "region"), ("por_canal", "canal_venta"),
        ("por_medio_pago", "medio_pago"), ("por_devolucion", "devolucion"),
    ]:
        tables[name] = group_sales(data, column)
        tables[name].to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")

    # La serie mensual usa únicamente operaciones verificadas con fecha válida.
    dated = data.loc[data["fecha_valida"] & data["importe_verificado"]].copy()
    monthly = (
        dated.groupby("mes", observed=True)
        .agg(ingresos_verificados=("total_venta_validado", "sum"),
             operaciones=("id_registro", "size"))
        .reset_index().sort_values("mes")
    )
    monthly.to_csv(output / "por_mes.csv", index=False, encoding="utf-8-sig")
    tables["por_mes"] = monthly

    # La estacionalidad es solo descriptiva: comparar meses de años completos
    # antes de afirmar un patrón, y nunca confundirlo con una causa.
    by_year_month = dated.groupby(
        [dated["fecha_venta"].dt.year.rename("anio"),
         dated["fecha_venta"].dt.month.rename("mes_calendario")]
    )["total_venta_validado"].sum().reset_index(name="ingresos")
    by_year_month.to_csv(output / "por_anio_mes.csv", index=False, encoding="utf-8-sig")

    numeric = [
        "cantidad", "precio_unitario", "descuento_pct", "total_venta_validado",
        "edad_cliente", "dias_entrega", "satisfaccion",
    ]
    data[numeric].describe().T.to_csv(output / "descriptivos_numericos.csv", encoding="utf-8-sig")
    corr = data[numeric].corr()
    corr.to_csv(output / "correlaciones_pearson.csv", encoding="utf-8-sig")

    kpis = {
        "operaciones_preparadas": int(len(data)),
        "operaciones_con_importe_verificado": int(data["importe_verificado"].sum()),
        "cobertura_ingresos_pct": round(100 * data["importe_verificado"].mean(), 2),
        "ingresos_verificados": round(float(data["total_venta_validado"].sum()), 2),
        "unidades_en_registros_con_cantidad_valida": round(float(data["cantidad"].sum()), 2),
        "registros_con_cantidad_valida": int(data["cantidad"].count()),
        "precio_unitario_medio": round(float(data["precio_unitario"].mean()), 2),
        "ticket_medio_verificado": round(float(data["total_venta_validado"].mean()), 2),
        "tasa_devolucion_pct": round(float(data["devolucion"].mean()) * 100, 2),
        "devoluciones_observadas": int(data["devolucion"].count()),
        "descuento_medio_pct": round(float(data["descuento_pct"].mean()) * 100, 2),
        "satisfaccion_media": round(float(data["satisfaccion"].mean()), 2),
        "satisfaccion_observada": int(data["satisfaccion"].count()),
        "dias_entrega_medios": round(float(data["dias_entrega"].mean()), 2),
        "dias_entrega_observados": int(data["dias_entrega"].count()),
        "fechas_validas": int(data["fecha_valida"].sum()),
        "inicio_periodo": str(data["fecha_venta"].min().date()),
        "fin_periodo": str(data["fecha_venta"].max().date()),
    }
    for filename, obj in [("auditoria_limpieza.json", audit), ("indicadores.json", kpis)]:
        (output / filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams["figure.figsize"] = (10, 5)

    def save(filename: str) -> None:
        plt.tight_layout()
        plt.savefig(charts / filename, dpi=170, bbox_inches="tight")
        plt.close()

    if not monthly.empty:
        ax = sns.lineplot(data=monthly, x="mes", y="ingresos_verificados", marker="o")
        ax.set(title="Ingresos mensuales verificados", xlabel="Mes",
               ylabel="Importe (u. monetarias no especificadas)")
        ax.tick_params(axis="x", rotation=75)
        save("01_evolucion_mensual.png")

    for name, label, filename in [
        ("por_categoria", "Categoría", "02_ingresos_categoria.png"),
        ("por_region", "Región", "03_ingresos_region.png"),
        ("por_producto", "Producto", "04_ingresos_producto.png"),
        ("por_canal", "Canal", "05_ingresos_canal.png"),
    ]:
        table = tables[name].sort_values("ingresos_verificados", ascending=False)
        field = {"por_categoria": "categoria", "por_region": "region",
                 "por_producto": "producto", "por_canal": "canal_venta"}[name]
        ax = sns.barplot(data=table, x=field, y="ingresos_verificados")
        ax.set(title=f"Ingresos verificados por {label.lower()}", xlabel=label,
               ylabel="Importe (u. monetarias no especificadas)")
        ax.tick_params(axis="x", rotation=30)
        save(filename)

    table = tables["por_producto"].sort_values("unidades_validas", ascending=False)
    ax = sns.barplot(data=table, x="producto", y="unidades_validas")
    ax.set(title="Unidades registradas por producto (cantidad válida)", xlabel="Producto", ylabel="Unidades")
    save("06_unidades_producto.png")

    ax = sns.barplot(data=tables["por_region"], x="region", y="tasa_devolucion_pct")
    ax.set(title="Tasa de devolución por región", xlabel="Región", ylabel="Porcentaje")
    save("07_devoluciones_region.png")

    ax = sns.barplot(data=tables["por_producto"], x="producto", y="tasa_devolucion_pct")
    ax.set(title="Tasa de devolución por producto", xlabel="Producto", ylabel="Porcentaje")
    save("07b_devoluciones_producto.png")

    ax = sns.barplot(data=tables["por_canal"], x="canal_venta", y="tasa_devolucion_pct")
    ax.set(title="Tasa de devolución por canal", xlabel="Canal", ylabel="Porcentaje")
    save("07c_devoluciones_canal.png")

    subset = data.loc[data["dias_entrega"].notna() & data["satisfaccion"].notna()]
    if not subset.empty:
        ax = sns.boxplot(data=subset, x="satisfaccion", y="dias_entrega")
        ax.set(title="Entrega y satisfacción (registros completos)", xlabel="Satisfacción", ylabel="Días de entrega")
        save("08_entrega_satisfaccion.png")

    subset = data.loc[data["edad_cliente"].notna() & data["importe_verificado"]]
    if not subset.empty:
        ax = sns.scatterplot(data=subset, x="edad_cliente", y="total_venta_validado",
                             alpha=0.4, s=18)
        ax.set(title="Edad e importe verificado", xlabel="Edad (años)",
               ylabel="Importe (u. monetarias no especificadas)")
        save("09_edad_importe.png")

    subset = data.loc[data["descuento_pct"].notna() & data["importe_verificado"]]
    if not subset.empty:
        ax = sns.scatterplot(data=subset, x="descuento_pct", y="total_venta_validado",
                             alpha=0.4, s=18)
        ax.set(title="Descuento e importe verificado", xlabel="Descuento (proporción)",
               ylabel="Importe (u. monetarias no especificadas)")
        save("10_descuento_importe.png")

    ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="BrBG", center=0, vmin=-1, vmax=1)
    ax.set(title="Correlaciones de Pearson (pares disponibles)")
    save("11_correlaciones.png")
    return kpis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "ventas_4.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "resultados_ventas")
    args = parser.parse_args()
    raw = pd.read_csv(args.input, dtype={"id_registro": "string"}, low_memory=False)
    data, audit = clean_sales(raw)
    kpis = export_analysis(data, audit, args.output)
    print(json.dumps({"salida": str(args.output.resolve()), "auditoria": audit,
                      "indicadores": kpis}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
