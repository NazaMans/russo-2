"""Evaluación exploratoria de devoluciones: corte temporal y referencia simple."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL = ["producto", "categoria", "region", "canal_venta", "medio_pago"]
NUMERIC = ["cantidad", "precio_unitario", "descuento_pct", "edad_cliente"]
FEATURES = CATEGORICAL + NUMERIC


def evaluate_returns(data: pd.DataFrame, output: Path) -> dict:
    """Pregunta: ¿se anticipa una devolución con datos conocidos al vender?"""
    output.mkdir(parents=True, exist_ok=True)
    # Evita cruces entre registros relacionados y no inventa orden para fechas nulas.
    eligible = data.loc[
        data["fecha_venta"].notna() & data["devolucion"].notna()
        & ~data["id_duplicado"]
    ].sort_values("fecha_venta").copy()
    if len(eligible) < 100 or eligible["devolucion"].nunique() < 2:
        raise ValueError("No hay suficientes fechas/etiquetas válidas para evaluar el modelo.")
    cutoff = eligible.iloc[int(len(eligible) * 0.75)]["fecha_venta"]
    train = eligible.loc[eligible["fecha_venta"] < cutoff]
    test = eligible.loc[eligible["fecha_venta"] >= cutoff]
    if train["devolucion"].nunique() < 2 or test["devolucion"].nunique() < 2:
        raise ValueError("Las dos clases deben existir en entrenamiento y prueba.")

    # astype(object) convierte pd.NA a None para el imputador de sklearn.
    x_train = train[FEATURES].copy()
    x_test = test[FEATURES].copy()
    for field in CATEGORICAL:
        x_train[field] = x_train[field].astype(object).where(x_train[field].notna(), None)
        x_test[field] = x_test[field].astype(object).where(x_test[field].notna(), None)
    y_train = train["devolucion"].astype(int)
    y_test = test["devolucion"].astype(int)
    preprocessor = ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERIC),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    model = Pipeline([
        ("preparar", preprocessor),
        ("clasificar", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    model.fit(x_train, y_train)

    def scores(estimator, x: pd.DataFrame) -> dict:
        pred = estimator.predict(x)
        prob = estimator.predict_proba(x)[:, list(estimator.classes_).index(1)]
        return {
            "exactitud": round(float(accuracy_score(y_test, pred)), 4),
            "exactitud_balanceada": round(float(balanced_accuracy_score(y_test, pred)), 4),
            "precision_clase_1": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "sensibilidad_clase_1": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1_clase_1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "auc_roc": round(float(roc_auc_score(y_test, prob)), 4),
            "average_precision": round(float(average_precision_score(y_test, prob)), 4),
            "matriz_confusion": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        }

    base = scores(baseline, x_test)
    fitted = scores(model, x_test)
    prob = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    chart_dir = output / "graficos"
    chart_dir.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, model.predict(x_test), labels=[0, 1],
        display_labels=["No", "Sí"], cmap="Greens", colorbar=False, ax=axes[0]
    )
    axes[0].set_title("Devoluciones: errores en período futuro")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Observado")
    labels = ["Exactitud bal.", "F1 devolución", "AUC ROC"]
    base_values = [base["exactitud_balanceada"], base["f1_clase_1"], base["auc_roc"]]
    model_values = [fitted["exactitud_balanceada"], fitted["f1_clase_1"], fitted["auc_roc"]]
    indices = range(3)
    axes[1].barh([i - 0.18 for i in indices], base_values, height=0.35,
                 label="Referencia: clase mayoritaria", color="#9e9e9e")
    axes[1].barh([i + 0.18 for i in indices], model_values, height=0.35,
                 label="Regresión logística", color="#2e7d32")
    axes[1].set_yticks(list(indices), labels)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Puntuación (0–1)")
    axes[1].set_title("Comparación con referencia")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_dir / "12_modelo_confusion_comparacion.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[0].plot(fpr, tpr, color="#2e7d32", label=f"Modelo (AUC {fitted['auc_roc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Azar (AUC 0,5)")
    axes[0].set(xlabel="Tasa falsos positivos", ylabel="Sensibilidad",
                title="Curva ROC — devoluciones")
    axes[0].legend()
    precision, recall, _ = precision_recall_curve(y_test, prob)
    axes[1].plot(recall, precision, color="#2e7d32",
                 label=f"Modelo (AP {fitted['average_precision']:.3f})")
    prevalence = float(y_test.mean())
    axes[1].axhline(prevalence, linestyle="--", color="gray",
                    label=f"Prevalencia ({prevalence:.3f})")
    axes[1].set(xlabel="Sensibilidad", ylabel="Precisión",
                title="Precisión–sensibilidad — devoluciones", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "13_modelo_curvas.png", dpi=170)
    plt.close(fig)

    details = {
        "pregunta": "¿Se pueden anticipar devoluciones con variables disponibles al vender?",
        "objetivo": "devolucion",
        "predictores": FEATURES,
        "excluidos": "id, fecha como predictor, total_venta, satisfaccion y dias_entrega",
        "motivo_exclusion": (
            "id no generaliza; importe deriva de otras variables; satisfacción y entrega "
            "pueden conocerse solo después de la venta. Fecha se usa solo para partición."
        ),
        "filas_elegibles": int(len(eligible)),
        "filas_excluidas": int(len(data) - len(eligible)),
        "corte_test_desde": str(cutoff.date()),
        "entrenamiento": int(len(train)),
        "prueba": int(len(test)),
        "prevalencia_entrenamiento": round(float(y_train.mean()), 4),
        "prevalencia_prueba": round(prevalence, 4),
        "referencia": base,
        "regresion_logistica": fitted,
        "semilla_modelo": 42,
        "umbral_clasificacion": 0.5,
        "advertencia": "Ejercicio exploratorio, no evidencia de causalidad ni autorización para despliegue.",
    }
    (output / "evaluacion_modelo.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return details
