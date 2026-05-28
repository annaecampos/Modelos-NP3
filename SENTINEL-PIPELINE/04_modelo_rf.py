"""
Modelo baseline: Random Forest Regressor para estimativa de forragem.

Por que Random Forest como primeiro modelo?
  - Lida bem com poucos dados (experimental farm) e sem normalização
  - Robusto a outliers e variáveis correlacionadas
  - Feature importance nativa para análise exploratória
  - Resultados interpretáveis para apresentar ao orientador
  - Base de comparação para LSTM/GRU da fase seguinte

Features: bandas S2 brutas + índices (NDVI, EVI, NDRE, SAVI) + clima + temporais
Alvo    : definido em config.py (padrão = mstotal)

Saída em resultados_rf/:
  - metricas.txt          → RMSE, MAE, R² por piquete e geral
  - importancia_features.png → top-20 variáveis mais importantes
  - real_vs_predito.png   → scatter plot
  - residuos.png          → distribuição dos erros
  - modelo_rf.joblib      → modelo treinado (para inferência futura)

Uso:
    python 04_modelo_rf.py
"""

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from config import ALVO, DATA_DIR, BASE_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Colunas que nunca entram como feature
NAO_FEATURES = {
    "data", "idpotreiro", "idsubarea", "data_imagem",
    "mstotal", "msanoni", "msoutras", "media",
    "geometria",
}


def carregar_dataset() -> pd.DataFrame:
    f = DATA_DIR / "dataset_final.csv"
    if not f.exists():
        log.error("Arquivo não encontrado: %s\nExecute primeiro: python 03_monta_dataset.py", f)
        raise SystemExit(1)
    return pd.read_csv(f, sep=";", parse_dates=["data"])


def selecionar_features(df: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.Series]:
    colunas_feat = [c for c in df.columns if c not in NAO_FEATURES and c != alvo]
    # Remove colunas com muitos NaN (> 50%)
    colunas_feat = [c for c in colunas_feat if df[c].isna().mean() < 0.5]
    X = df[colunas_feat].copy()
    y = df[alvo].copy()

    # Preenche NaN com mediana (apenas para bandas brutas que podem estar ausentes)
    X = X.fillna(X.median(numeric_only=True))
    log.info("Features selecionadas: %d", len(colunas_feat))
    log.info("  %s", colunas_feat)
    return X, y


def avaliar_por_piquete(model, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, out: Path) -> None:
    linhas = []
    for pid in sorted(df["idpotreiro"].unique()):
        mask = df["idpotreiro"] == pid
        X_p, y_p = X[mask], y[mask]
        if len(y_p) < 3:
            continue
        yhat = model.predict(X_p)
        r2   = r2_score(y_p, yhat)
        rmse = np.sqrt(mean_squared_error(y_p, yhat))
        mae  = mean_absolute_error(y_p, yhat)
        linhas.append(f"  Piquete {pid}: R²={r2:.3f}  RMSE={rmse:.1f}  MAE={mae:.1f}  n={mask.sum()}")

    with open(out, "a") as f:
        f.write("\nPor piquete:\n")
        f.write("\n".join(linhas) + "\n")
    for l in linhas:
        log.info(l)


def plot_importancia(model, feature_names: list[str], out: Path) -> None:
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top-20 Features mais importantes (Random Forest)")
    ax.set_xlabel("Importância (Gini)")
    ax.axvline(imp.mean(), color="red", linestyle="--", alpha=0.6, label="média")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()
    log.info("Gráfico salvo: %s", out)


def plot_real_vs_predito(y_real: np.ndarray, y_pred: np.ndarray, alvo: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_real, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
    lim = [min(y_real.min(), y_pred.min()) * 0.95, max(y_real.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", label="1:1")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"{alvo} real")
    ax.set_ylabel(f"{alvo} predito")
    ax.set_title(f"Real × Predito — {alvo}")
    ax.legend()
    r2 = r2_score(y_real, y_pred)
    ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()
    log.info("Gráfico salvo: %s", out)


def plot_residuos(y_real: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    residuos = y_pred - y_real
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(y_pred, residuos, alpha=0.6, s=35, edgecolors="k", linewidths=0.3)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predito"); axes[0].set_ylabel("Resíduo (predito - real)")
    axes[0].set_title("Resíduos × Predito")

    axes[1].hist(residuos, bins=20, edgecolor="k", color="steelblue", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Resíduo"); axes[1].set_ylabel("Frequência")
    axes[1].set_title("Distribuição dos resíduos")

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()
    log.info("Gráfico salvo: %s", out)


def main() -> None:
    run_dir = BASE_DIR / "resultados_rf" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    df = carregar_dataset()
    log.info("Dataset: %d linhas | alvo: %s", len(df), ALVO)

    X, y = selecionar_features(df, ALVO)

    if len(y) < 20:
        log.warning("Poucos dados (%d linhas). Resultados podem ser instáveis.", len(y))

    # ── Validação cruzada (Leave-One-Piquete-Out quando possível) ─────────────
    # Usa KFold temporal: garante que treino nunca vê o futuro
    df_sorted = df.sort_values("data").reset_index(drop=True)
    X_s = X.loc[df_sorted.index]
    y_s = y.loc[df_sorted.index]

    model_cv = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    kf = KFold(n_splits=5, shuffle=False)  # sem shuffle = respeita ordem temporal
    cv_rmse = -cross_val_score(model_cv, X_s, y_s, cv=kf,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_r2   =  cross_val_score(model_cv, X_s, y_s, cv=kf,
                               scoring="r2", n_jobs=-1)

    log.info("Cross-val (5-fold temporal):")
    log.info("  RMSE : %.2f ± %.2f", cv_rmse.mean(), cv_rmse.std())
    log.info("  R²   : %.3f ± %.3f", cv_r2.mean(),  cv_r2.std())

    # ── Treino no conjunto completo ───────────────────────────────────────────
    model_final = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model_final.fit(X, y)
    y_pred = model_final.predict(X)

    rmse_treino = np.sqrt(mean_squared_error(y, y_pred))
    mae_treino  = mean_absolute_error(y, y_pred)
    r2_treino   = r2_score(y, y_pred)

    log.info("Treino completo:")
    log.info("  RMSE : %.2f", rmse_treino)
    log.info("  MAE  : %.2f", mae_treino)
    log.info("  R²   : %.3f", r2_treino)

    # ── Métricas em arquivo ───────────────────────────────────────────────────
    f_met = run_dir / "metricas.txt"
    with open(f_met, "w") as f:
        f.write(f"Alvo: {ALVO}\n")
        f.write(f"Data: {datetime.now()}\n")
        f.write(f"Registros de treino: {len(y)}\n\n")
        f.write("Cross-validation 5-fold temporal:\n")
        f.write(f"  RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}\n")
        f.write(f"  R²  : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}\n\n")
        f.write("Treino completo:\n")
        f.write(f"  RMSE: {rmse_treino:.2f}\n")
        f.write(f"  MAE : {mae_treino:.2f}\n")
        f.write(f"  R²  : {r2_treino:.3f}\n")
    avaliar_por_piquete(model_final, X, y, df, f_met)

    # ── Gráficos ──────────────────────────────────────────────────────────────
    plot_importancia(model_final, list(X.columns), run_dir / "importancia_features.png")
    plot_real_vs_predito(y.values, y_pred, ALVO, run_dir / "real_vs_predito.png")
    plot_residuos(y.values, y_pred, run_dir / "residuos.png")

    # ── Salva modelo ──────────────────────────────────────────────────────────
    joblib.dump(model_final, run_dir / "modelo_rf.joblib")
    log.info("Modelo salvo: %s", run_dir / "modelo_rf.joblib")

    log.info("=" * 55)
    log.info("Resultados em: %s", run_dir)
    log.info("=" * 55)


if __name__ == "__main__":
    main()
