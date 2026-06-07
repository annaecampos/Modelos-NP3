"""Treino Random Forest com split temporal — usado pelas pastas em treinos/."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from registro import RegistroTreino

log = logging.getLogger(__name__)

NAO_FEATURES = {
    "data", "idpotreiro", "idsubarea", "data_imagem",
    "mstotal", "msanoni", "msoutras", "media", "geometria",
}


def split_temporal(df: pd.DataFrame, fim_treino: str, fim_valid: str):
    ft, fv = pd.Timestamp(fim_treino), pd.Timestamp(fim_valid)
    treino = df[df["data"] <= ft].copy()
    valid = df[(df["data"] > ft) & (df["data"] <= fv)].copy()
    teste = df[df["data"] > fv].copy()
    return treino, valid, teste


def preparar_features(df_treino: pd.DataFrame, dfs: list[pd.DataFrame], alvo: str):
    candidatas = [c for c in df_treino.columns if c not in NAO_FEATURES and c != alvo]
    candidatas = [c for c in candidatas if df_treino[c].isna().mean() < 0.5]
    medianas = df_treino[candidatas].median(numeric_only=True)
    X_list, y_list = [], []
    for df in dfs:
        X_list.append(df[candidatas].copy().fillna(medianas))
        y_list.append(df[alvo].copy())
    return candidatas, X_list, y_list


def _plot_importancia(model, names, out):
    imp = pd.Series(model.feature_importances_, index=names).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top-20 Features — treino")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def _plot_scatter(y, pred, alvo, out, titulo):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, pred, alpha=0.6, s=40, edgecolors="k", linewidths=0.3)
    lim = [min(y.min(), pred.min()) * 0.95, max(y.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "r--")
    ax.set_xlabel(f"{alvo} real")
    ax.set_ylabel(f"{alvo} predito")
    ax.set_title(f"Real × Predito — {titulo}")
    ax.text(0.05, 0.92, f"R² = {r2_score(y, pred):.3f}", transform=ax.transAxes)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def _plot_residuos(y, pred, out, titulo):
    res = pred - y
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(pred, res, alpha=0.6, s=35)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"Resíduos — {titulo}")
    axes[1].hist(res, bins=20, edgecolor="k", color="steelblue", alpha=0.8)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def executar_treino(cfg) -> Path:
    """cfg = módulo config_treino da pasta específica."""
    treino_dir = Path(cfg.TREINO_DIR)
    dataset_path = treino_dir / "dados" / cfg.DATASET_ARQUIVO
    if not dataset_path.exists():
        log.error("Dataset não encontrado: %s", dataset_path)
        raise SystemExit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = treino_dir / "resultados" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, sep=";", parse_dates=["data"]).sort_values("data")
    treino, valid, teste = split_temporal(df, cfg.SPLIT_TREINO_FIM, cfg.SPLIT_VALID_FIM)
    _, [X_tr, X_va, X_te], [y_tr, y_va, y_te] = preparar_features(
        treino, [treino, valid, teste], cfg.ALVO,
    )

    hip = {
        "n_estimators": cfg.N_ESTIMATORS,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestRegressor(**hip)
    model.fit(X_tr, y_tr)

    resultados = []
    for nome, X, y in [("Treino", X_tr, y_tr), ("Validação", X_va, y_va), ("Teste", X_te, y_te)]:
        pred = model.predict(X)
        resultados.append({
            "nome": nome, "n": len(y),
            "r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
        })
        log.info("%s: R²=%.3f RMSE=%.1f n=%d", nome, resultados[-1]["r2"], resultados[-1]["rmse"], len(y))

    f_met = run_dir / "metricas.txt"
    with f_met.open("w") as f:
        f.write(f"Tipo treino: {cfg.TIPO_TREINO}\n")
        f.write(f"Dataset: {cfg.DATASET_ARQUIVO}\n")
        f.write(f"Data: {datetime.now()}\n\n")
        for r in resultados:
            f.write(f"{r['nome']} (n={r['n']}): R²={r['r2']:.3f} RMSE={r['rmse']:.1f}\n")

    y_pred = model.predict(X_te)
    _plot_importancia(model, list(X_tr.columns), run_dir / "importancia_features.png")
    _plot_scatter(y_te.values, y_pred, cfg.ALVO, run_dir / "real_vs_predito.png", "teste")
    _plot_residuos(y_te.values, y_pred, run_dir / "residuos.png", "teste")
    joblib.dump(model, run_dir / "modelo_rf.joblib")

    reg = RegistroTreino(cfg.TIPO_TREINO, treino_dir, treino_dir.parents[3])
    reg.registrar({
        "run_id": run_id,
        "executado_em": datetime.now().isoformat(),
        "script": "treinar.py",
        "notas": cfg.RUN_NOTAS,
        "config": {
            "protocolo": "split_treino_validacao_teste",
            "dataset_arquivo": cfg.DATASET_ARQUIVO,
            "alvo": cfg.ALVO,
            "split_treino_fim": cfg.SPLIT_TREINO_FIM,
            "split_valid_fim": cfg.SPLIT_VALID_FIM,
            "max_dias_campo_imagem": getattr(cfg, "MAX_DIAS_CAMPO_IMAGEM", None),
            **hip,
        },
        "dataset": {
            "linhas_total": len(df),
            "treino": len(treino),
            "validacao": len(valid),
            "teste": len(teste),
        },
        "features": list(X_tr.columns),
        "metricas": {
            "treino": resultados[0],
            "validacao": resultados[1],
            "teste": resultados[2],
        },
        "arquivos": sorted(p.name for p in run_dir.iterdir() if p.is_file()),
    })

    log.info("Resultados: %s", run_dir)
    return run_dir
