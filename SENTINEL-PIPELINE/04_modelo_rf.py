"""
Modelo baseline: Random Forest Regressor para estimativa de forragem.

Split temporal obrigatório (treino / validação / teste) — ver config.py.
  - Treino   : aprende o modelo
  - Validação: diagnóstico intermediário (ajuste futuro de hiperparâmetros)
  - Teste    : avaliação final honesta; gráficos usam só este conjunto

Saída em resultados_rf/<timestamp>/:
  - metricas.txt
  - run_manifest.json     (config + métricas + diff vs run anterior)
  - importancia_features.png
  - real_vs_predito.png   (conjunto de teste)
  - residuos.png          (conjunto de teste)
  - modelo_rf.joblib

Histórico central (nunca apagar):
  - resultados_rf/experimentos.jsonl
  - resultados_rf/REGISTRO_EXPERIMENTOS.md

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    ALVO,
    BASE_DIR,
    DATA_DIR,
    DATASET_ARQUIVO,
    RUN_NOTAS,
    SPLIT_TREINO_FIM,
    SPLIT_VALID_FIM,
)
from experimento_log import montar_manifest, registrar_experimento

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NAO_FEATURES = {
    "data", "idpotreiro", "idsubarea", "data_imagem",
    "mstotal", "msanoni", "msoutras", "media",
    "geometria",
}


def carregar_dataset() -> pd.DataFrame:
    f = DATA_DIR / DATASET_ARQUIVO
    if not f.exists():
        log.error("Arquivo não encontrado: %s\nExecute primeiro: python 03_monta_dataset.py", f)
        raise SystemExit(1)
    df = pd.read_csv(f, sep=";", parse_dates=["data"])
    return df.sort_values("data").reset_index(drop=True)


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fim_treino = pd.Timestamp(SPLIT_TREINO_FIM)
    fim_valid = pd.Timestamp(SPLIT_VALID_FIM)

    treino = df[df["data"] <= fim_treino].copy()
    valid = df[(df["data"] > fim_treino) & (df["data"] <= fim_valid)].copy()
    teste = df[df["data"] > fim_valid].copy()

    if len(treino) < 10 or len(valid) < 5 or len(teste) < 5:
        log.error(
            "Split temporal inválido (treino=%d, val=%d, teste=%d). "
            "Ajuste SPLIT_TREINO_FIM / SPLIT_VALID_FIM em config.py.",
            len(treino), len(valid), len(teste),
        )
        raise SystemExit(1)

    log.info(
        "Split temporal: treino=%d (%s → %s) | val=%d (%s → %s) | teste=%d (%s → %s)",
        len(treino), treino["data"].min().date(), treino["data"].max().date(),
        len(valid), valid["data"].min().date(), valid["data"].max().date(),
        len(teste), teste["data"].min().date(), teste["data"].max().date(),
    )
    return treino, valid, teste


def preparar_features(
    df_treino: pd.DataFrame,
    dfs: list[pd.DataFrame],
    alvo: str,
) -> tuple[list[str], list[pd.DataFrame], list[pd.Series]]:
    """Seleciona colunas e imputa NaN usando estatísticas só do treino."""
    candidatas = [c for c in df_treino.columns if c not in NAO_FEATURES and c != alvo]
    candidatas = [c for c in candidatas if df_treino[c].isna().mean() < 0.5]
    medianas = df_treino[candidatas].median(numeric_only=True)

    X_list, y_list = [], []
    for df in dfs:
        X = df[candidatas].copy().fillna(medianas)
        y = df[alvo].copy()
        X_list.append(X)
        y_list.append(y)

    log.info("Features selecionadas (%d): %s", len(candidatas), candidatas)
    return candidatas, X_list, y_list


def metricas_conjunto(nome: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    log.info("%s: R²=%.3f  RMSE=%.1f  MAE=%.1f  n=%d", nome, r2, rmse, mae, len(y_true))
    return {"nome": nome, "r2": r2, "rmse": rmse, "mae": mae, "n": len(y_true)}


def avaliar_por_piquete(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    out: Path,
    titulo: str,
) -> None:
    linhas = [f"\nPor piquete ({titulo}):"]
    for pid in sorted(df["idpotreiro"].unique()):
        mask = df["idpotreiro"] == pid
        if mask.sum() < 3:
            continue
        yhat = model.predict(X[mask])
        r2 = r2_score(y[mask], yhat)
        rmse = np.sqrt(mean_squared_error(y[mask], yhat))
        mae = mean_absolute_error(y[mask], yhat)
        linhas.append(f"  Piquete {pid}: R²={r2:.3f}  RMSE={rmse:.1f}  MAE={mae:.1f}  n={mask.sum()}")

    with open(out, "a") as f:
        f.write("\n".join(linhas) + "\n")
    for linha in linhas[1:]:
        log.info(linha)


def plot_importancia(model, feature_names: list[str], out: Path) -> None:
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top-20 Features — treino")
    ax.set_xlabel("Importância (Gini)")
    ax.axvline(imp.mean(), color="red", linestyle="--", alpha=0.6, label="média")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()
    log.info("Gráfico salvo: %s", out)


def plot_real_vs_predito(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    alvo: str,
    out: Path,
    titulo_conjunto: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_real, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
    lim = [min(y_real.min(), y_pred.min()) * 0.95, max(y_real.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"{alvo} real")
    ax.set_ylabel(f"{alvo} predito")
    ax.set_title(f"Real × Predito — {titulo_conjunto}")
    ax.legend()
    r2 = r2_score(y_real, y_pred)
    ax.text(
        0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()
    log.info("Gráfico salvo: %s", out)


def plot_residuos(y_real: np.ndarray, y_pred: np.ndarray, out: Path, titulo_conjunto: str) -> None:
    residuos = y_pred - y_real
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(y_pred, residuos, alpha=0.6, s=35, edgecolors="k", linewidths=0.3)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predito")
    axes[0].set_ylabel("Resíduo (predito - real)")
    axes[0].set_title(f"Resíduos × Predito — {titulo_conjunto}")

    axes[1].hist(residuos, bins=20, edgecolor="k", color="steelblue", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Resíduo")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title(f"Distribuição — {titulo_conjunto}")

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()
    log.info("Gráfico salvo: %s", out)


def escrever_metricas(
    path: Path,
    resultados: list[dict],
    split_info: str,
) -> None:
    with open(path, "w") as f:
        f.write(f"Alvo: {ALVO}\n")
        f.write(f"Data: {datetime.now()}\n")
        f.write(f"Dataset: {DATASET_ARQUIVO}\n\n")
        f.write("Split temporal (treino / validação / teste):\n")
        f.write(f"  {split_info}\n\n")
        for r in resultados:
            f.write(f"{r['nome']} (n={r['n']}):\n")
            f.write(f"  RMSE: {r['rmse']:.2f}\n")
            f.write(f"  MAE : {r['mae']:.2f}\n")
            f.write(f"  R²  : {r['r2']:.3f}\n\n")
        f.write("Métrica principal da dissertação: TESTE (conjunto nunca usado no treino).\n")


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    executado_em = datetime.now().isoformat()
    run_dir = BASE_DIR / "resultados_rf" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    df = carregar_dataset()
    log.info("Dataset: %d linhas | alvo: %s", len(df), ALVO)

    treino, valid, teste = split_temporal(df)
    _, [X_tr, X_va, X_te], [y_tr, y_va, y_te] = preparar_features(
        treino, [treino, valid, teste], ALVO,
    )

    hiperparametros = {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestRegressor(**hiperparametros)
    model.fit(X_tr, y_tr)

    resultados = []
    for nome, X, y, part in [
        ("Treino", X_tr, y_tr, treino),
        ("Validação", X_va, y_va, valid),
        ("Teste", X_te, y_te, teste),
    ]:
        pred = model.predict(X)
        resultados.append(metricas_conjunto(nome, y, pred))

    split_info = (
        f"treino ≤ {SPLIT_TREINO_FIM} | "
        f"validação {SPLIT_TREINO_FIM} < data ≤ {SPLIT_VALID_FIM} | "
        f"teste > {SPLIT_VALID_FIM}"
    )
    f_met = run_dir / "metricas.txt"
    escrever_metricas(f_met, resultados, split_info)
    avaliar_por_piquete(model, X_te, y_te, teste, f_met, "teste")

    y_pred_te = model.predict(X_te)
    plot_importancia(model, list(X_tr.columns), run_dir / "importancia_features.png")
    plot_real_vs_predito(y_te.values, y_pred_te, ALVO, run_dir / "real_vs_predito.png", "teste")
    plot_residuos(y_te.values, y_pred_te, run_dir / "residuos.png", "teste")

    joblib.dump(model, run_dir / "modelo_rf.joblib")
    log.info("Modelo salvo: %s", run_dir / "modelo_rf.joblib")

    metricas_log = {
        "treino": {
            "n": resultados[0]["n"], "r2": resultados[0]["r2"],
            "rmse": resultados[0]["rmse"], "mae": resultados[0]["mae"],
        },
        "validacao": {
            "n": resultados[1]["n"], "r2": resultados[1]["r2"],
            "rmse": resultados[1]["rmse"], "mae": resultados[1]["mae"],
        },
        "teste": {
            "n": resultados[2]["n"], "r2": resultados[2]["r2"],
            "rmse": resultados[2]["rmse"], "mae": resultados[2]["mae"],
        },
    }
    arquivos = sorted(p.name for p in run_dir.iterdir() if p.is_file())
    manifest = montar_manifest(
        run_id=run_id,
        executado_em=executado_em,
        config={
            "protocolo": "split_treino_validacao_teste",
            "dataset_arquivo": DATASET_ARQUIVO,
            "alvo": ALVO,
            "split_treino_fim": SPLIT_TREINO_FIM,
            "split_valid_fim": SPLIT_VALID_FIM,
            **hiperparametros,
        },
        dataset_info={
            "linhas_total": len(df),
            "treino": len(treino),
            "validacao": len(valid),
            "teste": len(teste),
            "periodo_treino": f"{treino['data'].min().date()} → {treino['data'].max().date()}",
            "periodo_teste": f"{teste['data'].min().date()} → {teste['data'].max().date()}",
        },
        features=list(X_tr.columns),
        metricas=metricas_log,
        arquivos=arquivos,
        notas=RUN_NOTAS,
    )
    manifest = registrar_experimento(manifest)
    log.info("Registro salvo: resultados_rf/experimentos.jsonl")
    if manifest["mudancas_vs_anterior"]:
        log.info("Mudanças vs experimento anterior:")
        for m in manifest["mudancas_vs_anterior"]:
            log.info("  • %s", m)

    log.info("=" * 55)
    log.info("Resultados em: %s", run_dir)
    log.info("=" * 55)


if __name__ == "__main__":
    main()
