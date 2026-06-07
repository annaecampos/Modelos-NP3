"""
Registro de experimentos de treino — qualificação / dissertação.

Cada execução de 04_modelo_rf.py grava:
  - resultados_rf/<timestamp>/run_manifest.json   (config + métricas daquela run)
  - resultados_rf/experimentos.jsonl            (histórico append-only)
  - resultados_rf/REGISTRO_EXPERIMENTOS.md        (tabela legível, regenerada)

Uso manual:
    python experimento_log.py --backfill   # importa runs antigas sem manifest
    python experimento_log.py --listar     # lista runs registradas
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent
RESULTADOS_DIR = BASE_DIR / "resultados_rf"
JSONL_PATH = RESULTADOS_DIR / "experimentos.jsonl"
REGISTRO_MD = RESULTADOS_DIR / "REGISTRO_EXPERIMENTOS.md"


def _git_info() -> dict[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=BASE_DIR.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE_DIR.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"branch": branch, "commit": commit}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"branch": "desconhecido", "commit": "desconhecido"}


def _ler_jsonl() -> list[dict[str, Any]]:
    if not JSONL_PATH.exists():
        return []
    runs = []
    for linha in JSONL_PATH.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha:
            runs.append(json.loads(linha))
    return runs


def _escrever_jsonl(runs: list[dict[str, Any]]) -> None:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")


def _manifest_path(run_id: str) -> Path:
    return RESULTADOS_DIR / run_id / "run_manifest.json"


def carregar_manifest(run_id: str) -> dict[str, Any] | None:
    path = _manifest_path(run_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def comparar_com_anterior(atual: dict[str, Any], anterior: dict[str, Any] | None) -> list[str]:
    if anterior is None:
        return ["Primeiro experimento registrado (sem run anterior para comparar)."]

    mudancas: list[str] = []
    chaves_config = [
        ("protocolo", "Protocolo de avaliação"),
        ("dataset_arquivo", "Dataset"),
        ("alvo", "Alvo"),
        ("split_treino_fim", "Fim do treino"),
        ("split_valid_fim", "Fim da validação"),
        ("n_estimators", "Árvores (n_estimators)"),
        ("max_features", "max_features"),
        ("min_samples_leaf", "min_samples_leaf"),
        ("random_state", "random_state"),
    ]
    cfg_a = atual.get("config", {})
    cfg_p = anterior.get("config", {})

    for chave, rotulo in chaves_config:
        va, vp = cfg_a.get(chave), cfg_p.get(chave)
        if va != vp:
            mudancas.append(f"{rotulo}: {vp!r} → {va!r}")

    fa = set(atual.get("features", []))
    fp = set(anterior.get("features", []))
    if fa != fp:
        novas = sorted(fa - fp)
        removidas = sorted(fp - fa)
        if novas:
            mudancas.append(f"Features novas ({len(novas)}): {', '.join(novas[:8])}" +
                            ("…" if len(novas) > 8 else ""))
        if removidas:
            mudancas.append(f"Features removidas ({len(removidas)}): {', '.join(removidas[:8])}" +
                            ("…" if len(removidas) > 8 else ""))

    ma = atual.get("metricas", {})
    mp = anterior.get("metricas", {})
    for conjunto in ("teste", "treino", "validacao", "cv"):
        if conjunto in ma and conjunto in mp:
            r2_a = ma[conjunto].get("r2")
            r2_p = mp[conjunto].get("r2")
            if r2_a is not None and r2_p is not None and r2_a != r2_p:
                mudancas.append(f"R² {conjunto}: {r2_p:.3f} → {r2_a:.3f}")

    if atual.get("notas") and atual.get("notas") != anterior.get("notas"):
        mudancas.append(f"Notas: {anterior.get('notas') or '(vazio)'} → {atual.get('notas')}")

    if not mudancas:
        mudancas.append("Configuração e features iguais à run anterior (métricas podem repetir).")
    return mudancas


def registrar_experimento(manifest: dict[str, Any]) -> dict[str, Any]:
    """Salva manifest na pasta da run, atualiza jsonl e REGISTRO_EXPERIMENTOS.md."""
    run_id = manifest["run_id"]
    run_dir = RESULTADOS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    runs = _ler_jsonl()
    anterior = runs[-1] if runs else None
    manifest["mudancas_vs_anterior"] = comparar_com_anterior(manifest, anterior)
    manifest["registrado_em"] = datetime.now(timezone.utc).isoformat()

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    runs = [r for r in runs if r.get("run_id") != run_id]
    runs.append(manifest)
    runs.sort(key=lambda r: r.get("run_id", ""))
    _escrever_jsonl(runs)
    regenerar_registro_md(runs)
    return manifest


def regenerar_registro_md(runs: list[dict[str, Any]] | None = None) -> None:
    if runs is None:
        runs = _ler_jsonl()

    linhas = [
        "# Registro de experimentos — Random Forest (mstotal)",
        "",
        "Histórico automático de treinos. **Não apague pastas em `resultados_rf/`** — "
        "cada subpasta é um experimento documentado.",
        "",
        f"Última atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Índice rápido",
        "",
        "| Run ID | Data | Protocolo | Dataset | n | Teste R² | Teste RMSE |",
        "|--------|------|-----------|---------|---|----------|------------|",
    ]

    for run in runs:
        cfg = run.get("config", {})
        met = run.get("metricas", {})
        teste = met.get("teste", {})
        cv = met.get("cv", {})
        n_total = run.get("dataset", {}).get("linhas_total", "—")
        protocolo = cfg.get("protocolo", "—")
        dataset = cfg.get("dataset_arquivo", "—")
        data_run = run.get("executado_em", run.get("run_id", ""))[:19]
        if teste.get("r2") is not None:
            r2s = f"{teste['r2']:.3f}"
            rmses = f"{teste.get('rmse', 0):.0f}"
        elif cv.get("r2") is not None:
            r2s = f"{cv['r2']:.3f} (CV)"
            rmses = f"{cv.get('rmse', 0):.0f} (CV)"
        else:
            r2s = rmses = "—"
        linhas.append(
            f"| `{run.get('run_id', '')}` | {data_run} | {protocolo} | {dataset} | {n_total} | {r2s} | {rmses} |"
        )

    linhas.extend(["", "## Detalhes por experimento", ""])

    for i, run in enumerate(runs):
        cfg = run.get("config", {})
        met = run.get("metricas", {})
        ds = run.get("dataset", {})
        linhas.append(f"### {run.get('run_id', '')}")
        linhas.append("")
        linhas.append(f"- **Executado em:** {run.get('executado_em', '—')}")
        linhas.append(f"- **Git:** `{run.get('git', {}).get('branch', '—')}` @ `{run.get('git', {}).get('commit', '—')}`")
        linhas.append(f"- **Protocolo:** {cfg.get('protocolo', '—')}")
        linhas.append(f"- **Dataset:** `{cfg.get('dataset_arquivo', '—')}` ({ds.get('linhas_total', '—')} linhas)")
        if ds.get("treino") is not None:
            linhas.append(
                f"- **Split:** treino={ds.get('treino')} | val={ds.get('validacao')} | teste={ds.get('teste')}"
            )
        linhas.append(f"- **Alvo:** `{cfg.get('alvo', '—')}`")
        linhas.append(f"- **Features:** {len(run.get('features', []))} colunas")
        if run.get("notas"):
            linhas.append(f"- **Notas:** {run['notas']}")

        linhas.append("- **Métricas:**")
        for nome, valores in met.items():
            if isinstance(valores, dict) and "r2" in valores:
                extra = ""
                if "r2_std" in valores:
                    extra = f" ± {valores['r2_std']:.3f}"
                linhas.append(
                    f"  - **{nome}** (n={valores.get('n', '—')}): "
                    f"R²={valores.get('r2', 0):.3f}{extra}, RMSE={valores.get('rmse', 0):.0f}"
                )

        linhas.append("- **Arquivos:**")
        for arq in run.get("arquivos", []):
            linhas.append(f"  - `{arq}`")

        mudancas = run.get("mudancas_vs_anterior", [])
        if mudancas:
            linhas.append("- **Mudanças em relação ao experimento anterior:**")
            for m in mudancas:
                linhas.append(f"  - {m}")

        linhas.append("")
        if i < len(runs) - 1:
            linhas.append("---")
            linhas.append("")

    linhas.extend([
        "## Como registrar o próximo treino",
        "",
        "1. (Opcional) Edite `RUN_NOTAS` em `config.py` descrevendo o que mudou.",
        "2. Execute `python 04_modelo_rf.py`.",
        "3. Confira esta página e `experimentos.jsonl`.",
        "",
        "## Regra de ouro",
        "",
        "**Nunca apague** pastas em `resultados_rf/`. Novos treinos criam **nova** subpasta; os antigos permanecem.",
    ])

    REGISTRO_MD.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def _parse_metricas_txt(texto: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Extrai config e métricas de metricas.txt (formatos antigo CV e novo split)."""
    config: dict[str, str] = {}
    metricas: dict[str, Any] = {}

    if m := re.search(r"^Dataset:\s*(.+)$", texto, re.M):
        config["dataset_arquivo"] = m.group(1).strip()
    if m := re.search(r"^Alvo:\s*(.+)$", texto, re.M):
        config["alvo"] = m.group(1).strip()
    if "Cross-validation" in texto:
        config["protocolo"] = "cv_5fold_temporal + treino_completo"
    elif "Split temporal" in texto:
        config["protocolo"] = "split_treino_validacao_teste"
        if m := re.search(r"treino ≤ (.+?) \|", texto):
            config["split_treino_fim"] = m.group(1).strip()
        if m := re.search(r"data ≤ (.+?) \|", texto):
            config["split_valid_fim"] = m.group(1).strip()

    if m := re.search(
        r"Cross-validation.*?\n\s*RMSE:\s*([\d.]+).*?\n\s*R²\s*:\s*([\d.]+)",
        texto, re.S,
    ):
        metricas["cv"] = {
            "rmse": float(m.group(1)),
            "r2": float(m.group(2)),
            "n": None,
        }
        if m2 := re.search(r"R²\s*:\s*[\d.]+\s*±\s*([\d.]+)", texto):
            metricas["cv"]["r2_std"] = float(m2.group(1))

    for bloco, chave in [
        (r"Treino(?: completo)? \(n=(\d+)\):\s*\n\s*RMSE:\s*([\d.]+).*?R²\s*:\s*([\d.]+)", "treino"),
        (r"Validação \(n=(\d+)\):\s*\n\s*RMSE:\s*([\d.]+).*?R²\s*:\s*([-\d.]+)", "validacao"),
        (r"Teste \(n=(\d+)\):\s*\n\s*RMSE:\s*([\d.]+).*?R²\s*:\s*([-\d.]+)", "teste"),
    ]:
        if m := re.search(bloco, texto, re.S):
            metricas[chave] = {
                "n": int(m.group(1)),
                "rmse": float(m.group(2)),
                "r2": float(m.group(3)),
            }

    if m := re.search(r"Registros de treino:\s*(\d+)", texto):
        config["registros_treino_cv"] = m.group(1)

    return metricas, config


def backfill_runs() -> int:
    """Cria manifests para pastas antigas que só têm metricas.txt."""
    if not RESULTADOS_DIR.exists():
        return 0

    runs_existentes = {r["run_id"] for r in _ler_jsonl()}
    novos = 0

    for pasta in sorted(RESULTADOS_DIR.iterdir()):
        if not pasta.is_dir() or not re.match(r"^\d{8}_\d{6}$", pasta.name):
            continue
        run_id = pasta.name
        if (pasta / "run_manifest.json").exists():
            manifest = carregar_manifest(run_id)
            if manifest and run_id not in runs_existentes:
                runs_existentes.add(run_id)
                novos += 1
                registrar_experimento(manifest)
            continue

        f_met = pasta / "metricas.txt"
        if not f_met.exists():
            continue

        texto = f_met.read_text(encoding="utf-8")
        metricas, cfg_parsed = _parse_metricas_txt(texto)

        if m := re.search(r"^Data:\s*(.+)$", texto, re.M):
            executado_em = m.group(1).strip()
        else:
            executado_em = datetime.strptime(run_id, "%Y%m%d_%H%M%S").isoformat()

        notas_backfill = {
            "20260521_123457": "Baseline Ana — dataset_final.csv sem limpeza (1463 linhas, duplicatas). CV 5-fold.",
            "20260527_174135": "Baseline Rafael/docs — dataset corrigido 680 linhas. CV 5-fold.",
            "20260529_081748": "Mesmo dataset 680 linhas + variáveis de animais. CV 5-fold.",
            "20260529_095329": "Primeiro treino com split treino/val/teste (orientação do professor).",
            "20260606_100433": "Repetição pós-merge docs→main; mesmo split e mesmo resultado.",
        }

        manifest: dict[str, Any] = {
            "run_id": run_id,
            "executado_em": executado_em,
            "script": "04_modelo_rf.py",
            "git": {"branch": "backfill", "commit": "backfill"},
            "notas": notas_backfill.get(run_id, "Importado retroativamente (backfill)."),
            "config": {
                "protocolo": cfg_parsed.get("protocolo", "desconhecido"),
                "dataset_arquivo": cfg_parsed.get(
                    "dataset_arquivo",
                    "dataset_final.csv" if "1463" in texto else "dataset_final_corrigido_sem_outlier.csv",
                ),
                "alvo": cfg_parsed.get("alvo", "mstotal"),
                "split_treino_fim": cfg_parsed.get("split_treino_fim"),
                "split_valid_fim": cfg_parsed.get("split_valid_fim"),
                "n_estimators": 500 if "Split temporal" in texto else "300_cv + 500_final",
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "random_state": 42,
            },
            "dataset": {
                "linhas_total": int(cfg_parsed["registros_treino_cv"]) if "registros_treino_cv" in cfg_parsed else (
                    sum(
                        metricas.get(k, {}).get("n") or 0
                        for k in ("treino", "validacao", "teste")
                    ) or None
                ),
                "treino": metricas.get("treino", {}).get("n"),
                "validacao": metricas.get("validacao", {}).get("n"),
                "teste": metricas.get("teste", {}).get("n"),
            },
            "features": [],
            "metricas": metricas,
            "arquivos": sorted(
                str(p.relative_to(RESULTADOS_DIR / run_id))
                for p in pasta.iterdir()
                if p.is_file()
            ),
            "backfill": True,
        }
        registrar_experimento(manifest)
        novos += 1

    return novos


def montar_manifest(
    *,
    run_id: str,
    executado_em: str,
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    features: list[str],
    metricas: dict[str, Any],
    arquivos: list[str],
    notas: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "executado_em": executado_em,
        "script": "04_modelo_rf.py",
        "git": _git_info(),
        "notas": notas,
        "config": config,
        "dataset": dataset_info,
        "features": features,
        "metricas": metricas,
        "arquivos": arquivos,
        "backfill": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Registro de experimentos RF")
    parser.add_argument("--backfill", action="store_true", help="Importa runs antigas")
    parser.add_argument("--listar", action="store_true", help="Lista runs registradas")
    args = parser.parse_args()

    if args.backfill:
        n = backfill_runs()
        print(f"Backfill concluído: {n} run(s) processada(s).")
        print(f"Registro: {REGISTRO_MD}")
    elif args.listar:
        for run in _ler_jsonl():
            cfg = run.get("config", {})
            teste = run.get("metricas", {}).get("teste", {})
            r2 = teste.get("r2", "—")
            print(f"{run['run_id']}  {cfg.get('protocolo', '—')}  teste_R2={r2}")
    else:
        parser.print_help()
