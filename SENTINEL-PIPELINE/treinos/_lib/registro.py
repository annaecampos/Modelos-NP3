"""Registro de experimentos por pasta de treino (qualificação / dissertação)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_info(repo_root: Path) -> dict[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return {"branch": branch, "commit": commit}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"branch": "desconhecido", "commit": "desconhecido"}


def comparar_com_anterior(atual: dict[str, Any], anterior: dict[str, Any] | None) -> list[str]:
    if anterior is None:
        return ["Primeiro experimento nesta pasta de treino."]
    mudancas: list[str] = []
    for chave, rotulo in [
        ("dataset_arquivo", "Dataset"),
        ("max_dias_campo_imagem", "Máx. dias campo↔imagem"),
        ("split_treino_fim", "Fim treino"),
        ("split_valid_fim", "Fim validação"),
    ]:
        ca, cp = atual.get("config", {}).get(chave), anterior.get("config", {}).get(chave)
        if ca != cp:
            mudancas.append(f"{rotulo}: {cp!r} → {ca!r}")
    mt, mp = atual.get("metricas", {}).get("teste", {}), anterior.get("metricas", {}).get("teste", {})
    if mt.get("r2") is not None and mp.get("r2") is not None and mt["r2"] != mp["r2"]:
        mudancas.append(f"R² teste: {mp['r2']:.3f} → {mt['r2']:.3f}")
    if atual.get("notas") and atual.get("notas") != anterior.get("notas"):
        mudancas.append(f"Notas atualizadas.")
    return mudancas or ["Mesma configuração da run anterior (métricas podem variar por sorteio)."]


class RegistroTreino:
    def __init__(self, treino_tipo: str, treino_dir: Path, repo_root: Path):
        self.treino_tipo = treino_tipo
        self.treino_dir = treino_dir
        self.resultados_dir = treino_dir / "resultados"
        self.jsonl_path = treino_dir / "experimentos.jsonl"
        self.registro_md = treino_dir / "REGISTRO.md"
        self.repo_root = repo_root

    def _ler(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        return [json.loads(ln) for ln in self.jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def registrar(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self.resultados_dir.mkdir(parents=True, exist_ok=True)
        runs = self._ler()
        anterior = runs[-1] if runs else None
        manifest["tipo_treino"] = self.treino_tipo
        manifest["mudancas_vs_anterior"] = comparar_com_anterior(manifest, anterior)
        manifest["registrado_em"] = datetime.now(timezone.utc).isoformat()
        manifest["git"] = git_info(self.repo_root)

        run_dir = self.resultados_dir / manifest["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

        runs = [r for r in runs if r.get("run_id") != manifest["run_id"]]
        runs.append(manifest)
        runs.sort(key=lambda r: r["run_id"])
        with self.jsonl_path.open("w", encoding="utf-8") as f:
            for r in runs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        self._regenerar_md(runs)
        self._atualizar_registro_geral()
        return manifest

    def _regenerar_md(self, runs: list[dict[str, Any]]) -> None:
        linhas = [
            f"# Registro — {self.treino_tipo}",
            "",
            f"Pasta: `{self.treino_dir.name}/`",
            "",
            "| Run | Data | n | Teste R² | Teste RMSE |",
            "|-----|------|---|----------|------------|",
        ]
        for r in runs:
            t = r.get("metricas", {}).get("teste", {})
            r2 = f"{t.get('r2', 0):.3f}" if t.get("r2") is not None else "—"
            rmse = f"{t.get('rmse', 0):.0f}" if t.get("rmse") is not None else "—"
            n = r.get("dataset", {}).get("linhas_total", "—")
            linhas.append(
                f"| `{r['run_id']}` | {r.get('executado_em', '')[:19]} | {n} | {r2} | {rmse} |"
            )
        for r in runs:
            linhas.extend(["", f"## {r['run_id']}", ""])
            if r.get("notas"):
                linhas.append(f"- **Notas:** {r['notas']}")
            for m in r.get("mudancas_vs_anterior", []):
                linhas.append(f"- {m}")
        self.registro_md.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    def _atualizar_registro_geral(self) -> None:
        geral = self.treino_dir.parent / "REGISTRO_GERAL.md"
        tipos = sorted(p for p in self.treino_dir.parent.iterdir() if p.is_dir() and p.name != "_lib")
        linhas = ["# Registro geral de treinos", ""]
        for pasta in tipos:
            reg = pasta / "REGISTRO.md"
            linhas.append(f"- **{pasta.name}** — ver [`{pasta.name}/REGISTRO.md`]({pasta.name}/REGISTRO.md)")
        geral.write_text("\n".join(linhas) + "\n", encoding="utf-8")
