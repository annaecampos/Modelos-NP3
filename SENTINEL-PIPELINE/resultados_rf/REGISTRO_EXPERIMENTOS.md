# Registro de experimentos — Random Forest (mstotal)

Histórico automático de treinos. **Não apague pastas em `resultados_rf/`** — cada subpasta é um experimento documentado.

Última atualização: 2026-06-07 18:53:20

## Índice rápido

| Run ID | Data | Protocolo | Dataset | n | Teste R² | Teste RMSE |
|--------|------|-----------|---------|---|----------|------------|
| `20260521_123457` | 2026-05-21 12:35:07 | cv_5fold_temporal + treino_completo | dataset_final.csv | 1463 | 0.139 (CV) | 3197 (CV) |
| `20260527_174135` | 2026-05-27 17:41:39 | cv_5fold_temporal + treino_completo | dataset_final_corrigido_sem_outlier.csv | 680 | 0.219 (CV) | 2113 (CV) |
| `20260529_081748` | 2026-05-29 08:18:10 | cv_5fold_temporal + treino_completo | dataset_final_corrigido_sem_outlier.csv | 680 | 0.219 (CV) | 2113 (CV) |
| `20260529_095329` | 2026-05-29 09:53:33 | split_treino_validacao_teste | dataset_final_corrigido_sem_outlier.csv | 680 | 0.108 | 2275 |
| `20260606_100433` | 2026-06-06 10:04:35 | split_treino_validacao_teste | dataset_final_corrigido_sem_outlier.csv | 680 | 0.108 | 2275 |

## Detalhes por experimento

### 20260521_123457

- **Executado em:** 2026-05-21 12:35:07.361702
- **Git:** `backfill` @ `backfill`
- **Protocolo:** cv_5fold_temporal + treino_completo
- **Dataset:** `dataset_final.csv` (1463 linhas)
- **Alvo:** `mstotal`
- **Features:** 0 colunas
- **Notas:** Baseline Ana — dataset_final.csv sem limpeza (1463 linhas, duplicatas). CV 5-fold.
- **Métricas:**
  - **cv** (n=None): R²=0.139 ± 0.259, RMSE=3197
- **Arquivos:**
  - `importancia_features.png`
  - `metricas.txt`
  - `real_vs_predito.png`
  - `residuos.png`
- **Mudanças em relação ao experimento anterior:**
  - Primeiro experimento registrado (sem run anterior para comparar).

---

### 20260527_174135

- **Executado em:** 2026-05-27 17:41:39.070199
- **Git:** `backfill` @ `backfill`
- **Protocolo:** cv_5fold_temporal + treino_completo
- **Dataset:** `dataset_final_corrigido_sem_outlier.csv` (680 linhas)
- **Alvo:** `mstotal`
- **Features:** 0 colunas
- **Notas:** Baseline Rafael/docs — dataset corrigido 680 linhas. CV 5-fold.
- **Métricas:**
  - **cv** (n=None): R²=0.219 ± 0.105, RMSE=2113
- **Arquivos:**
  - `importancia_features.png`
  - `metricas.txt`
  - `real_vs_predito.png`
  - `residuos.png`
- **Mudanças em relação ao experimento anterior:**
  - Dataset: 'dataset_final.csv' → 'dataset_final_corrigido_sem_outlier.csv'
  - R² cv: 0.139 → 0.219
  - Notas: Baseline Ana — dataset_final.csv sem limpeza (1463 linhas, duplicatas). CV 5-fold. → Baseline Rafael/docs — dataset corrigido 680 linhas. CV 5-fold.

---

### 20260529_081748

- **Executado em:** 2026-05-29 08:18:10.143638
- **Git:** `backfill` @ `backfill`
- **Protocolo:** cv_5fold_temporal + treino_completo
- **Dataset:** `dataset_final_corrigido_sem_outlier.csv` (680 linhas)
- **Alvo:** `mstotal`
- **Features:** 0 colunas
- **Notas:** Mesmo dataset 680 linhas + variáveis de animais. CV 5-fold.
- **Métricas:**
  - **cv** (n=None): R²=0.219 ± 0.105, RMSE=2113
- **Arquivos:**
  - `importancia_features.png`
  - `metricas.txt`
  - `modelo_rf.joblib`
  - `real_vs_predito.png`
  - `residuos.png`
- **Mudanças em relação ao experimento anterior:**
  - Notas: Baseline Rafael/docs — dataset corrigido 680 linhas. CV 5-fold. → Mesmo dataset 680 linhas + variáveis de animais. CV 5-fold.

---

### 20260529_095329

- **Executado em:** 2026-05-29 09:53:33.634902
- **Git:** `backfill` @ `backfill`
- **Protocolo:** split_treino_validacao_teste
- **Dataset:** `dataset_final_corrigido_sem_outlier.csv` (680 linhas)
- **Split:** treino=470 | val=94 | teste=116
- **Alvo:** `mstotal`
- **Features:** 0 colunas
- **Notas:** Primeiro treino com split treino/val/teste (orientação do professor).
- **Métricas:**
  - **treino** (n=470): R²=0.845, RMSE=1043
  - **validacao** (n=94): R²=-0.095, RMSE=1749
  - **teste** (n=116): R²=0.108, RMSE=2275
- **Arquivos:**
  - `importancia_features.png`
  - `metricas.txt`
  - `modelo_rf.joblib`
  - `real_vs_predito.png`
  - `residuos.png`
- **Mudanças em relação ao experimento anterior:**
  - Protocolo de avaliação: 'cv_5fold_temporal + treino_completo' → 'split_treino_validacao_teste'
  - Fim do treino: None → '2018-08-20'
  - Fim da validação: None → '2019-01-14'
  - Árvores (n_estimators): '300_cv + 500_final' → 500
  - Notas: Mesmo dataset 680 linhas + variáveis de animais. CV 5-fold. → Primeiro treino com split treino/val/teste (orientação do professor).

---

### 20260606_100433

- **Executado em:** 2026-06-06 10:04:35.677715
- **Git:** `backfill` @ `backfill`
- **Protocolo:** split_treino_validacao_teste
- **Dataset:** `dataset_final_corrigido_sem_outlier.csv` (680 linhas)
- **Split:** treino=470 | val=94 | teste=116
- **Alvo:** `mstotal`
- **Features:** 0 colunas
- **Notas:** Repetição pós-merge docs→main; mesmo split e mesmo resultado.
- **Métricas:**
  - **treino** (n=470): R²=0.845, RMSE=1043
  - **validacao** (n=94): R²=-0.095, RMSE=1749
  - **teste** (n=116): R²=0.108, RMSE=2275
- **Arquivos:**
  - `importancia_features.png`
  - `metricas.txt`
  - `modelo_rf.joblib`
  - `real_vs_predito.png`
  - `residuos.png`
- **Mudanças em relação ao experimento anterior:**
  - Notas: Primeiro treino com split treino/val/teste (orientação do professor). → Repetição pós-merge docs→main; mesmo split e mesmo resultado.

## Como registrar o próximo treino

1. (Opcional) Edite `RUN_NOTAS` em `config.py` descrevendo o que mudou.
2. Execute `python 04_modelo_rf.py`.
3. Confira esta página e `experimentos.jsonl`.

## Regra de ouro

**Nunca apague** pastas em `resultados_rf/`. Novos treinos criam **nova** subpasta; os antigos permanecem.
