# Treinos experimentais (separados do pipeline principal)

Os scripts `01`–`04` na raiz do `SENTINEL-PIPELINE` **não são alterados**.
Cada estratégia de treino vive em **sua própria pasta**, com dataset, scripts e resultados.

## Estrutura

```
treinos/
├── README.md                 ← este arquivo
├── REGISTRO_GERAL.md         ← índice de todos os tipos de treino
├── _lib/                     ← código compartilhado (não executar direto)
├── split_680_limpo/          ← baseline: 680 linhas, split temporal
│   ├── config_treino.py
│   ├── treinar.py
│   ├── dados/                ← cópia ou link do dataset usado
│   └── resultados/           ← uma subpasta por execução
└── alinhado_10d/             ← limpeza + ≤10 dias campo↔satélite
    ├── config_treino.py
    ├── monta_dataset.py
    ├── treinar.py
    ├── dados/
    └── resultados/
```

## Como rodar

```bash
cd SENTINEL-PIPELINE/treinos/alinhado_10d
python monta_dataset.py    # só onde existir
python treinar.py
```

## Regra

- **Nunca apague** pastas em `resultados/` — cada subpasta é um experimento.
- Antes de treinar, edite `RUN_NOTAS` em `config_treino.py`.
- Pipeline principal (`01`→`04`) continua independente em `resultados_rf/`.
