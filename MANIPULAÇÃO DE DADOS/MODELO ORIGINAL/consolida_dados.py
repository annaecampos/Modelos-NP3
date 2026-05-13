"""
Consolida os dados do banco PostgreSQL e gera arquivo CSV para o Modelo Original.

Uso:
    python3 consolida_dados.py

Requisito: banco 'db_2019' acessível em localhost:5432 com usuário postgres.
"""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg2
from scipy.stats.mstats import mquantiles

# ── Configuração ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "db_2019",
    "user": "postgres",
    "password": "12345",
}

# ── Inicialização do log ──────────────────────────────────────────────────────
now = datetime.now()
log = open(BASE_DIR / "log.txt", "a")
log.write("------ Consolida dados ModeloOriginal Execução iniciada em: " + str(now) + "\n\n")
log.close()

# ── Conexão com o banco ───────────────────────────────────────────────────────
try:
    con = psycopg2.connect(**DB_CONFIG)
    cur = con.cursor()
except Exception as e:
    with open(BASE_DIR / "log.txt", "a") as log:
        log.write("\nFalha na conexão com o BD: " + str(e) + "\n")
    raise SystemExit(1)

# ── Consultas SQL ─────────────────────────────────────────────────────────────
def executar_sql(cursor, conexao, caminho_sql, params=None):
    sql = " ".join(open(caminho_sql).readlines())
    if params:
        cursor.execute(sql, params)
    else:
        cursor.execute(sql)
    conexao.commit()
    return cursor.fetchall()

tdo    = executar_sql(cur, con, BASE_DIR / "selectent.sql")
dados1 = executar_sql(cur, con, BASE_DIR / "selectmstotalD.sql")
dados2 = executar_sql(cur, con, BASE_DIR / "selectmstotalF.sql")

mstotalD = [float(d[0]) for d in dados1]
quartilD = mquantiles(mstotalD, prob=[0.75])

mstotalF = [float(d[0]) for d in dados2]
quartilF = mquantiles(mstotalF, prob=[0.75])

print("Quartil D:", quartilD)
print("Quartil F:", quartilF)

# ── Construção das entradas ───────────────────────────────────────────────────
ent = []
ant = None

for x in tdo:
    if ant is None:
        ant = x
        continue

    if (ant[1] != x[1]) and x[0] == "D":
        clima = executar_sql(
            cur, con,
            BASE_DIR / "selectclima.sql",
            params=(ant[1].isoformat(), x[1].isoformat()),
        )

        mstotal_futuro = float(quartilD[0]) if x[3] > quartilD else x[3]
        mstotal_ant    = float(quartilF[0]) if ant[3] > quartilF else ant[3]

        linha = [
            x[1],
            (x[1] - ant[1]).days,
            ant[2],
            mstotal_ant,
            clima[0][0],  clima[0][7],  clima[0][14],
            clima[0][1],  clima[0][8],  clima[0][15],
            clima[0][2],  clima[0][9],  clima[0][16],
            clima[0][3],  clima[0][10], clima[0][17],
            clima[0][4],  clima[0][11], clima[0][18],
            clima[0][5],  clima[0][12], clima[0][19],
            clima[0][6],  clima[0][13], clima[0][20],
            clima[0][21], clima[0][22], clima[0][23],
            clima[0][24], clima[0][25], clima[0][26],
            clima[0][27], clima[0][28], clima[0][29],
            "%.2f" % float(quartilD[0]),
        ]
        ent.append(linha)

    ant = x

# ── Exportação CSV ────────────────────────────────────────────────────────────
header = [
    "data", "numerodias", "alturamediaanterior", "mstotalanterior",
    "tmin", "dptmin", "vartmin",
    "tmed", "dptmed", "vartmed",
    "tmax", "dptmax", "vartmax",
    "umidade", "dpumidade", "varumidade",
    "velocidadevento", "dpvelocidadevento", "varvelovidadevento",
    "radiacaosolar", "dpradiacaosolar", "varradiacaosolar",
    "chuva", "dpchuva", "varchuva",
    "somatermica", "dpsomatermica", "varsomatermica",
    "def", "dpdef", "vardef",
    "exc", "dpexc", "varexc",
    "taxaacumulo",
]

saida = BASE_DIR / "e1_Treino.csv"
with open(saida, "w", newline="") as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=";")
    wr.writerow(header)
    wr.writerows(ent)

now = datetime.now()
with open(BASE_DIR / "log.txt", "a") as log:
    log.write("------Consolida dados ModeloOriginal Execução terminada em: " + str(now) + "\n\n")

print(f"CSV gerado: {saida}")
