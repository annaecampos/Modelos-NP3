"""
Modelo Autoajustado com Keras Tuner (Random Search).

Uso:
    python3 "MODELO AUTOAJUSTADO COM KT.py" <N_EXECUCOES>
"""

import sys
import shutil
import time
from math import sqrt
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np
from numpy import concatenate
from scipy import stats

from pandas import read_csv, DataFrame

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot

import keras
import keras_tuner
from keras_tuner import RandomSearch

# ── Argumento de linha de comando ─────────────────────────────────────────────
try:
    valor_inteiro = int(sys.argv[1])
except (IndexError, ValueError):
    print("Uso: python3 script.py <N_EXECUCOES>")
    sys.exit(1)

# ── Configuração de caminhos ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# ── Loop de execuções ─────────────────────────────────────────────────────────
media1 = media4 = media5 = media6 = 0.0
now = datetime.now()
nomelogmedia = now

for aux in range(valor_inteiro):
    now = datetime.now()
    RUN_DIR = BASE_DIR / str(now)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log = open(BASE_DIR / "ManDados.txt", "a")
    log.write("------Manipulação dos dados iniciada em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    # ── Carga e pré-processamento ─────────────────────────────────────────────
    CSV_PATH = BASE_DIR / "e1_leo_2019.csv"
    dataset1 = read_csv(CSV_PATH, header=0, index_col=0, delimiter=";")
    values1 = dataset1.values

    values1 = values1[~np.isnan(values1).any(axis=1)]
    values1 = values1.astype("float32")

    scaler = MinMaxScaler()
    scaled1 = scaler.fit_transform(values1)
    scaled1 = DataFrame(scaled1)
    values1 = scaled1.values

    n_train = 36
    train1 = values1[:n_train, :]
    test1  = values1[n_train:, :]

    train_X1, train_y1 = train1[:, :-1], train1[:, -1]
    test_X1,  test_y1  = test1[:, :-1],  test1[:, -1]

    train_X1 = train_X1.reshape((train_X1.shape[0], 1, train_X1.shape[1]))
    test_X1  = test_X1.reshape((test_X1.shape[0],  1, test_X1.shape[1]))

    now = datetime.now()
    fim = time.time()
    t1 = fim - inicio
    log = open(BASE_DIR / "ManDados.txt", "a")
    log.write("------ Manipulação encerrada em: " + str(now) + "\n\nTempo: " + str(t1))
    log.close()
    shutil.move(str(BASE_DIR / "ManDados.txt"), str(RUN_DIR))

    # ── Keras Tuner – Random Search ───────────────────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "RS.txt", "a")
    log.write("------Random Search iniciada em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    n_features = train_X1.shape[2]

    def build_model(hp):
        model = keras.Sequential([
            keras.layers.LSTM(
                units=hp.Int("lstm_units", min_value=1, max_value=512, step=1),
                activation=hp.Choice("lstm_activation", ["relu", "tanh"]),
                return_sequences=True,
                input_shape=(1, n_features),
            ),
            keras.layers.LSTM(
                units=hp.Int("lstm_units_1", min_value=1, max_value=512, step=1),
                activation=hp.Choice("lstm_activation_1", ["relu", "tanh"]),
            ),
        ])
        if hp.Boolean("dropout"):
            model.add(keras.layers.Dropout(rate=0.25))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    kt_dir = RUN_DIR / "KT"
    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=3,
        directory=str(kt_dir),
        project_name="p20_infestado",
    )
    tuner.search_space_summary()

    train_X1 = train_X1.reshape(-1, 1, n_features)
    test_X1  = test_X1.reshape(-1, 1, n_features)
    tuner.search(train_X1, train_y1, epochs=5, validation_data=(test_X1, test_y1))

    best_model = tuner.get_best_models(num_models=1)[0]

    now = datetime.now()
    fim = time.time()
    t4 = fim - inicio
    log = open(BASE_DIR / "RS.txt", "a")
    log.write("------ Random Search encerrada em: " + str(now) + "\n\nTempo: " + str(t4))
    log.close()
    shutil.move(str(BASE_DIR / "RS.txt"), str(RUN_DIR))

    with open(str(RUN_DIR / "modelsummary.txt"), "w") as f:
        with redirect_stdout(f):
            best_model.summary()

    # ── Treinamento com melhores hiperparâmetros ──────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "TreinoMA.txt", "a")
    log.write("------Treinamento iniciado em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    model = best_model
    history = model.fit(train_X1, train_y1, epochs=5000,
                        validation_data=(test_X1, test_y1))

    now = datetime.now()
    fim = time.time()
    t5 = fim - inicio
    log = open(BASE_DIR / "TreinoMA.txt", "a")
    log.write("------ Treinamento encerrado em: " + str(now) + "\n\nTempo: " + str(t5))
    log.close()
    shutil.move(str(BASE_DIR / "TreinoMA.txt"), str(RUN_DIR))

    # ── Predição e métricas ───────────────────────────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "CalcCKT.txt", "a")
    log.write("------Cálculos iniciados em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    yhat1 = model.predict(test_X1)
    test_X1 = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))

    inv_yhat1 = concatenate((test_X1, yhat1), axis=1)
    inv_yhat1 = scaler.inverse_transform(inv_yhat1)
    inv_yhat1 = inv_yhat1[:, -1]

    test_y1 = test_y1.reshape((len(test_y1), 1))
    inv_y1 = concatenate((test_X1, test_y1), axis=1)
    inv_y1 = scaler.inverse_transform(inv_y1)
    inv_y1 = inv_y1[:, -1]

    rmse = sqrt(mean_squared_error(inv_y1, inv_yhat1))
    desvioAmostralpred1   = np.std(inv_yhat1)
    varianciaAmostralpred1 = inv_yhat1.var()
    desvioAmostralreal1   = np.std(inv_y1)
    varianciaAmostralreal1 = inv_y1.var()

    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y1, inv_yhat1)

    coeffs1 = np.polyfit(inv_y1, inv_yhat1, 5)
    p1_poly = np.poly1d(coeffs1)
    yhat1_poly = p1_poly(inv_y1)
    ybar1 = np.sum(inv_yhat1) / len(inv_yhat1)
    ssreg1 = np.sum((yhat1_poly - ybar1) ** 2)
    sstot1 = np.sum((inv_yhat1 - ybar1) ** 2)
    r1 = ssreg1 / sstot1

    now = datetime.now()
    fim = time.time()
    t6 = fim - inicio
    log = open(BASE_DIR / "CalcCKT.txt", "a")
    log.write("------ Cálculos encerrados em: " + str(now) + "\n\nTempo: " + str(t6))
    log.close()
    shutil.move(str(BASE_DIR / "CalcCKT.txt"), str(RUN_DIR))

    # ── Gráficos ──────────────────────────────────────────────────────────────
    erro = inv_yhat1 - inv_y1

    fig1, ax1 = pyplot.subplots()
    pyplot.boxplot([inv_y1, inv_yhat1, erro], labels=["Real", "Predito", "Erro"])
    pyplot.title("P20 - Infestado")
    dpi = fig1.get_dpi()
    pyplot.savefig(str(RUN_DIR / "GBoxP_CKT.png"), dpi=dpi * 2)
    pyplot.close()

    pyplot.scatter(inv_yhat1, inv_y1)
    plot_range = [min(inv_y1.min(), inv_yhat1.min()), max(inv_y1.max(), inv_yhat1.max())]
    pyplot.xlim(left=0, right=110)
    pyplot.ylim(bottom=0, top=110)
    pyplot.plot(plot_range, plot_range, "red")
    pyplot.title("P20 Infestado - Real x Predito")
    pyplot.ylabel("Real")
    pyplot.xlabel("Predito")
    pyplot.savefig(str(RUN_DIR / "GDisp_CKT.png"), dpi=dpi * 2)
    pyplot.close()

    with open(str(RUN_DIR / "Log_CKT.txt"), "w") as file:
        file.write("P20 - Infestado\n")
        file.write("Test RMSE: %.3f\n" % rmse)
        file.write("R2 linear: " + str(r_value ** 2) + "\n")
        file.write("R2 Polinomial: " + str(r1) + "\n")
        file.write("Desvio Real: " + str(desvioAmostralreal1) + "\n")
        file.write("Variancia Real: " + str(varianciaAmostralreal1) + "\n")
        file.write("Desvio Predito: " + str(desvioAmostralpred1) + "\n")
        file.write("Variancia Predito: " + str(varianciaAmostralpred1) + "\n")
        file.write("Real\n")
        for a in inv_y1:
            file.write("%.2f, " % a)
        file.write("\nPredito\n")
        for b in inv_yhat1:
            file.write("%.2f, " % b)

    media1 += t1
    media4 += t4
    media5 += t5
    media6 += t6

# ── Médias finais ─────────────────────────────────────────────────────────────
media1 /= valor_inteiro
media4 /= valor_inteiro
media5 /= valor_inteiro
media6 /= valor_inteiro

with open(str(BASE_DIR / f"log_medias_{nomelogmedia}.txt"), "a") as log:
    log.write(f"Média de manipulação de dados: {media1:.4f}s\n\n")
    log.write(f"Média de Random Search: {media4:.4f}s\n\n")
    log.write(f"Média de treinamento com KT: {media5:.4f}s\n\n")
    log.write(f"Média de cálculos com KT: {media6:.4f}s\n\n")
