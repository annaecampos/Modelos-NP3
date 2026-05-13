"""
Modelo Ajustado e Autoajustado com Keras Tuner usando camadas GRU.

Uso:
    python3 "MODELO AJUSTADO E AUTOAJUSTADO COM KT - GRU.py" <N_EXECUCOES>
"""

import sys
import shutil
import time
from math import sqrt
from pathlib import Path
from datetime import datetime

import numpy as np
from numpy import concatenate
from scipy import stats

from pandas import read_csv, DataFrame

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot

import keras
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.utils import plot_model
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
media1 = media2 = media3 = media4 = media5 = media6 = 0.0
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
    CSV_PATH = BASE_DIR / "e1.csv"
    dataset1 = read_csv(CSV_PATH, header=0, index_col=0, delimiter=",")
    values1 = dataset1.values

    values1 = values1[~np.isnan(values1).any(axis=1)]
    values1 = values1.astype("float32")

    scaler = MinMaxScaler()
    scaled1 = scaler.fit_transform(values1)
    scaled1 = DataFrame(scaled1)
    values1 = scaled1.values

    n_train = 38
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

    # ── Modelo GRU ajustado (sem KT) ─────────────────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "DefTreinoMO.txt", "a")
    log.write("------Modelo GRU Ajustado iniciado em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    model = Sequential([
        GRU(30, input_shape=(train_X1.shape[1], train_X1.shape[2]),
            kernel_initializer="normal", return_sequences=True),
        GRU(15, input_shape=(train_X1.shape[1], train_X1.shape[2]),
            kernel_initializer="normal"),
        Dense(1, kernel_initializer="normal"),
    ])
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    history = model.fit(
        train_X1, train_y1,
        epochs=5000, batch_size=72,
        validation_data=(test_X1, test_y1),
        verbose=0, shuffle=False,
    )

    now = datetime.now()
    fim = time.time()
    t2 = fim - inicio
    log = open(BASE_DIR / "DefTreinoMO.txt", "a")
    log.write("------ GRU Ajustado encerrado em: " + str(now) + "\n\nTempo: " + str(t2))
    log.close()
    shutil.move(str(BASE_DIR / "DefTreinoMO.txt"), str(RUN_DIR))

    # ── Predição e métricas (sem KT) ─────────────────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "CalcSKT.txt", "a")
    log.write("------Cálculos sem KT iniciados em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    yhat1 = model.predict(test_X1)
    test_X1_2d = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))

    inv_yhat1 = concatenate((test_X1_2d, yhat1), axis=1)
    inv_yhat1 = scaler.inverse_transform(inv_yhat1)
    inv_yhat1 = inv_yhat1[:, -1]

    test_y1_col = test_y1.reshape((len(test_y1), 1))
    inv_y1 = concatenate((test_X1_2d, test_y1_col), axis=1)
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
    t3 = fim - inicio
    log = open(BASE_DIR / "CalcSKT.txt", "a")
    log.write("------ Cálculos sem KT encerrados em: " + str(now) + "\n\nTempo: " + str(t3))
    log.close()
    shutil.move(str(BASE_DIR / "CalcSKT.txt"), str(RUN_DIR))

    plot_model(model, to_file=str(RUN_DIR / "model_plot_P20-Infestado.png"),
               show_shapes=True, show_layer_names=True)

    erro = inv_yhat1 - inv_y1
    fig1, ax1 = pyplot.subplots()
    pyplot.boxplot([inv_y1, inv_yhat1, erro], labels=["Real", "Predito", "Erro"])
    pyplot.title("P20 - Infestado")
    dpi = fig1.get_dpi()
    pyplot.savefig(str(RUN_DIR / "GBoxP_SKT.png"), dpi=dpi * 2)
    pyplot.close()

    pyplot.scatter(inv_yhat1, inv_y1)
    plot_range = [min(inv_y1.min(), inv_yhat1.min()), max(inv_y1.max(), inv_yhat1.max())]
    pyplot.xlim(left=0, right=8000)
    pyplot.ylim(bottom=0, top=8000)
    pyplot.plot(plot_range, plot_range, "red")
    pyplot.title("P20 Infestado - Real x Predito")
    pyplot.ylabel("Real")
    pyplot.xlabel("Predito")
    pyplot.savefig(str(RUN_DIR / "GDisp_SKT.png"), dpi=dpi * 2)
    pyplot.close()

    with open(str(RUN_DIR / "Log_SKT.txt"), "w") as file:
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

    # ── Keras Tuner – Random Search (GRU) ────────────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "RS.txt", "a")
    log.write("------Random Search GRU iniciada em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    n_features = train_X1.shape[2]

    def build_model(hp):
        gru_model = keras.Sequential([
            keras.layers.GRU(
                units=hp.Int("gru_units", min_value=30, max_value=512, step=32),
                activation=hp.Choice("gru_activation", ["relu", "tanh"]),
                return_sequences=True,
                input_shape=(1, n_features),
            ),
            keras.layers.GRU(
                units=hp.Int("gru_units_1", min_value=15, max_value=512, step=32),
                activation=hp.Choice("gru_activation_1", ["relu", "tanh"]),
            ),
        ])
        if hp.Boolean("dropout"):
            gru_model.add(keras.layers.Dropout(rate=0.25))
        gru_model.add(keras.layers.Dense(1, activation="sigmoid"))

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        gru_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return gru_model

    kt_dir = RUN_DIR / "KT"
    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=5,
        executions_per_trial=3,
        directory=str(kt_dir),
        project_name="p20_infestado",
    )
    tuner.search_space_summary()

    train_X1_kt = train_X1.reshape(-1, 1, n_features)
    test_X1_kt  = test_X1.reshape(-1, 1, n_features)
    tuner.search(train_X1_kt, train_y1, epochs=5, validation_data=(test_X1_kt, test_y1))

    best_model = tuner.get_best_models(num_models=1)[0]

    now = datetime.now()
    fim = time.time()
    t4 = fim - inicio
    log = open(BASE_DIR / "RS.txt", "a")
    log.write("------ Random Search encerrada em: " + str(now) + "\n\nTempo: " + str(t4))
    log.close()
    shutil.move(str(BASE_DIR / "RS.txt"), str(RUN_DIR))

    # ── Treinamento final com KT ──────────────────────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "TreinoMA.txt", "a")
    log.write("------Treinamento GRU+KT iniciado em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    model_kt = best_model
    history = model_kt.fit(train_X1_kt, train_y1, epochs=5000,
                           validation_data=(test_X1_kt, test_y1))

    now = datetime.now()
    fim = time.time()
    t5 = fim - inicio
    log = open(BASE_DIR / "TreinoMA.txt", "a")
    log.write("------ Treinamento encerrado em: " + str(now) + "\n\nTempo: " + str(t5))
    log.close()
    shutil.move(str(BASE_DIR / "TreinoMA.txt"), str(RUN_DIR))

    # ── Predição e métricas (com KT) ─────────────────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "CalcCKT.txt", "a")
    log.write("------Cálculos com KT iniciados em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    yhat1_kt = model_kt.predict(test_X1_kt)
    test_X1_kt_2d = test_X1_kt.reshape((test_X1_kt.shape[0], test_X1_kt.shape[2]))

    inv_yhat1_kt = concatenate((test_X1_kt_2d, yhat1_kt), axis=1)
    inv_yhat1_kt = scaler.inverse_transform(inv_yhat1_kt)
    inv_yhat1_kt = inv_yhat1_kt[:, -1]

    test_y1_col = test_y1.reshape((len(test_y1), 1))
    inv_y1_kt = concatenate((test_X1_kt_2d, test_y1_col), axis=1)
    inv_y1_kt = scaler.inverse_transform(inv_y1_kt)
    inv_y1_kt = inv_y1_kt[:, -1]

    rmse_kt = sqrt(mean_squared_error(inv_y1_kt, inv_yhat1_kt))
    desvioAmostralpred_kt   = np.std(inv_yhat1_kt)
    varianciaAmostralpred_kt = inv_yhat1_kt.var()
    desvioAmostralreal_kt   = np.std(inv_y1_kt)
    varianciaAmostralreal_kt = inv_y1_kt.var()

    slope_kt, intercept_kt, r_value_kt, p_value_kt, std_err_kt = stats.linregress(inv_y1_kt, inv_yhat1_kt)
    coeffs_kt = np.polyfit(inv_y1_kt, inv_yhat1_kt, 5)
    p_poly_kt = np.poly1d(coeffs_kt)
    yhat_poly_kt = p_poly_kt(inv_y1_kt)
    ybar_kt = np.sum(inv_yhat1_kt) / len(inv_yhat1_kt)
    ssreg_kt = np.sum((yhat_poly_kt - ybar_kt) ** 2)
    sstot_kt = np.sum((inv_yhat1_kt - ybar_kt) ** 2)
    r1_kt = ssreg_kt / sstot_kt

    now = datetime.now()
    fim = time.time()
    t6 = fim - inicio
    log = open(BASE_DIR / "CalcCKT.txt", "a")
    log.write("------ Cálculos com KT encerrados em: " + str(now) + "\n\nTempo: " + str(t6))
    log.close()
    shutil.move(str(BASE_DIR / "CalcCKT.txt"), str(RUN_DIR))

    erro_kt = inv_yhat1_kt - inv_y1_kt
    fig1, ax1 = pyplot.subplots()
    pyplot.boxplot([inv_y1_kt, inv_yhat1_kt, erro_kt], labels=["Real", "Predito", "Erro"])
    pyplot.title("P20 - Infestado (KT)")
    dpi = fig1.get_dpi()
    pyplot.savefig(str(RUN_DIR / "GBoxP_CKT.png"), dpi=dpi * 2)
    pyplot.close()

    pyplot.scatter(inv_yhat1_kt, inv_y1_kt)
    plot_range_kt = [min(inv_y1_kt.min(), inv_yhat1_kt.min()),
                     max(inv_y1_kt.max(), inv_yhat1_kt.max())]
    pyplot.xlim(left=0, right=8000)
    pyplot.ylim(bottom=0, top=8000)
    pyplot.plot(plot_range_kt, plot_range_kt, "red")
    pyplot.title("P20 Infestado - Real x Predito (KT)")
    pyplot.ylabel("Real")
    pyplot.xlabel("Predito")
    pyplot.savefig(str(RUN_DIR / "GDisp_CKT.png"), dpi=dpi * 2)
    pyplot.close()

    with open(str(RUN_DIR / "Log_CKT.txt"), "w") as file:
        file.write("P20 - Infestado (GRU + KT)\n")
        file.write("Test RMSE: %.3f\n" % rmse_kt)
        file.write("R2 linear: " + str(r_value_kt ** 2) + "\n")
        file.write("R2 Polinomial: " + str(r1_kt) + "\n")
        file.write("Desvio Real: " + str(desvioAmostralreal_kt) + "\n")
        file.write("Variancia Real: " + str(varianciaAmostralreal_kt) + "\n")
        file.write("Desvio Predito: " + str(desvioAmostralpred_kt) + "\n")
        file.write("Variancia Predito: " + str(varianciaAmostralpred_kt) + "\n")
        file.write("Real\n")
        for a in inv_y1_kt:
            file.write("%.2f, " % a)
        file.write("\nPredito\n")
        for b in inv_yhat1_kt:
            file.write("%.2f, " % b)

    media1 += t1
    media2 += t2
    media3 += t3
    media4 += t4
    media5 += t5
    media6 += t6

# ── Médias finais ─────────────────────────────────────────────────────────────
n = valor_inteiro
with open(str(BASE_DIR / f"log_medias_{nomelogmedia}.txt"), "a") as log:
    log.write(f"Média de manipulação de dados: {media1/n:.4f}s\n\n")
    log.write(f"Média de definição + treinamento GRU sem KT: {media2/n:.4f}s\n\n")
    log.write(f"Média de cálculos sem KT: {media3/n:.4f}s\n\n")
    log.write(f"Média de Random Search: {media4/n:.4f}s\n\n")
    log.write(f"Média de treinamento com KT: {media5/n:.4f}s\n\n")
    log.write(f"Média de cálculos com KT: {media6/n:.4f}s\n\n")
