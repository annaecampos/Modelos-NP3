"""
Modelo Autoajustado com Árvores de Decisão (AD).

Uso:
    python3 "MODELO AUTOAJUSTADO COM AD.py" <N_EXECUCOES>

Onde <N_EXECUCOES> é o número de rodadas de busca e treinamento.
"""

import sys
import shutil
import time
import random
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

# ── Argumento de linha de comando ─────────────────────────────────────────────
try:
    valor_inteiro = int(sys.argv[1])
except (IndexError, ValueError):
    print("Uso: python3 script.py <N_EXECUCOES>")
    sys.exit(1)

# ── Configuração de caminhos ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# ── Árvore de Decisão para classificação de hiperparâmetros ──────────────────
def AD_classificar_ajuste(dropout, learning_rate, layer1, layer2, activation):
    if dropout == "True":
        if learning_rate <= 0.008:
            if learning_rate <= 0.001:
                return "Ajuste Adequado"
            else:
                if learning_rate <= 0.006:
                    return "Ajuste Inadequado"
                else:
                    if activation == "relu":
                        if layer2 <= 89:
                            return "Ajuste Adequado"
                        else:
                            if learning_rate <= 0.007:
                                return "Ajuste Adequado"
                            else:
                                return "Ajuste Inadequado"
                    else:
                        return "Ajuste Inadequado"
        else:
            return "Ajuste Adequado"
    else:
        if layer2 <= 185:
            if learning_rate <= 0.002:
                return "Ajuste Adequado"
            else:
                if activation == "relu":
                    if learning_rate <= 0.007:
                        return "Ajuste Adequado"
                    else:
                        return "Ajuste Inadequado"
                else:
                    return "Ajuste Adequado"
        else:
            if learning_rate <= 0.005:
                return "Ajuste Adequado"
            else:
                if activation == "relu":
                    return "Ajuste Adequado"
                else:
                    return "Ajuste Adequado"

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

    # ── Busca de hiperparâmetros via AD ───────────────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "Sintonização.txt", "a")
    log.write("------Sintonização AD iniciada em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    qualidade = "Ajuste Inadequado"
    while qualidade == "Ajuste Inadequado":
        learning_rate = random.uniform(0.0, 0.01)
        dropout       = random.choice(["True", "False"])
        layer1        = random.randint(1, 512)
        layer2        = random.randint(1, 512)
        activation    = random.choice(["relu", "tanh"])
        qualidade = AD_classificar_ajuste(dropout, learning_rate, layer1, layer2, activation)
        print("\n______________________________________")
        print(f"Layer 1: {layer1}  Layer 2: {layer2}")
        print(f"Dropout: {dropout}  Activation: {activation}")
        print(f"Learning Rate: {learning_rate:.6f}")
        print(qualidade)

    with open(str(RUN_DIR / "hiperparâmetros.txt"), "w") as arquivo:
        arquivo.write(f"Layer 1: {layer1}\n")
        arquivo.write(f"Layer 2: {layer2}\n")
        arquivo.write(f"Dropout: {dropout}\n")
        arquivo.write(f"Activation: {activation}\n")
        arquivo.write(f"Learning Rate: {learning_rate}\n")

    now = datetime.now()
    fim = time.time()
    t4 = fim - inicio
    log = open(BASE_DIR / "Sintonização.txt", "a")
    log.write("------ Sintonização encerrada em: " + str(now) + "\n\nTempo: " + str(t4))
    log.close()
    shutil.move(str(BASE_DIR / "Sintonização.txt"), str(RUN_DIR))

    # ── Construção e treinamento do modelo ────────────────────────────────────
    now = datetime.now()
    log = open(BASE_DIR / "TreinoMA.txt", "a")
    log.write("------Treinamento iniciado em: " + str(now) + "\n\n")
    log.close()
    inicio = time.time()

    n_features = train_X1.shape[2]
    layers_list = [
        keras.layers.LSTM(
            units=layer1, activation=activation,
            return_sequences=True, input_shape=(1, n_features),
        ),
        keras.layers.LSTM(units=layer2, activation=activation),
    ]
    if dropout == "True":
        layers_list.append(keras.layers.Dropout(rate=0.25))
    layers_list.append(keras.layers.Dense(1, activation="sigmoid"))

    model = keras.Sequential(layers_list)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
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
    log.write(f"Média de sintonização AD: {media4:.4f}s\n\n")
    log.write(f"Média de treinamento: {media5:.4f}s\n\n")
    log.write(f"Média de cálculos: {media6:.4f}s\n\n")
