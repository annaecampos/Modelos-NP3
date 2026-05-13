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
from keras.layers import Dense, LSTM
from keras.utils import plot_model

# ── Configuração de caminhos ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

now = datetime.now()
RUN_DIR = BASE_DIR / str(now)
RUN_DIR.mkdir(parents=True, exist_ok=True)

log = open(BASE_DIR / "ManDados.txt", "a")
log.write("------Manipulação dos dados iniciada em: " + str(now) + "\n\n")
log.close()
inicio = time.time()

# ── Carga e pré-processamento ─────────────────────────────────────────────────
CSV_PATH = BASE_DIR / "e1-leonardo.csv"
dataset1 = read_csv(CSV_PATH, header=0, index_col=0, delimiter=";")
values1 = dataset1.values

values1 = values1[~np.isnan(values1).any(axis=1)]
values1 = values1.astype("float32")

real1 = values1[:, -1]

scaler = MinMaxScaler()
scaled1 = scaler.fit_transform(values1)
scaled1 = DataFrame(scaled1)

values1 = scaled1.values

n_train = 26
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
log.write("------ Manipulação dos dados encerrada em: " + str(now) + "\n\nTempo de execução: " + str(t1))
log.close()
shutil.move(str(BASE_DIR / "ManDados.txt"), str(RUN_DIR))

# ── Definição e treinamento ───────────────────────────────────────────────────
now = datetime.now()
log = open(BASE_DIR / "DefTreinoMO.txt", "a")
log.write("------ModeloOriginal Execução iniciada em: " + str(now) + "\n\n")
log.close()
inicio = time.time()

model = Sequential([
    LSTM(44, input_shape=(train_X1.shape[1], train_X1.shape[2]),
         kernel_initializer="normal", return_sequences=True),
    LSTM(22, input_shape=(train_X1.shape[1], train_X1.shape[2]),
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
log.write("------ Execução Total encerrada em: " + str(now) + "\n\nTempo de execução: " + str(t2))
log.close()
shutil.move(str(BASE_DIR / "DefTreinoMO.txt"), str(RUN_DIR))

# ── Predição e métricas ───────────────────────────────────────────────────────
now = datetime.now()
log = open(BASE_DIR / "CalcSKT.txt", "a")
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
p1 = np.poly1d(coeffs1)
yhat1_poly = p1(inv_y1)
ybar1 = np.sum(inv_yhat1) / len(inv_yhat1)
ssreg1 = np.sum((yhat1_poly - ybar1) ** 2)
sstot1 = np.sum((inv_yhat1 - ybar1) ** 2)
r1 = ssreg1 / sstot1

now = datetime.now()
fim = time.time()
t3 = fim - inicio
log = open(BASE_DIR / "CalcSKT.txt", "a")
log.write("------ Cálculos encerrados em: " + str(now) + "\n\nTempo de execução: " + str(t3))
log.close()
shutil.move(str(BASE_DIR / "CalcSKT.txt"), str(RUN_DIR))

# ── Visualização da arquitetura ───────────────────────────────────────────────
plot_model(model, to_file=str(RUN_DIR / "model_plot_P20-Infestado.png"),
           show_shapes=True, show_layer_names=True)

# ── Gráficos ──────────────────────────────────────────────────────────────────
erro = inv_yhat1 - inv_y1

fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y1, inv_yhat1, erro], labels=["Real", "Predito", "Erro"])
pyplot.title("P20 - Infestado")
dpi = fig1.get_dpi()
pyplot.savefig(str(RUN_DIR / "GBoxP_SKT.png"), dpi=dpi * 2)
pyplot.close()

pyplot.scatter(inv_yhat1, inv_y1)
plot_range = [min(inv_y1.min(), inv_yhat1.min()), max(inv_y1.max(), inv_yhat1.max())]
pyplot.xlim(left=0, right=110)
pyplot.ylim(bottom=0, top=110)
pyplot.plot(plot_range, plot_range, "red")
pyplot.title("P20 Infestado - Real x Predito")
pyplot.ylabel("Real")
pyplot.xlabel("Predito")
pyplot.savefig(str(RUN_DIR / "GDisp_SKT.png"), dpi=dpi * 2)
pyplot.close()

# ── Log de resultados ─────────────────────────────────────────────────────────
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
