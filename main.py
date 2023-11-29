import pandas as pd
import numpy as np

from lab_2.lab_2 import NN

error = None


def __check_is_stop_train(perceptron):
    if error == 0:
        return True
    else:
        values = list(perceptron.w_matrix_map.values())
        if any([v > 2 for v in values]):
            return True
        else:
            return False


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values

inputSize = X.shape[1]  # количество входных сигналов равно количеству признаков задачи
hiddenSizes = 10  # задаем число нейронов скрытого (А) слоя
outputSize = 1 if len(y.shape) else y.shape[1]  # количество выходных сигналов равно количеству классов задачи

nn = NN(X, y)

for i in range(1):
    nn.train()
