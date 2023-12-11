import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import time


class Perceptron(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, is_sigmoid=False):
        nn.Module.__init__(self)
        self.train_time = 0
        self.lossFn = nn.MSELoss()
        self.linear = nn.Linear(3, 2)
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=0.01)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет и позволяет запускать их одновременно
        if is_sigmoid:
            self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),  # слой линейных сумматоров
                                        nn.Sigmoid(),  # функция активации
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.Sigmoid(),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.Sigmoid(),
                                        nn.Linear(hidden_size, out_size),
                                        nn.Sigmoid(),
                                        )
        else:
            self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),  # слой линейных сумматоров
                                        nn.ReLU(),  # функция активации
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, out_size),
                                        nn.ReLU(),
                                        )

    # прямой проход
    def forward(self, x):
        return self.layers(x)

    def train2(self, x, y, num_iter):
        loss = None
        start_time = time()

        for i in range(0, num_iter):
            # Делаем предсказание
            pred = self.forward(x)

            # Вычисляем ошибку
            loss = self.lossFn(pred, y)

            # Обратное распределение
            loss.backward()

            # Шаг оптимизации
            self.optimizer.step()

            if i % 100 == 0:
                print('Ошибка на ' + str(i + 1) + ' итерации: ', loss.item())
        self.train_time = time() - start_time
        return loss.item()


df = pd.read_csv('../data.csv')
df = df.iloc[np.random.permutation(len(df))]

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
y = LabelEncoder().fit_transform(y)
X = df.iloc[0:100, 0:2].values
Y = np.eye(2)[y]

# Деление данных на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

inputSize = X.shape[1]  # количество входных сигналов равно количеству признаков задачи
hiddenSizes = 50  # задаем число нейронов скрытого слоя
outputSize = Y.shape[1] if len(Y.shape) else 1  # количество выходных сигналов равно количеству классов задачи

net_relu = Perceptron(inputSize, hiddenSizes, outputSize)
net_sigmoid = Perceptron(inputSize, hiddenSizes, outputSize, is_sigmoid=True)
num_iter = 5000

net_relu.train2(torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(Y.astype(np.float32)), num_iter)
net_sigmoid.train2(torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(Y.astype(np.float32)), num_iter)

print(f"Время обуяения с использванием функции активации ReLU: {net_relu.train_time}")
print(f"Время обуяения с использванием функции активации Sigmoid: {net_sigmoid.train_time}")

pred = net_relu.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
a = pred > 0.5
b = a - Y_test
err = sum(abs((pred > 0.5) - Y_test))
print(err)
