import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Perceptron(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.lossFn = nn.MSELoss()
        self.linear = nn.Linear(3, 2)
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=0.01)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет и позволяет запускать их одновременно
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
        return loss.item()


df = pd.read_csv('../data.csv')
df = df.iloc[np.random.permutation(len(df))]

# X = df.iloc[0:100, 0:3].values
# y = df.iloc[0:100, 4]
# y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor': 3}).values.reshape(-1, 1)
# Y = np.zeros((y.shape[0], np.unique(y).shape[0]))
# for i in np.unique(y):
#     Y[:, i - 1] = np.where(y == i, 1, 0).reshape(1, -1)
#
# X_test = df.iloc[100:150, 0:3].values
# y = df.iloc[100:150, 4]
# y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor': 3}).values.reshape(-1, 1)
# Y_test = np.zeros((y.shape[0], np.unique(y).shape[0]))
# for i in np.unique(y):
#     Y_test[:, i - 1] = np.where(y == i, 1, 0).reshape(1, -1)

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = LabelEncoder().fit_transform(y)
X = df.iloc[0:100, 0:3].values
Y = np.eye(3)[y]

# Деление данных на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

inputSize = X.shape[1]  # количество входных сигналов равно количеству признаков задачи
hiddenSizes = 50  # задаем число нейронов скрытого слоя
outputSize = Y.shape[1] if len(Y.shape) else 1  # количество выходных сигналов равно количеству классов задачи

net = Perceptron(inputSize, hiddenSizes, outputSize)

net.train2(torch.from_numpy(X.astype(np.float32)),
           torch.from_numpy(Y.astype(np.float32)), 5000)

pred = net.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
a = pred > 0.5
b = a - Y_test
err = sum(abs((pred > 0.5) - Y_test))
print(err)
