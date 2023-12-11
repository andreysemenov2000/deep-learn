import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Загрузка и подготовка данных
df = pd.read_csv('../data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = LabelEncoder().fit_transform(y)
X = df.iloc[0:100, 0:3].values
y_encoded = np.eye(3)[y]

# Деление данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


class NeuralNetwork:
    def __init__(self, input_shape, hidden_shape, output_shape):
        # Инициализация весов синапсов скрытого слоя
        self.W_hidden = np.random.uniform(size=(input_shape, hidden_shape))

        # Инициализируем мапу, куда будем сохранять веса
        self.w_matrix_map = {}

        # Инициализация весов синапсов выходного слоя
        self.W_output = np.random.uniform(size=(hidden_shape, output_shape))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, X):
        # Прямой проход через скрытый слой
        hidden_layer_output = self.sigmoid(np.dot(X, self.W_hidden))

        # Прямой проход через выходной слой
        output_layer_output = self.sigmoid(np.dot(hidden_layer_output, self.W_output))

        return output_layer_output, hidden_layer_output

    def backward_pass(self, X, y, output, hidden_output, learning_rate):
        # Обратное распространение ошибки на выходном слое
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        # Обратное распространение ошибки на скрытом слое
        hidden_error = np.dot(output_delta, self.W_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

        # Корректировка весов синапсов выходного и скрытого слоя
        self.W_output += learning_rate * np.dot(hidden_output.T, output_delta)

        self.W_hidden += learning_rate * np.dot(X.T, hidden_delta)

        self.save_weight_matrix()

    def train(self, X, y, learning_rate):
        while not self.check_is_stop_train(X, y):
            randIndexes =np.random.permutation(len(y))
            X = X[randIndexes]
            y = y[randIndexes]
            # Стохастический градиентный спуск
            for j in range(len(X)):
                X_sample = X[j].reshape(1, -1)
                y_sample = y[j].reshape(1, -1)

                output, hidden_output = self.forward_pass(X_sample)
                self.backward_pass(X_sample, y_sample, output, hidden_output, learning_rate)

    def predict(self, X):
        output, _ = self.forward_pass(X)
        return output

    def check_is_stop_train(self, X, Y):
        if self.check_is_accuracy_good(X, Y):
            return True
        if self.check_is_weights_repeated():
            return True
        return False

    def check_is_accuracy_good(self, X, Y):
        # Предсказание
        predictions = neural_network.predict(X)

        # Округление активационных значений до 0 или 1 для классификации
        predictions_rounded = np.round(predictions)

        # Вычисление точности
        accuracy = np.mean(predictions_rounded == Y)

        print(accuracy)
        return accuracy >= 0.95

    def save_weight_matrix(self):
        weights_list = list(self.W_output.reshape(-1)) + list(self.W_hidden.reshape(-1))
        weight_matrix_key = ''.join(str(w) for w in weights_list)
        keys = self.w_matrix_map.keys()
        r = list(filter(lambda k: k == weight_matrix_key, keys))
        if len(r) != 0:
            self.w_matrix_map[weight_matrix_key] += 1
        else:
            self.w_matrix_map[weight_matrix_key] = 1

    def check_is_weights_repeated(self):
        values = list(self.w_matrix_map.values())
        if any([v > 2 for v in values]):
            return True
        return False


# Создание и обучение нейронной сети
input_shape = X_train.shape[1]
output_shape = y_train.shape[1]

neural_network = NeuralNetwork(input_shape=input_shape, hidden_shape=10, output_shape=output_shape)
neural_network.train(X_train, y_train, learning_rate=0.01)

# Предсказание
test_predictions = neural_network.predict(X_test)

# Округление активационных значений до 0 или 1 для классификации
test_predictions_rounded = np.round(test_predictions)

# Вычисление точности на обучающей и тестовой выборке
test_accuracy = np.mean(test_predictions_rounded == y_test)

print('Точность на тестовой выборке', test_accuracy)
