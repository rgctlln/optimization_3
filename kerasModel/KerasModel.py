import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import psutil
import numpy as np
from sklearn.metrics import mean_squared_error


class KerasLinearModel:
    def __init__(self, input_dim):
        self.model = keras.Sequential([
            layers.Dense(1, input_shape=(input_dim,))
        ])

    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer, loss='mse')

    def train(self, x_train, y_train, epochs=50, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def evaluate(self, x_test, y_test):
        pred = self.model.predict(x_test, verbose=0).ravel()
        mse = mean_squared_error(y_test, pred)
        return mse, pred

    def train_keras(opt_name, xtr, ytr, xte, yte, lr_dict):
        # Инициализация оптимизатора
        lr = lr_dict[opt_name]
        if opt_name == "SGD":
            optimizer = keras.optimizers.SGD(learning_rate=lr)
        elif opt_name == "SGD+Momentum":
            optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif opt_name == "SGD+Nesterov":
            optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        elif opt_name == "Adagrad":
            optimizer = keras.optimizers.Adagrad(learning_rate=lr)
        elif opt_name == "RMSprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        elif opt_name == "Adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # Обучение
        model = KerasLinearModel(input_dim=xtr.shape[1])
        model.compile(optimizer)

        start_time = time.time()
        process = psutil.Process()

        model.train(xtr, ytr)

        duration = time.time() - start_time
        memory = process.memory_info().rss / 1024 ** 2

        mse, pred = model.evaluate(xte, yte)

        print(f"[{opt_name}] True y (scaled): {yte.numpy().squeeze()[:3]}")
        print(f"[{opt_name}] Pred y (scaled): {pred[:3]}")

        return {
            "optimizer": opt_name,
            "framework": "TensorFlow/Keras",
            "mse": mse,
            "time": duration,
            "memory": memory,
        }
