"""
    @author:tb
    @time:2024-5-12
    LSTM模型构建模块
"""
import os
import csv
import math
import numpy as np
import datetime as dt
from keras.api.layers import Dense, LSTM, Dropout
from keras.api.models import load_model, Sequential
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class Model:
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print(f"[model] Loading model from file {filepath}")
        self.model = load_model(filepath)

    def build_model(self, configs):
        for layer in configs["model"]["layers"]:
            neurons = layer["neurons"] if "neurons" in layer else None
            dropout_rate = layer["rate"] if "rate" in layer else None
            activation = layer["activation"] if "activation" in layer else None
            return_seq = layer["return_seq"] if "return_seq" in layer else None
            input_time_steps = layer["input_time_steps"] if "input_time_steps" in layer else None
            input_dim = layer["input_dim"] if "input_dim" in layer else None
            if layer["type"] == "dense":
                self.model.add(Dense(neurons, activation=activation))
            if layer["type"] == "lstm":
                self.model.add(LSTM(neurons, input_shape=(input_time_steps, input_dim), return_sequences=return_seq))
            if layer["type"] == "dropout":
                self.model.add(Dropout(dropout_rate))
            # TODO 添加其他模型的解析

        self.model.compile(loss=configs["model"]["loss"], optimizer=configs["model"]["optimizer"])

        return self.model

    def train(self, x, y, test_x, test_y, epochs, batch_size, save_dir, optimizer):
        print("[Model] Train starting --- ")
        print(f"[Model] {epochs} epochs, {batch_size} batch_size")

        save_file_name = os.path.join(save_dir,
                                      f"{dt.datetime.now().strftime('%Y%m%d-%H%M')}-{optimizer}-e{epochs}.keras")
        callbacks = [
            EarlyStopping(monitor="loss", patience=10),
            ModelCheckpoint(filepath=save_file_name, monitor="loss", save_best_only=True)
        ]
        history = self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            validation_data=(test_x, test_y)
        )
        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        self.model.save(save_file_name)
        print(f"[model] Training Completed,Model saved as {save_file_name}")
        return train_loss, val_loss

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epochs, save_dir):
        print("[Model] Training started")
        print(f"[Model] {epochs} epochs, {batch_size} batch_size, {steps_per_epochs} batches per epoch")
        save_file_name = os.path.join(save_dir, f"{dt.datetime.now().strftime('%d%m%Y-%H%M%S')}-e{epochs}.h5")
        callbacks = [
            ModelCheckpoint(filepath=save_file_name, monitor="loss", save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epochs=steps_per_epochs,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )
        print(f"[model] Training Completed,Model saved as {save_file_name}")

    def predict_point_by_point(self, data):
        print("[Model] Predicting Point-by-Point...")
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len, debug=False):
        if not debug:
            print("[Model] Predicting Sequences Multiple...")
            prediction_seqs = []
            for i in range(int(len(data) / prediction_len)):
                cur_frame = data[i * prediction_len]
                predicted = []
                for j in range(prediction_len):
                    predicted.append(self.model.predict(cur_frame[np.newaxis, :, :])[0, 0])
                    cur_frame = cur_frame[1:]
                    cur_frame = np.insert(cur_frame, [window_size - 2], predicted[-1], axis=0)
                prediction_seqs.append(predicted)
            return prediction_seqs
        else:
            print("[Model] Predicting Sequences Multiple...")
            prediction_seqs = []
            for i in range(int(len(data) / prediction_len)):
                print(data.shape)
                cur_frame = data[i * prediction_len]
                print(cur_frame)
                predicted = []
                for j in range(prediction_len):
                    predict_result = self.model.predict(cur_frame[np.newaxis, :, :])
                    print(predict_result)
                    final_result = predict_result[0, 0]
                    predicted.append(final_result)
                    cur_frame = cur_frame[1:]
                    print(cur_frame)
                    cur_frame = np.insert(cur_frame, [window_size - 2], predicted[-1], axis=0)
                    print(cur_frame)
                prediction_seqs.append(predicted)

    def predict_sequence_full(self, data, window_size):
        print("[Model] Predicting Sequences Full")
        cur_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(cur_frame[np.newaxis, :, :])[0, 0])
            cur_frame = cur_frame[1:]
            cur_frame = np.insert(cur_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted

    # 这里可使用两种，加载模型评估和就地进行模型评估
    def model_evaluate(self, x_test, y_test, name, addition=False):
        """
        模型评估，如果要加载模型进行评估，取消注释，添加filepath参数
        就地的有mse/mae/mape/mlse/rmse/r2/lse/cs
        :param addition: 是否需要格外增加评估方法
        :param x_test:
        :param y_test:
        :param name:
        :return: Null
        """
        path = f"loss/{name}.csv"
        if addition:
            # 额外的评估
            mse_score = mean_squared_error(x_test[0], y_test[:, 0])
            r2 = r2_score(x_test[0], y_test[:, 0])
            mae_score = mean_absolute_error(x_test[0], y_test[:, 0])
            mape_score = np.mean(np.abs((x_test[0] - y_test[:, 0]) / x_test[0])) * 100
            smape = 2.0 * np.mean(
                np.abs(y_test[:, 0] - x_test[0]) / (np.abs(y_test[:, 0]) + np.abs(x_test[0]))) * 100
            print(mse_score, r2, mae_score, mape_score, smape)
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(mse_score)
            print(f"[Loss] save as {name}.csv in {path}")
        else:
            # self.load_model(filepath)
            loss = self.model.evaluate(x_test, y_test, verbose=1)
            print(loss)
