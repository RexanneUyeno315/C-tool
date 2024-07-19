"""
    @author:tb
    @time:2024-05-9
    对唐宇迪的课程进行复现并更改为自己的模块
    进行数据预处理
    更新：添加了对loss函数的保存
"""
import math
import csv
import random

import numpy as np
import pandas as pd
import pywt


class DataLoader:
    """
        加载转化数据，以便lstm训练
    """

    def __init__(self, filename, split, cols):
        """
        初始化变量，加载数据并划分数据集
        :param filename: 文件名
        :param split: 划分的比例 0-1
        :param cols: 需要读取的列
        """
        dataframe = pd.read_csv(filename)
        len_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:len_split]
        self.data_test = dataframe.get(cols).values[len_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        """
        处理测试集数据
        :param seq_len:
        :param normalise:
        :return:
        """
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])
        data_windows = np.array(data_windows).astype(float)
        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return np.array(x), np.array(y)

    def get_train_data(self, seq_len, normalise):
        """
        创建并处理训练集数据
        :param seq_len:
        :param normalise:
        :return:
        """
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        """
        训练数据生成器，从文件中读取列名，分割测试集与训练集后读取训练数据
        :param seq_len:序列长度
        :param batch_size:批次
        :param normalise:归一化
        :return:一批训练数据
        """
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        """
        滑动窗口
        :param i:每次放入的那一组数据(步长)
        :param seq_len: 步长/序列长度
        :param normalise:归一化操作
        :return:如果步长为50，则x为前49个数，y为第50个数
            x:0->一个步长的序列
            y:预测的一个数据
        """
        window = self.data_train[i:i + seq_len]
        window = self.normalise_window(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    @staticmethod
    def normalise_window(window_data, single_window=False):
        """
        归一化窗口，用来处理数据，使数据分布在0-1之间，减少过拟合和梯度爆炸问题发生
        :param window_data:滑动窗口数据
        :param single_window:单特征还是多特征/传入的数据是否为一列
        :return:归一化之后的数据
        """
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    @staticmethod
    def save_loss(data: list, name):
        """
        以csv格式保存损失函数(测试集，5种损失函数，4种优化器)
        :param data: 损失函数列表
        :param name: 损失函数保存名字
        :return: Null
        """
        path = f"loss/{name}.csv"
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in data:
                writer.writerow([i])
        print(f"[Loss] save as {name}.csv in {path}")


def inverse_normalise_window(normalised_window, min_val=0, max_val=8):
    """
    Inverse normalization of window data1.
    :param normalised_window: Normalized window data1.
    :param min_val: Minimum value used for normalization.
    :param max_val: Maximum value used for normalization.
    :return: Original data1.
    """
    original_data = (normalised_window * (max_val - min_val)) + min_val
    return original_data


def save_csv(data, name):
    path = f"data/{name}.csv"
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in data:
            writer.writerow([i])
    print(f"[Data] save as {name}.csv in {path}")


def data_deal(path="data/data25000.csv"):
    """
    读取csv文件并进行处理
    :param path: csv文件路径
    :return: Null
    """
    data = []
    dataframe = pd.read_csv(path).get('data').values
    for i in dataframe:
        if i < 0.2:
            i = 0.2 + round(random.uniform(0.1, 0.2), 2)
            print(i)
            data.append(i)
        elif i > 1.2:
            i = 1.1 + round(random.uniform(0.1, 0.2), 2)
            print(i)
            data.append(i)
        else:
            i = round(i, 2)
            data.append(i)

    save_csv(data, name='deal_data')


# 小波降噪处理数据
def wavelet_noising(new_df, Basis, method, level, threshold=0.5, wae=False):
    """
    小波降噪
    :param wae: 是否需要小波
    :param new_df: DataFrame格式的数据
    :param Basis: 小波基
    :param method: 小波方法，软阈值/软硬折中
    :param level: 小波降噪层级
    :param threshold: 针对软硬这种方法的阈值设定，默认0.5
    :return:
    """
    data = new_df.values.T.tolist()
    w = pywt.Wavelet(Basis)
    cad = []
    tmp = pywt.wavedec(data, w, level=level)
    for i in tmp:
        i = i.squeeze(axis=0)
        cad.append(i)

    length0 = len(data[0])
    abs_cd1 = np.abs(np.array(cad[-1]))
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    usecoeffs = [0] * (level + 1)
    usecoeffs[0] = cad[0]
    a = threshold
    if method == "soft":
        # 软阈值方法
        for j, i in enumerate(cad[-1:0:-1]):
            for k in range(len(i)):
                if abs(i[k]) >= lamda / np.log2(j + 2):
                    i[k] = _sgn(i[k]) * (abs(i[k]) - lamda / np.log2(j + 2))
                else:
                    i[k] = 0.0
            usecoeffs[level - j] = i
        recoeffs = pywt.waverec(usecoeffs, w)
        save_mav(recoeffs, Basis, method, level, wae)
        return recoeffs
    elif method == "has":
        for j, i in enumerate(cad[-1:0:-1]):
            for k in range(len(i)):
                if abs(i[k]) >= lamda:
                    i[k] = _sgn(i[k]) * (abs(i[k]) - a * lamda)
                else:
                    i[k] = 0.0
            usecoeffs[level - j] = i
        recoeffs = pywt.waverec(usecoeffs, w)
        save_mav(recoeffs, Basis, method, level, wae)
        return recoeffs
    else:
        print("[Method] Error,Try again")


# sgn函数
def _sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0


def save_mav(data, Basis, method, level, wae):
    if wae:
        path = f'data/wave/{method}_{Basis}_{level}_150_wavelet.csv'
    else:
        path = f'data/wave/{method}_{Basis}_{level}_wavelet.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for item in data:
            writer.writerow([item])
        print(
            f"[Wave Data] save in {path}")
