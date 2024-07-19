"""
    @author:tb
    @time:2024-5-12
    绘制图形做可视化展示
"""
import os.path
import matplotlib
import pandas as pd
import glob

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class PicturePlt:
    @staticmethod
    def plot_results(predicted_data, true_data, name, path):
        """
        绘制真实值与预测值比较的图(序列->单值)
        :param path: 保存的路径
        :param name: 保存的名字
        :param predicted_data:预测值
        :param true_data: 真实值
        :return: Null
        """
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        ax.plot(true_data, label="True Data")
        plt.plot(predicted_data, label="Prediction")
        plt.legend()
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(f"{path}/{name}.jpg")
        plt.show()

    @staticmethod
    def plot_results_multiple(predicted_data, true_data, predicted_len, name, path):
        """
        绘制真实值与预测值比较的图(序列->序列)
        :param path: 保存的路径
        :param name: 保存的名字
        :param predicted_data: 预测值
        :param true_data: 真实值
        :param predicted_len: 预测长度
        :return: Null
        """
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        ax.plot(true_data, label="True Data")
        plt.legend()
        for i, data in enumerate(predicted_data):
            padding = [None for p in range(i * predicted_len)]
            plt.plot(padding + data, label="Prediction")
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(f"{path}/{name}.jpg")
        plt.show()

    @staticmethod
    def plot_loss_multiple(exp, dirname="loss", single=True):
        """
        读取文件夹下面所有的loss.csv文件，并绘制在一张图上
        :param exp: 试验次数
        :param single: 是否是同一个模型
        :param dirname: 文件夹名字，默认是loss下面
        :return: Null
        """
        csv_files = glob.glob(dirname + '/*.csv')
        if len(csv_files) < 4:
            print("文件数量不足，需要大于4")
            return None
        # 创建一个新的图形
        plt.figure(figsize=(10, 6))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        # 遍历每个 CSV 文件并读取数据并绘制曲线
        for i, file in enumerate(csv_files):
            # 使用 pandas 读取 CSV 文件
            df = pd.read_csv(file, header=None, names=['y'])
            # 使用默认索引作为 x 值
            x_values = range(len(df))
            color = colors[i % len(colors)]
            # 绘制曲线
            plt.plot(x_values, df['y'], label=file.split('/')[-1].split('.')[0], color=color)

        # 添加标题和标签
        if single:
            plt.title("Comparison of four loss functions")
        else:
            plt.title("Compare four models")
        plt.xlabel("Index")
        plt.ylabel("Loss")

        plt.legend()
        if single:
            plt.savefig(f"{exp}/loss_multiple.jpg")
        else:
            plt.savefig(f"{exp}/model_multiple.jpg")
        plt.show()

    @staticmethod
    def plot_loss_single(train: list, val: list, exp):
        """
        在做完一次试验后，马上画出训练集和测试集的损失函数图
        :param train: 训练集损失函数
        :param val: 测试集损失函数
        :param exp: 实验次数
        :return: Null
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        ax1.plot(train, label="train", color="orange")
        ax1.set_title('Train Loss')
        ax1.legend()

        ax2.plot(val, label="val", color="red")
        ax2.set_title('Val Loss')
        ax2.legend()

        plt.tight_layout()
        save_path = f"{exp}_loss_single.jpg"
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_data_haar():
        filepath1 = ''
        filepath2 = ''
        dataframe = pd.read_csv(filepath1, usecols='data')
        data = dataframe.get('data').values

        haar_frame = pd.read_csv(filepath2, usecols='data')
        haar = haar_frame.get('data').values

        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        ax.plot(data, label="Origin Data")
        plt.legend()
        for i, data in enumerate(haar):
            padding = [None for p in range(i * haar)]
            plt.plot(padding + data, label="haar")
        plt.savefig(f"result/haar deal.jpg")
        plt.show()

    @staticmethod
    def plot_wavelet_multiple(dirname="data/wave", single=False):
        csv_files = glob.glob(dirname + '/*.csv')
        # 创建一个新的图形
        plt.figure(figsize=(10, 6))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        # 遍历每个 CSV 文件并读取数据并绘制曲线
        for i, file in enumerate(csv_files):
            # 使用 pandas 读取 CSV 文件
            df = pd.read_csv(file, header=None, names=['y'])
            # 使用默认索引作为 x 值
            x_values = range(len(df))
            color = colors[i % len(colors)]
            # 绘制曲线
            plt.plot(x_values, df['y'], label=file.split('/')[-1].split('.')[0], color=color)

        # 添加标题和标签
        if single:
            plt.title("Compare the original data with the deionised data")
        else:
            plt.title("Compare different noise reduction functions")
        plt.xlabel("Index")
        plt.ylabel("Gas")

        plt.legend()
        if single:
            plt.savefig(f"result/True_Wave.jpg")
        else:
            plt.savefig(f"result/Different_Wave.jpg")
        plt.show()

    @staticmethod
    def local_wave_true(data1, filename):
        data2 = pd.read_csv(f'data/wave/{filename}')
        # data2 = data2[14806:14956]
        data2 = data2.reset_index(drop=True)
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        ax.plot(data1, label="Origin Data")
        plt.legend()
        plt.plot(data2, label=f'{filename}')  # 显示去噪结果
        plt.legend()
        plt.savefig(f'result/{filename}小波降噪对比.jpg')
        plt.show()

    @staticmethod
    def plot_wave_results(true_1, predicted_1, true_2, predicted_2, name, path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

        ax1.plot(true_1, label="True Data")
        ax1.plot(predicted_1, label='Prediction')
        ax1.set_title('Before')
        ax1.legend()

        ax2.plot(true_2, label="True Data")
        ax2.plot(predicted_2, label="Prediction")
        ax2.set_title('After')

        ax2.legend()

        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(f"{path}/{name}.jpg")
        plt.show()
