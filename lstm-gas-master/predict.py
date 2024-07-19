"""
    @author:tb
    @time:2024-5-15
    加载训练模型，进行画图对比
"""
import pandas as pd
from pandas import read_csv
import json
import os
from core.data_processor import data_deal, wavelet_noising
from core.plt_picture import PicturePlt
from core.model import Model


# 加载模型/预测/绘图
def main(model_name, exp):
    plt = PicturePlt()
    model = Model()
    # 读取数据
    configs = json.load(open('config.json', 'r'))
    file_deal = 'data/deal_data.csv'
    path_wave = 'data/wave/'

    Basis = configs['data']['wavelet']['Basis']
    method = configs['data']['wavelet']['method']
    level = configs['data']['wavelet']['level']

    file_150 = 'data/data150.csv'
    wave_150_name = f'{method}_{Basis}_{level}_150_wavelet.csv'

    wave_name = f'{method}_{Basis}_{level}_wavelet.csv'
    file_wave = os.path.join(path_wave, wave_name)

    file_150_wave = os.path.join(path_wave, wave_150_name)

    SAVE_DIR = configs['model']['save_dir']
    PATH = f'result/exp{exp}'
    model_path = os.path.join(SAVE_DIR, model_name)
    if not os.path.isfile(file_deal):
        data_deal()
    if not os.path.isfile(file_150_wave):
        dataframe = read_csv(file_150)
        wavelet_noising(dataframe, Basis, method, level, wae=True)

        # 小波处理前后对比图,150个数据
        plt.local_wave_true(dataframe, wave_150_name)

    if not os.path.isfile(file_wave):
        dataframe = read_csv(file_deal)
        wavelet_noising(dataframe, Basis, method, level)
        # 小波处理前后对比图,25000个数据
        plt.local_wave_true(dataframe, wave_name)

    # 不同小波对比图，wave下所有csv文件
    plt.plot_wavelet_multiple()

    # 预测数据传入
    data1 = read_csv(file_150)
    data2 = read_csv(file_150_wave)

    # 小波降噪局部对比，150个数据
    plt.local_wave_true(data1, wave_150_name)

    # 加载模型进行预测
    model.load_model(model_path)
    predict_1 = model.predict_point_by_point(data1)
    predict_2 = model.predict_point_by_point(data2)

    # 降噪前
    plt.plot_results(predict_1, data1, 'Origin', PATH)

    # 降噪后
    plt.plot_results(predict_2, data2, 'WAE', PATH)

    # 效果对比
    plt.plot_wave_results(data1, predict_1, data2, predict_2, 'Origin_WAE', PATH)


if __name__ == '__main__':
    model1 = '20240513-0509-adam-e10.keras'
    num = input('输入试验次数：')
    main(model1, num)
