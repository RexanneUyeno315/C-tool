import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt
from math import log
import json


# sgn函数
def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0


def wavelet_noising(new_df, Basis, method, level, threshold=0.5):
    data = new_df.values.T.tolist()
    w = pywt.Wavelet(Basis)  # 选择dB10小波基
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
                    i[k] = sgn(i[k]) * (abs(i[k]) - lamda / np.log2(j + 2))
                else:
                    i[k] = 0.0
            usecoeffs[level - j] = i
        recoeffs = pywt.waverec(usecoeffs, w)
        return recoeffs
    elif method == "has":
        for j, i in enumerate(cad[-1:0:-1]):
            for k in range(len(i)):
                if abs(i[k]) >= lamda:
                    i[k] = sgn(i[k]) * (abs(i[k]) - a * lamda)
                else:
                    i[k] = 0.0
            usecoeffs[level - j] = i
        recoeffs = pywt.waverec(usecoeffs, w)
        return recoeffs
    else:
        print("[Method] Error,Try again")


if __name__ == '__main__':
    path = 'data/deal_data.csv'
    level1 = 3
    method1 = 'has'
    Basis1 = 'dB10'
    # 提取数据
    dataframe = pd.read_csv(path)
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)
    ax.plot(dataframe, label="Origin Data")
    plt.legend()

    data_denoising = wavelet_noising(dataframe, Basis=Basis1, method=method1, level=level1)
    # print(data_denoising)
    # plt.figure()
    plt.plot(data_denoising, label=f'wavelet{level1}')  # 显示去噪结果
    plt.legend()
    plt.savefig(f'result/{method1}_{Basis1}_{level1}层小波降噪对比.jpg')
    plt.show()
