from core.data_processor import wavelet_noising
from core.plt_picture import PicturePlt
import pandas as pd

plt = PicturePlt()
# level1 = 5
# method1 = 'has'
# Basis1 = 'dB10'
data = pd.read_csv('data/deal_data.csv')
# wavelet_noising(data, Basis1, method1, level1)
plt.plot_wavelet_multiple()
