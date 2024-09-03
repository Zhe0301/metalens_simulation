"""
配置并启动仿真
By 周王哲
2024.7.22
"""
import sys

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import rcParams

import matplotlib

config = {"font.family": 'serif',
          "font.size": 20,
          "mathtext.fontset": 'stix',
          "font.serif": ['Times New Roman']
          }
rcParams.update(config)

matplotlib.use('qt5agg')
import scipy.interpolate
from scipy import interpolate
from MySimulation import three_element_zoom_system
from Grid import Grid
from Lens import Lens
from Source import Source
from PropOperator import PropOperator
from Tools import *



# 建立仿真网络
G = Grid(2048, 2.5)
# 仿真参数
wavelength_vacuum = 632.8 * 1e-6  # 真空波长，单位mm
refractive_index = 1.457  # SiO2折射率
# 光源
s = Source(G.d2_x, G.d2_y, wavelength_vacuum, 1)
s.plane_wave(np.pi / 2, np.pi / 2)
# 相位离散量
discrete = 0
# 储存位置
save_path = r'E:/Research/WavePropagation/metalens_simulation/Zoom_6×/discrete_{}_'.format(discrete)  # 数据存放目录
# 镜片
L1 = Lens(G)
L1.binary2_d(0.35, 1, 1, [-1.101022155962093e3, 1.561111683794811e2, -4.370083784673994e2,
                          5.311254043289745, 1.319833645746903e4, -4.438157186306786e4], d=discrete)
# L1.plot_phase(save_path + 'L1_d_')
L2 = Lens(G)
L2.binary2_d(0.25, 1, 1, [6.135581697936878e3, -7.322451090814901e3, 1.398975370488559e5,
                          -2.930717746220838e6, 2.458970940437622e7, -8.236160880759975e4], d=discrete)
# L2.plot_phase(save_path + 'L2_d_')
L3 = Lens(G)
L3.binary2_d(1, 1, 1, [-3.562933126401510e3, 1.769595618032093e2, -8.644704466929417e1,
                       1.352032462467216e2, -1.107469566432938e2, 3.104094666020063e1], d=discrete)
# L3.plot_phase(save_path + 'L3_d_')
d_lens = 0.6
d_12 = 9.738333468442362e-1  # 距离，单位mm
d_23 = 3.026166651061564
d_bfl = 1.599999981606190
efl = 0.7  # 有效焦距，单位mm，用于图片命名
method = 'FFT-DI'
# method = 'AS'
# method = "BL-AS"
three_element_zoom_system(s, L1, L2, L3, G, d_lens, d_12, d_23, d_bfl, efl, wavelength_vacuum, refractive_index,
                          save_path, magnification=100, sampling_point=0, interval=1,  method=method,gpu_acceleration=True)
