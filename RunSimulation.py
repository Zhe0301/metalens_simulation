"""
配置并启动仿真
By 周王哲
2024.7.22
"""
import os
from datetime import datetime

from loguru import logger
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
# from MySimulation_Zoom_3_Element import three_element_zoom_system
from MySimulation_Zoom_3_Element_Reverse import three_element_zoom_system
from Grid import Grid
from Lens import Lens
from Source import Source
from PropOperator import PropOperator
from Tools import *



# 建立仿真网络

period = 347e-6  # 按单元结构周期仿真mm
length = 2
num_points = round(length/period)
# num_points = 4096

G = Grid(num_points, num_points*period)
efl = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2]  # 有效焦距，单位mm，用于图片命名
d_12 = [1.100000006257,1.785825240396,2.083347319444,2.25568813143,2.364753377126,2.386287993685]
d_23 = [2.924563548595,2.020781241757,1.568999114497,1.26635292265,1.031953399776,7.999999810325e-1]
d_bfl = [1.975436417502,2.193393520095,2.347653672864,2.477959067576,2.60329289574,2.813712156567]
# 仿真参数
wavelength_vacuum = 632.8 * 1e-6  # 真空波长，单位mm
refractive_index = 1.457  # SiO2折射率

# 相位离散量
discrete = 8
# 建立仿真目录
current_date = datetime.now().strftime('%Y%m%d')
# %y 两位数的年份表示（00-99）
# %Y 四位数的年份表示（000-9999）
# %m 月份（01-12）
# %d 月内中的一天（0-31）
# %H 24小时制小时数（0-23）
# %I 12小时制小时数（01-12）
# %M 分钟数（00=59）
# %S 秒（00-59）

save_path = r'E:/Research/WavePropagation/metalens_simulation/Zoom_6×/{}_discrete_{}/'.format(current_date,
                                                                                              discrete)  # 数据存放目录
if not os.path.exists(save_path):
    # 如果不存在则创建
    os.makedirs(save_path)
    print(f"文件夹 '{save_path}' 已创建。")
else:
    print(f"文件夹 '{save_path}' 已存在。")
# 建立log
for i in range(len(efl)):
    logger.add(save_path+'f_{:.1f}.log'.format(efl[i]))
    logger.info("EFFL = {:.1f} mm".format(efl[i]))
    logger.info("simulation mesh:{}×{}. simulation lenth:{:.5f} mm".format(num_points, num_points,num_points*period))

    # 光源
    s = Source(G.d2_x, G.d2_y, wavelength_vacuum, 1)
    s.plane_wave(np.pi / 2, np.pi / 2)

    # 镜片
    L1 = Lens(G)
    L1.binary2_d(0.375, 1, 1, [-1.134335348014e3, 1.997482968732e2, -7.744689756243e2,
                              2.350435931257e3, 3.245672141492e3, -2.507104417945e4], d=discrete)

    np.save(save_path + "Lens1.npz",L1.phase_2pi)
    # L1.plot_phase(save_path + 'L1_d_')
    L2 = Lens(G)
    L2.binary2_d(0.25, 1, 1, [6.658448719319e3, -1.180564189085e4, 3.123599533268e5,
                              -8.915821224249e6, 1.353360477873e8, -8.261794056944e8], d=discrete)
    np.save(save_path + "Lens2.npz", L2.phase_2pi)
    # L2.plot_phase(save_path + 'L2_d_')
    L3 = Lens(G)
    L3.binary2_d(0.95, 1, 1, [-3.655457426687e3, 2.169008952537e2, -1.402136806599e2,
                              2.517459986023e2, -2.498340816648e2, 9.683857736021e1], d=discrete)
    np.save(save_path + "Lens3.npz", L3.phase_2pi)
    # L3.plot_phase(save_path + 'L3_d_')
    d_lens = 0.6


    # method = 'FFT-DI'
    # method = 'AS'
    method = "BL-AS"
    logger.info("Using {} method".format(method))
    three_element_zoom_system(s, L1, L2, L3, G, d_lens, d_12[i], d_23[i], d_bfl[i], efl[i], refractive_index,
                              save_path, magnification=200, sampling_point=0, interval=1, method=method, show=True,
                              gpu_acceleration=False)
