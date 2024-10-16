"""
配置并启动仿真 摩尔透镜用
By 周王哲
2024.10.16
"""
import os
import time
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

period =500e-6  # 按单元结构周期仿真mm
length = 3
num_points = np.round(length / period)
# num_points = 4096
G = Grid(num_points, num_points * period)

# 建立基本结构参数
efl = [2, 4, 8, 12, 16, 20]  # 有效焦距，单位mm，用于图片命名
d_h1 = [0.0,2.09536116343, 6.179152444322, 1.02103448842e1, 1.422430876415e1, 1.822894084294e1]
d_bfl = [1.555790264959, 3.563493550424, 7.551701054241, 1.154570246076e1, 1.554137492033e1, 1.953695252082e1]
aper = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
fields = [[0.0, 1.16375978966424e1, 2.378616201553734e1, 3.720964602839081e1, 5.367607888974954e1],
         [0.0, 5.741961360433048, 1.154180490899745e1, 1.746286069337432e1, 2.358079480315429e1],
         [0.0, 2.866406673728428, 5.739931492423351, 8.627851964120964, 1.153777414984913e1],
         [0.0, 1.910336582520187, 3.822779882376703, 5.739457468523683, 7.66253918141267],
         [0.0, 1.43259573974526, 2.866080575196699, 4.301348549639792, 5.739303677308523],
         [0.0, 1.146019874196124, 2.292495154530162, 3.439882869384618, 4.588643307521111]]
# 波长和折射率
wavelength_vacuum = 532 * 1e-6  # 真空波长，单位mm
refractive_index = 1.4607063  # SiO2折射率

# name = "actual_cylinder_1064"
name = "ideal_532"
# 是否使用GPU加速
gpu_acceleration=True
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

save_path = r'E:/Research/WavePropagation/metalens_simulation/Moire_10x/{}_{}/'.format(current_date,
                                                                                              name)  # 数据存放目录
if not os.path.exists(save_path):
    # 如果不存在则创建
    os.makedirs(save_path)
    print(f"文件夹 '{save_path}' 已创建。")
else:
    print(f"文件夹 '{save_path}' 已存在。")
# 建立log
for i in range(len(efl)):
    logger.add(save_path + 'f_{:.1f}.log'.format(efl[i]))
    logger.info("EFFL = {:.1f} mm".format(efl[i]))
    logger.info("simulation mesh:{}×{}  simulation lenth:{:.5f} mm".format(num_points, num_points, num_points * period))
    t0 = time.time()

    # 镜片
    L1 = Lens(G)
    L1.binary2_d(0.72, 1, 1, [-5.848595615954e2, 3.20546768416e1, -1.422681034594e1, 2.984845643491], unit_phase, unit_t, boundaries,
                 gpu_acceleration=gpu_acceleration)

    np.save(save_path + "Lens1.npz", L1.phase)
    # L1.plot_phase(save_path + 'L1_d_')
    L2 = Lens(G)
    L2.binary2_d(0.3, 1, 1, [3.157248960338e3,-2.062812665413e3, 1.207061495482e4,-6.173754564586e4], unit_phase, unit_t, boundaries,
                 gpu_acceleration=gpu_acceleration)
    np.save(save_path + "Lens2.npz", L2.phase)
    # L2.plot_phase(save_path + 'L2_d_')

    for y_field in fields[i]:
        # 光源
        s = Source(G.d2_x, G.d2_y, wavelength_vacuum, 1)
        s.plane_wave(np.pi / 2, np.pi / 2 + y_field)
        t1 = time.time()
        logger.success("lens initialization complete, Elapsed time: {:.2f}".format(t1-t0))
        # L3.plot_phase(save_path + 'L3_d_')
        d_lens = 0.75

        # method = 'FFT-DI'
        # method = 'AS'
        method = "BL-AS"
        logger.info("Using {} method".format(method))
        three_element_zoom_system(s, L1, L2, L3, G, d_lens, d_12[i], d_23[i], d_bfl[i], name, refractive_index,
                                  save_path, magnification=200, sampling_point=100, interval=1, method=method, show=False,
                                  gpu_acceleration=True)
