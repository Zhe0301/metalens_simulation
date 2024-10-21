"""
配置并启动仿真
By 周王哲
2024.7.22
"""
import os
import time
from datetime import datetime
import h5py
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
efl = [0.8, 1.6, 2.4, 3.2, 4, 4.8]  # 有效焦距，单位mm，用于图片命名
d_12 = [1.000000131063, 1.846692355066, 2.222548182242, 2.452026972933, 2.6206631695450, 2.776068441725]
d_23 = [3.442929821959, 2.341826841284, 1.798827702873, 1.444344104929, 1.186273959296, 9.999998024761e-1]
d_bfl = [2.307069517468, 2.561481226825, 2.728624214413, 2.853629959039, 2.943064336454, 2.973929452009]


# 波长和折射率
wavelength_vacuum = 1064 * 1e-6  # 真空波长，单位mm
refractive_index = 1.5094756  # SiO2折射率

name = "actual_cylinder_1064"
# name = "ideal_1064"
# 是否使用GPU加速
gpu_acceleration=True
# 相位离散量
discrete = 8
# 理想离散
# unit_phase = np.linspace(0, 2 * np.pi, discrete, endpoint=False) + np.pi / discrete
# unit_t = np.ones_like(unit_phase)
# points = np.linspace(0, 1, discrete + 1)
# boundaries = np.array([[points[i], points[i + 1]] for i in range(discrete)])
# boundaries = boundaries * 2 * np.pi
# 实际离散
# unit_phase = [0, 0.8051036, 1.554511, 2.436956, 3.189646, 3.877856, 4.779246, 5.468306]
unit_phase = [0, 0.771214, 1.611515, 2.329676, 3.129286, 3.941896, 4.700876, 5.486196]

# unit_t = [0.948735, 0.886475, 0.810874, 0.760933, 0.743341, 0.728135, 0.800406, 0.823376]
unit_t = [0.957917, 0.885148, 0.799661, 0.756625, 0.738394, 0.713619, 0.756191, 0.81503]
points = np.linspace(0, 1, discrete + 1)
boundaries = np.array([[points[i], points[i + 1]] for i in range(discrete)])
boundaries = boundaries * 2 * np.pi
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

save_path = r'E:/Research/WavePropagation/metalens_simulation/Zoom_6x/{}_{}/'.format(current_date,
                                                                                              name)  # 数据存放目录
if not os.path.exists(save_path):
    # 如果不存在则创建
    os.makedirs(save_path)
    print(f"文件夹 '{save_path}' 已创建。")
else:
    print(f"文件夹 '{save_path}' 已存在。")
# 建立log
for i in range(len(efl)):
    if i < 4:
        continue
    logger.add(save_path + 'f_{:.1f}.log'.format(efl[i]))
    logger.info("EFFL = {:.1f} mm".format(efl[i]))
    logger.info("simulation mesh:{}×{}  simulation lenth:{:.5f} mm".format(num_points, num_points, num_points * period))
    logger.info("unit phase = {}".format(unit_phase))
    logger.info("unit transmittance = {}".format(unit_t))
    logger.info("boundaries = {}".format(boundaries))
    t0 = time.time()
    # 光源
    s = Source(G, wavelength_vacuum, 1)
    s.plane_wave(np.pi / 2, np.pi / 2)

    # 镜片
    L1 = Lens(G)
    L1.binary2_d(0.72, 1, 1, [-5.848595615954e2, 3.20546768416e1, -1.422681034594e1, 2.984845643491], unit_phase, unit_t, boundaries,
                 gpu_acceleration=gpu_acceleration)
    with h5py.File(save_path + "Lens1.h5", 'w') as f:
        dset = f.create_dataset('Lens1_phase', data=L1.phase, compression='gzip', compression_opts=9)
    # L1.plot_phase(save_path + 'L1_d_')
    L2 = Lens(G)
    L2.binary2_d(0.3, 1, 1, [3.157248960338e3,-2.062812665413e3, 1.207061495482e4,-6.173754564586e4], unit_phase, unit_t, boundaries,
                 gpu_acceleration=gpu_acceleration)
    with h5py.File(save_path + "Lens2.h5", 'w') as f:
        dset = f.create_dataset('Lens2_phase', data=L2.phase, compression='gzip', compression_opts=9)
    # L2.plot_phase(save_path + 'L2_d_')
    L3 = Lens(G)
    L3.binary2_d(1.2, 1, 1, [-1.845376080337e3, 6.935299381526e1, -7.928934283067,	1.711027879901], unit_phase, unit_t, boundaries,
                 gpu_acceleration=gpu_acceleration)
    with h5py.File(save_path + "Lens3.h5", 'w') as f:
        dset = f.create_dataset('Lens3_phase', data=L3.phase, compression='gzip', compression_opts=9)
    t1 = time.time()
    logger.success("lens initialization complete, Elapsed time: {:.2f}".format(t1-t0))
    # L3.plot_phase(save_path + 'L3_d_')
    d_lens = 0.75

    # method = 'FFT-DI'
    # method = 'AS'
    method = "BL-AS"
    logger.info("Using {} method".format(method))
    three_element_zoom_system(s, L1, L2, L3, G, d_lens, d_12[i], d_23[i], d_bfl[i], efl[i], refractive_index,
                              save_path, magnification=200, sampling_point=100, interval=1, method=method, show=False,
                              gpu_acceleration=True)
