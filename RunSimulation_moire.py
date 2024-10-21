"""
配置并启动仿真 摩尔透镜用
By 周王哲
2024.10.16
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
from MySimulation_Moire_LFOV import moire_zoom_system
from Grid import Grid
from Lens import Lens
from Source import Source
from PropOperator import PropOperator
from Tools import *

# 建立仿真网络

period = 500e-6  # 按单元结构周期仿真mm
length = 10
# num_points = np.round(length / period)
num_points = 5000
# G = Grid(num_points, num_points * period)
G = Grid(num_points, length)
# 建立基本结构参数
efl = [2, 4, 8, 12, 16, 20]  # 有效焦距，单位mm，用于图片命名
d_h1 = [0.0, 2.09536116343, 6.179152444322, 1.02103448842e1, 1.422430876415e1, 1.822894084294e1]
d_l1 = 2.5  # 片1 厚度，单位mm
d_12 = 5e-3  # 12 镜片的距离，单位mm
d_l2 = 0.6  # 片2 厚度，单位mm
d_bfl = [1.555790264959, 3.563493550424, 7.551701054241, 1.154570246076e1, 1.554137492033e1, 1.953695252082e1]

aper = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # F数
aper = np.divide(efl, aper)
fields = [[0.0, 1.16375978966424e1, 2.378616201553734e1, 3.720964602839081e1, 5.367607888974954e1],
          [0.0, 5.741961360433048, 1.154180490899745e1, 1.746286069337432e1, 2.358079480315429e1],
          [0.0, 2.866406673728428, 5.739931492423351, 8.627851964120964, 1.153777414984913e1],
          [0.0, 1.910336582520187, 3.822779882376703, 5.739457468523683, 7.66253918141267],
          [0.0, 1.43259573974526, 2.866080575196699, 4.301348549639792, 5.739303677308523],
          [0.0, 1.146019874196124, 2.292495154530162, 3.439882869384618, 4.588643307521111]]
quadratic_coef = [-2.974501081243e3, -1.476619562801e3, -7.382225157885e2, -4.921253787297e2, -3.690874360236e2,
                  -2.952675239615e2]  # 二次相位系数
# 波长和折射率
wavelength_vacuum = 532 * 1e-6  # 真空波长，单位mm
refractive_index = 1.4607063  # SiO2折射率
# 莫尔透镜系数
a = (quadratic_coef[0] - quadratic_coef[-1]) / (np.pi / 2)  # 最大旋转交为pi/2
f_offset_wavelength = np.pi / quadratic_coef[-1]  # 0度旋转时的补偿焦距和波长的乘积
logger.info("Coefficient a = {:.3f} ".format(a))
# name = "actual_cylinder_1064"
name = "ideal_532"
# 是否使用GPU加速
gpu_acceleration = True
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
    if i < 5:
        continue
    logger.add(save_path + 'f_{:.1f}.log'.format(efl[i]))
    logger.info("EFFL = {:.1f} mm".format(efl[i]))
    logger.info("simulation mesh:{}×{}  simulation lenth:{:.5f} mm".format(num_points, num_points, num_points * period))
    t0 = time.time()
    # 镜片
    L1 = Lens(G)
    L1.moire_quadratic(3.6, a, 0, round_off=True, f_offset_wavelength=f_offset_wavelength)
    with h5py.File(save_path + "Lens1.h5", 'w') as f:
        dset = f.create_dataset('Lens1_phase', data=L1.phase, compression='gzip', compression_opts=9)
    # L1.plot_phase(save_path + 'L1_')
    phi = (quadratic_coef[i] - quadratic_coef[-1]) / a  # 第二片旋转角
    L2 = Lens(G)
    L2.moire_quadratic(3.6, -a, phi, round_off=True, f_offset_wavelength=f_offset_wavelength)
    with h5py.File(save_path + "Lens2.h5", 'w') as f:
        dset = f.create_dataset('Lens2_phase', data=L2.phase, compression='gzip', compression_opts=9)
    # L2.plot_phase(save_path + 'L2_')
    H = Lens(G)
    H.hole(aper[i] / 2)

    # phase = np.angle(L1.complex_amplitude_t*L2.complex_amplitude_t)
    # phase_2pi = np.mod(phase, 2 * np.pi)
    # plt.figure(figsize=(16, 7))
    # plt.subplot(1, 2, 1)
    # plt.pcolormesh(G.d2_x, G.d2_y, phase, cmap="jet")
    # plt.title('Phase Distribution')
    # plt.xlabel(r'$x$(mm)')
    # plt.ylabel(r'$y$(mm)')
    # cb = plt.colorbar()
    # cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
    # plt.subplot(1, 2, 2)
    # plt.pcolormesh(G.d2_x, G.d2_y, phase_2pi, cmap="jet")
    # plt.title(r'Phase Distribution $0 \sim 2\pi$')
    # plt.xlabel(r'$x$(mm)')
    # plt.ylabel(r'$y$(mm)')
    # cb = plt.colorbar()
    # cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    for y_field in fields[i]:
        # 光源
        s = Source(G, wavelength_vacuum, 1)
        s.plane_wave(np.pi / 2 + y_field / 180 * np.pi, np.pi / 2)
        # s.plot_phase()
        t1 = time.time()
        logger.success("lens initialization complete, Elapsed time: {:.2f}".format(t1 - t0))
        # L3.plot_phase(save_path + 'L3_d_')
        # method = 'FFT-DI'
        # method = 'AS'
        method = "BL-AS"
        logger.info("Using {} method".format(method))
        name = "f_{:.1f}_field_{:.2f}".format(efl[i], y_field)
        moire_zoom_system(s, H, L1, L2, G, d_h1[i], d_l1, d_12, d_l2, d_bfl[i], refractive_index, name,
                          save_path, magnification=200, sampling_point=0, interval=1, method=method, show=True,
                          gpu_acceleration=True)
