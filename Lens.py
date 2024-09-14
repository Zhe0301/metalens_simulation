"""
镜头和孔径类
By 周王哲
2024.7.11
"""
import time
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cupy as cp
import matplotlib

config = {"font.family": 'serif',
          "font.size": 20,
          "mathtext.fontset": 'stix',
          "font.serif": ['Times New Roman']
          }
rcParams.update(config)

matplotlib.use('qt5agg')


class Lens:
    def __init__(self, Grid,):
        """
        :param: Grid网格类，
        :param: t光强透过率
        """
        self.Grid = Grid
        self.complex_amplitude_t = None
        self.phase = None
        self.mask = None
        self.amplitude = None

    def binary2(self, r_max, m, r_0, a, t=1):
        """
        二元面2模拟:使用镜像坐标的偶次多项式描述相位面
        :param: r_max最大半径,单位 mm;
        :param: m衍射阶数;
        :param: r_0归一化半径,单位 mm;
        :param: a多项式系数数组;
        :param: t 能量透过率
        """
        mask_index = self.Grid.d2_r <= r_max
        self.mask = np.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.phase = np.zeros_like(self.Grid.d2_r)

        for i, a_i in enumerate(a):  # enumerate组合为一个索引序列，同时列出数据和数据下标
            self.phase += m * a_i * (self.Grid.d2_r / r_0) ** (2 * (i + 1))
        self.phase = self.phase * self.mask  # zemax中相位图以角度为单位，此处以弧度
        self.amplitude = np.ones_like(self.Grid.d2_r) * self.mask
        self.complex_amplitude_t = self.amplitude * np.exp(1j * self.phase) * t * self.mask  # 复振幅透过率

    def binary2_d(self, r_max, m, r_0, a, unit_phase=None, unit_t=None,  boundaries=None, gpu_acceleration=True):
        """
        相位离散型的二元面2，

        :param r_max:  镜片最大半径;
        :param m: 衍射阶数,d相位的均分份数;
        :param r_0: 归一化半径，单位mm
        :param a: 多项式系数数组;
        :param unit_phase: 数组，离散单元的相位值
        :param unit_t: 数组，离散单元的能量透过率
        :param gpu_acceleration: 是否使用GPU加速


        """
        t0 = time.time()
        gpu_acceleration = True


        # 相位离散型二元面2，最大半径r_max,衍射阶数m,归一化半径r_0,单位 mm,多项式系数数组a，d相位的均分份数
        if unit_phase is None:
            unit_phase = np.array([0, 1 / 4, 2 / 4, 3 / 4, 1, 5 / 4, 6 / 4, 7 / 4]) * np.pi + np.pi / 8
        if unit_t is None:
            unit_t = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        if  boundaries is None:
            partition_range = np.array([[0, 1 / 8], [1 / 8, 2 / 8], [2 / 8, 3 / 8], [3 / 8, 4 / 8], [4 / 8, 5 / 8],
                                        [5 / 8, 6 / 8], [6 / 8, 7 / 8], [7 / 8, 1]]) * 2 * np.pi
        xp = cp if gpu_acceleration else np
        if gpu_acceleration:
            unit_phase = cp.asarray(unit_phase)
            unit_t = cp.asarray(unit_t)
            boundaries = cp.asarray(boundaries)
            self.Grid.d2_r = cp.asarray(self.Grid.d2_r)

        mask_index = self.Grid.d2_r <= r_max
        self.mask = xp.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.phase = xp.zeros_like(self.Grid.d2_r)
        self.amplitude = xp.ones_like(self.Grid.d2_r) * self.mask
        for i, a_i in enumerate(a):  # enumerate组合为一个索引序列，同时列出数据和数据下标
            self.phase += m * a_i * (self.Grid.d2_r / r_0) ** (2 * (i + 1))
        # zemax中相位图以角度为单位，此处以弧度
        self.phase = xp.mod(self.phase, 2 * np.pi)
        if len(unit_phase) == 0 or len(unit_t) == 0 or len(boundaries) == 0:
            # 计算离散相位
            logger.error("需输入正确的离散值,结果未离散")
            self.phase = self.phase * self.mask
        else:
            for i, b in enumerate(boundaries):
                mask = (self.phase >= b[0]) & (self.phase < b[1])
                self.phase[mask] = unit_phase[i]
                self.amplitude[mask] = xp.sqrt(unit_t[i])
            self.phase = self.phase * self.mask

        self.complex_amplitude_t = self.amplitude * np.exp(1j * self.phase) * self.mask  # 复振幅透过率
        if  gpu_acceleration:
            self.complex_amplitude_t = cp.asnumpy(self.complex_amplitude_t)
            self.phase = cp.asnumpy(self.phase)
            self.amplitude = cp.asnumpy(self.amplitude)
            self.mask = cp.asnumpy(self.mask)
            self.Grid.d2_r = cp.asnumpy(self.Grid.d2_r)
        t1 = time.time()
        logger.success("binary2_d initialization complete: {:.2f}s".format(t1-t0))

    def ideal_lens(self, r_max, focal_length, wavelength_vacuum,t=1):
        """
        理想透镜
        :param r_max: 镜片最大半径;
        :param focal_length: 焦距;
        :param wavelength_vacuum: 真空中波长
        """
        mask_index = self.Grid.d2_r <= r_max
        self.mask = np.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.phase = np.zeros_like(self.Grid.d2_r)
        self.phase = -(focal_length - np.sqrt(focal_length ** 2 + self.Grid.d2_r ** 2)) * 2 * np.pi / wavelength_vacuum
        self.phase = self.phase * self.mask
        self.amplitude = np.ones_like(self.Grid.d2_r) * self.mask
        self.amplitude = np.ones_like(self.Grid.d2_r)
        self.complex_amplitude_t = self.amplitude * np.exp(- 1j * self.phase) * t * self.mask
        # self.complex_amplitude_t = self.amplitude * np.exp(-1j * self.phase) * t

    def hole(self, r_max,t=1):
        """
        圆孔
        :param r_max: 孔的半径;
        """
        mask_index = self.Grid.d2_r <= r_max
        self.mask = np.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.amplitude = np.ones_like(self.Grid.d2_r) * self.mask
        self.phase = np.zeros_like(self.Grid.d2_r) * self.mask
        self.complex_amplitude_t = self.amplitude * np.exp(1j * self.phase) * t * self.mask

    def square_hole(self, lenth_x, lenth_y,t=1):
        """
        方孔
        :param lenth_x: 孔的x长度;
        :param lenth_y: 孔的y长度;
        """
        mask_index = (abs(self.Grid.d2_x) <= lenth_x / 2) & (abs(self.Grid.d2_x) >= - lenth_x / 2) & (
                self.Grid.d2_y <= lenth_y / 2) & (self.Grid.d2_y >= - lenth_y / 2)
        self.mask = np.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.amplitude = np.ones_like(self.Grid.d2_r) * self.mask
        self.phase = np.zeros_like(self.Grid.d2_r) * self.mask
        self.complex_amplitude_t = self.amplitude * np.exp(1j * self.phase) * t * self.mask

    def plot_phase(self, save_path=r'/'):
        phase_2pi = np.mod(self.phase, 2 * np.pi)
        plt.figure(figsize=(16, 9))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(self.Grid.d2_x, self.Grid.d2_y, self.phase, cmap="jet")
        plt.title('Phase Distribution')
        plt.xlabel(r'$x$(mm)')
        plt.ylabel(r'$y$(mm)')
        cb = plt.colorbar()
        cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
        plt.subplot(1, 2, 2)
        plt.pcolormesh(self.Grid.d2_x, self.Grid.d2_y, phase_2pi, cmap="jet")
        plt.title(r'Phase Distribution $0 ~ 2\pi$')
        plt.xlabel(r'$x$(mm)')
        plt.ylabel(r'$y$(mm)')
        cb = plt.colorbar()
        cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
        if save_path is not None:
            plt.savefig(save_path + 'Phase.png')
        plt.show()
        plt.close()

    def plot_intensity(self, save_path=None):
        plt.figure(figsize=(9, 7))
        plt.pcolormesh(self.Grid.d2_x, self.Grid.d2_y, np.abs(self.complex_amplitude_t) ** 2, cmap="jet")
        plt.title('Intesity')
        plt.xlabel(r'$x$(mm)')
        plt.ylabel(r'$y$(mm)')
        cb = plt.colorbar()
        cb.set_label(r'Intesity')  # 给colorbar添加标题
        if save_path is not None:
            plt.savefig(save_path + 'Intesity.png')
        plt.show()
        plt.close()
