"""
镜头和孔径类
By 周王哲
2024.7.11
"""
import numpy as np
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


class Lens:
    def __init__(self, Grid, t=1):
        """
        Grid网格类，
        t光强透过率
        """
        self.phase_2pi = None
        self.Grid = Grid
        self.t = t
        self.complex_amplitude_t = None
        self.phase = None
        self.mask = None
        self.amplitude = None

    def binary2(self, r_max, m, r_0, a):
        """
        二元面2模拟:使用镜像坐标的偶次多项式描述相位面
        r_max最大半径,单位 mm;
        m衍射阶数;
        r_0归一化半径,单位 mm;
        a多项式系数数组
        """
        mask_index = self.Grid.d2_r <= r_max
        self.mask = np.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.phase = np.zeros_like(self.Grid.d2_r)

        for i, a_i in enumerate(a):  # enumerate组合为一个索引序列，同时列出数据和数据下标
            self.phase += m * a_i * (self.Grid.d2_r / r_0) ** (2 * (i + 1))
        self.phase = self.phase * self.mask  # zemax中相位图以角度为单位，此处以弧度
        self.phase_2pi = np.mod(self.phase, 2 * np.pi) * self.mask  # 相位，取余数到0到2pi
        self.amplitude = np.ones_like(self.Grid.d2_r) * self.mask
        self.complex_amplitude_t = self.amplitude * np.exp(1j * self.phase) * self.t * self.mask  # 复振幅透过率

    def binary2_d(self, r_max, m, r_0, a, d=8):
        """
        相位离散型的二元面2，
        r_max最大半径,
        m衍射阶数,归一化半径r_0,单位 mm,多项式系数数组a，d相位的均分份数
        """
        # 相位离散型二元面2，最大半径r_max,衍射阶数m,归一化半径r_0,单位 mm,多项式系数数组a，d相位的均分份数
        mask_index = self.Grid.d2_r <= r_max
        self.mask = np.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.phase = np.zeros_like(self.Grid.d2_r)

        for i, a_i in enumerate(a):  # enumerate组合为一个索引序列，同时列出数据和数据下标
            self.phase += m * a_i * (self.Grid.d2_r / r_0) ** (2 * (i + 1))
          # zemax中相位图以角度为单位，此处以弧度
        self.phase_2pi = np.mod(self.phase, 2 * np.pi)
        if d > 0:
            interval = 2 * np.pi / d
            # 计算离散相位
            # self.phase_num = np.floor( self.phase_2pi / interval)
            self.phase_2pi = (np.floor( self.phase_2pi / interval) + 1 / 2) * interval
            self.phase_2pi =  self.phase_2pi * self.mask
            self.phase = self.phase_2pi
        else:
            print("需输入正确的离散值,结果未离散")
            self.phase = self.phase * self.mask
        self.amplitude = np.ones_like(self.Grid.d2_r) * self.mask
        self.complex_amplitude_t = self.amplitude * np.exp(1j *  self.phase) * self.t * self.mask  # 复振幅透过率

    def ideal_lens(self, r_max, focal_length, wavelength_vacuum):
        mask_index = self.Grid.d2_r <= r_max
        self.mask = np.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.phase = np.zeros_like(self.Grid.d2_r)
        self.phase = -(focal_length - np.sqrt(focal_length ** 2 + self.Grid.d2_r ** 2)) * 2 * np.pi / wavelength_vacuum
        self.phase = self.phase * self.mask
        self.phase_2pi = (self.phase % (2 * np.pi)) * self.mask  # 相位，取余数到0到2pi
        self.amplitude = np.ones_like(self.Grid.d2_r) * self.mask
        self.amplitude = np.ones_like(self.Grid.d2_r)
        self.complex_amplitude_t = self.amplitude * np.exp(- 1j * self.phase) * self.t * self.mask
        # self.complex_amplitude_t = self.amplitude * np.exp(-1j * self.phase) * self.t

    def hole(self, r_max):
        mask_index = self.Grid.d2_r <= r_max
        self.mask = np.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.amplitude = np.ones_like(self.Grid.d2_r) * self.mask
        self.phase = np.zeros_like(self.Grid.d2_r) * self.mask
        self.phase_2pi = (self.phase % (2 * np.pi)) * self.mask  # 相位，取余数到0到2pi
        self.complex_amplitude_t = self.amplitude * np.exp(1j * self.phase) * self.t * self.mask

    def square_hole(self, lenth_x, lenth_y):
        mask_index = (abs(self.Grid.d2_x) <= lenth_x / 2) & (abs(self.Grid.d2_x) >= - lenth_x / 2) & (
                    self.Grid.d2_y <= lenth_y / 2) & (self.Grid.d2_y >= - lenth_y / 2)
        self.mask = np.zeros_like(self.Grid.d2_r)
        self.mask[mask_index] = 1
        self.amplitude = np.ones_like(self.Grid.d2_r) * self.mask
        self.phase = np.zeros_like(self.Grid.d2_r) * self.mask
        self.phase_2pi = (self.phase % (2 * np.pi)) * self.mask  # ��位，取余数到0到2pi
        self.complex_amplitude_t = self.amplitude * np.exp(1j * self.phase) * self.t * self.mask

    def plot_phase(self, save_path=r'/'):
        plt.figure(figsize=(16, 9))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(self.Grid.d2_x, self.Grid.d2_y, self.phase, cmap="jet")
        plt.title('Phase Distribution')
        plt.xlabel(r'$x$(mm)')
        plt.ylabel(r'$y$(mm)')
        cb = plt.colorbar()
        cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
        plt.subplot(1, 2, 2)
        plt.pcolormesh(self.Grid.d2_x, self.Grid.d2_y, self.phase_2pi, cmap="jet")
        plt.title(r'Phase Distribution $0 ~ 2\pi$')
        plt.xlabel(r'$x$(mm)')
        plt.ylabel(r'$y$(mm)')
        cb = plt.colorbar()
        cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
        if save_path is not None:
            plt.savefig(save_path + 'Phase.png')
        plt.show()

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
