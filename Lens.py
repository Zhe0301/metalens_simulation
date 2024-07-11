import numpy as np


class Lens:
    def __init__(self, d2_x, d2_y, d2_r, t=1):
        #  坐标矩阵d2_x, d2_y，坐标平方和（极坐标径向坐标）d2_r，光强透过率t
        self.d2_x = d2_x
        self.d2_y = d2_y
        self.t = t
        self.d2_r = d2_r
        self.complex_amplitude_t = None
        self.phase = None
        self.mask = None
        self.amplitude = None

    # 二元面2模拟
    def binary2(self, r_max, m, r_0, a):
        # 最大半径r_max,衍射阶数m,归一化半径r_0,单位 mm,多项式系数数组a
        mask_index = self.d2_r <= r_max
        self.mask = np.zeros_like(self.d2_r)
        self.mask[mask_index] = 1
        self.phase = np.zeros_like(self.d2_r)

        for i, a_i in enumerate(a):  # enumerate组合为一个索引序列，同时列出数据和数据下标
            self.phase += m * a_i * (self.d2_r / r_0) ** (2 * (i + 1))
        self.phase = self.phase * self.mask  # zemax中相位图以角度为单位，此处以弧度
        self.amplitude = np.ones_like(self.d2_r) * self.mask
        self.complex_amplitude_t = self.amplitude * np.exp(1j * self.phase) * self.t * self.mask  # 复振幅透过率

    def ideal_lens(self, r_max, focal_length, wavelength_vacuum, refractive_index=1):
        mask_index = self.d2_r <= r_max
        self.mask = np.zeros_like(self.d2_r)
        self.mask[mask_index] = 1
        self.phase = np.zeros_like(self.d2_r)
        self.phase = -(focal_length - np.sqrt(focal_length ** 2 + self.d2_r ** 2)) * 2 * np.pi/wavelength_vacuum
        self.amplitude = np.ones_like(self.d2_r)
        self.phase = self.phase * self.mask
        self.complex_amplitude_t = self.amplitude * np.exp(- 1j * self.phase) * self.t * self.mask
        # self.complex_amplitude_t = self.amplitude * np.exp(-1j * self.phase) * self.t

    def hole(self, r_max):
        mask_index = self.d2_r <= r_max
        self.mask = np.zeros_like(self.d2_r)
        self.mask[mask_index] = 1
        self.amplitude = np.ones_like(self.d2_r) * self.mask
        self.phase = np.zeros_like(self.d2_r) * self.mask
        self.complex_amplitude_t = self.amplitude * np.exp(1j * self.phase) * self.t * self.mask
