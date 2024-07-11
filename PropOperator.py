# 角谱传输
import numpy as np
from matplotlib import pyplot as plt


class PropOperator:
    def __init__(self, d2_fft_x, d2_fft_y, wavelength_vacuum, dist, refractive_index=1, paraxial=False):
        #  complex_amplitude输入复振幅，wavelength_vacuum真空中波长，d2_fft_x，d2_fft_y频谱坐标矩阵
        self.e_out = None
        self.dist = dist
        self.n = refractive_index  # 折射率
        self.wavelength = wavelength_vacuum / self.n
        self.k_prop = 2 * np.pi / self.wavelength
        grid_d2_fft_x2 = d2_fft_x ** 2
        grid_d2_fft_y2 = d2_fft_y ** 2
        self.mat = np.ones_like(grid_d2_fft_x2, dtype=complex)
        if paraxial:
            self.mat = np.exp(1j * (self.k_prop * self.dist -
                                    np.pi * self.wavelength * (grid_d2_fft_x2 + grid_d2_fft_y2) * self.dist))
        else:
            condition = 1 / self.wavelength ** 2 - grid_d2_fft_x2 - grid_d2_fft_y2  # 根号下内容
            self.mat[condition > 0] = np.exp(
                1j * 2 * np.pi * np.sqrt(condition[condition > 0]) * self.dist)
            self.mat[condition <= 0] = np.exp(
                - 2 * np.pi * np.sqrt(- condition[condition < 0]) * self.dist)

    def prop(self, complex_amplitude):
        wave_fft = np.fft.fftshift(np.fft.fft2(complex_amplitude))
        # plt.matshow(np.angle(complex_amplitude))
        # plt.show()
        wave_prop = wave_fft * self.mat
        self.e_out = np.fft.ifft2(np.fft.fftshift(wave_prop))
        return self.e_out
