# 角谱传输
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt


class PropOperator:
    def __init__(self, Grid, wavelength_vacuum, dist, refractive_index=1, method='AS',
                 paraxial=False, gpu_acceleration=False):
        """
        Grid网格类的实例，
        wavelength_vacuum真空中波长，
        dist传输距离，
        refractive_index折射率，
        method衍射计算方法，AS为角谱，FFT-DI是基于FFT的直接积分法。paraxial是否进行近轴近似，仅在AS方法下有效
        gpu_acceleration是否使用cupy进行加速计算
        """
        xp = cp if gpu_acceleration else np  # 使用cupy进行GPU加速计算
        self.e_out = None
        self.dist = dist
        self.n = refractive_index  # 折射率
        self.wavelength = wavelength_vacuum / self.n
        self.k_prop = 2 * xp.pi / self.wavelength
        self.gpu_acceleration = gpu_acceleration
        self.method = method
        if self.method == "AS":
            """
            使用角谱理论
            """
            grid_d2_fft_x2 = xp.power(Grid.d2_fft_x, 2)
            grid_d2_fft_y2 = xp.power(Grid.d2_fft_y, 2)
            self.mat = xp.ones_like(grid_d2_fft_x2, dtype=complex)
            if paraxial:  # 使用近轴近似
                self.mat = xp.exp(1j * (self.k_prop * self.dist -
                                        xp.pi * self.wavelength * (grid_d2_fft_x2 + grid_d2_fft_y2) * self.dist))
            else:  # 不使用近轴近似
                condition = 1 / xp.power(self.wavelength, 2) - grid_d2_fft_x2 - grid_d2_fft_y2  # 根号下内容，判断空间频率是否大于波长平方倒数
                self.mat = xp.where(
                    condition > 0,
                    xp.exp(1j * 2 * xp.pi * xp.sqrt(condition.clip(min=0)) * self.dist),
                    xp.exp(-2 * xp.pi * xp.sqrt(xp.abs(condition.clip(max=0))) * self.dist)
                )
        elif self.method == "BL-AS":
            """
            带限角谱理论 
            Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far 
            and Near Fields
            避免角谱中由于传递函数的采样问题，产生的严重数值误差
            不考虑倏逝波           
            """
            grid_d2_fft_x2 = xp.power(Grid.d2_fft_x, 2)
            grid_d2_fft_y2 = xp.power(Grid.d2_fft_y, 2)
            self.mat = xp.ones_like(Grid.d2_fft_x, dtype=complex)
            condition = 1 / xp.power(self.wavelength, 2) - grid_d2_fft_x2 - grid_d2_fft_y2  # 根号下内容，判断空间频率是否大于波长平方倒数
            # self.mat = xp.where(
            #     condition > 0,
            #     xp.exp(1j * 2 * xp.pi * xp.sqrt(condition.clip(min=0)) * self.dist),
            #     0
            # )
            self.mat[condition > 0] = np.exp(
                1j * 2 * np.pi * np.sqrt(condition[condition > 0]) * self.dist)
            self.mat[condition <= 0] = 0  # 不考虑倏逝波
            fft_limit_sq = 1 / ((2 * Grid.step_fft * self.dist) ** 2 + 1) / self.wavelength ** 2  # x,y方向间距相等，公用一个极限条件即可
            condition1 = grid_d2_fft_x2 / fft_limit_sq + grid_d2_fft_y2 * self.wavelength ** 2 - 1
            condition2 = grid_d2_fft_x2 * self.wavelength ** 2 + grid_d2_fft_y2 / fft_limit_sq - 1
            self.mat[(condition1 > 0) | (condition2 > 0)] = 0  # 频带限制

        elif self.method == "FFT-DI":
            """
            基于FFT的瑞利-索末菲直接积分法
            Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld diffraction formula
            """
            dr_real = np.sqrt(Grid.step ** 2 + Grid.step ** 2)  # 实际采样间隔
            rmax = np.sqrt((np.max(Grid.axis) ** 2) + (np.max(Grid.axis) ** 2))
            dr_ideal = np.sqrt(self.wavelength ** 2 + rmax ** 2 + 2 *
                               self.wavelength * np.sqrt(rmax ** 2 + self.dist ** 2)) - rmax  # 理想采样间隔
            self.quality = dr_ideal / dr_real  # >1时采样准确
            if self.quality >= 1:
                print('Good result: factor {:2.2f}'.format(self.quality))
            else:
                print('Needs denser sampling: factor {:2.2f}'.format(self.quality))
            self.mat_DI = xp.zeros((2 * Grid.num_points - 1, 2 * Grid.num_points - 1), dtype=complex)
            # 生成一个(2N-1)×(2N-1)矩阵
            X_tmp_1D = xp.linspace(2 * Grid.axis[0], -2 * Grid.axis[0], 2 * Grid.num_points - 1)
            Y_tmp_1D = xp.linspace(2 * Grid.axis[0], -2 * Grid.axis[0], 2 * Grid.num_points - 1)
            X_tmp_2D, Y_tmp_2D = xp.meshgrid(X_tmp_1D, Y_tmp_1D)
            tmp_r = np.sqrt(X_tmp_2D ** 2 + Y_tmp_2D ** 2 + self.dist ** 2)
            self.mat_DI = 1 / (2 * np.pi) * np.exp(1j * self.k_prop * tmp_r) * self.dist / tmp_r ** 2 * (
                        1 / tmp_r - 1j * self.k_prop)  # 论文中式(7)
            self.mat_DI = self.mat_DI * Grid.step ** 2

    def prop(self, complex_amplitude):
        #  complex_amplitude输入复振幅
        xp = cp if self.gpu_acceleration else np  # 是否使用GPU加速

        if self.method == "AS" or self.method == "BL-AS":  # 使用角谱理论
            wave_fft = xp.fft.fftshift(xp.fft.fft2(complex_amplitude))
            # plt.matshow(xp.angle(complex_amplitude))
            # plt.show()
            wave_prop = wave_fft * self.mat
            self.e_out = xp.fft.ifft2(xp.fft.fftshift(wave_prop))
        elif self.method == "FFT-DI":  # 基于FFT的瑞利-索末菲直接积分法
            N = complex_amplitude.shape[0]
            Mat_U = xp.zeros((2 * N - 1, 2 * N - 1), dtype=complex)  # 左上角为复振幅的值
            if xp.mod(complex_amplitude.shape[0], 2) != 0 and complex_amplitude.shape[0] > 3:  # 使用辛普森法则
                print("Simpson's rule is used")
                B = xp.empty(N, dtype=int)  # 辛普森系数生成
                B[0] = 1
                B[-1] = 1
                B[1:-1:2] = 4
                B[2:-1:2] = 2
                W = xp.outer(B, B) / 9  # 外积B*B转置
                Mat_U[:N, :N] = complex_amplitude * W  # 论文中式(11) :N不包括N
            else:
                Mat_U[:N, :N] = complex_amplitude
            S = xp.fft.ifft2(xp.fft.fft2(Mat_U) * xp.fft.fft2(self.mat_DI))
            self.e_out = S[N-1:, N-1:]  # 右下角为输出复振幅结果
            # self.e_out = S[:N, :N]
        return self.e_out
