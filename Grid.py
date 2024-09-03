"""
建立网格类
By 周王哲
2023.7.22
"""
import numpy as np


class Grid:
    def __init__(self, num_points=256, length=1.0):
        #  num_points: 离散点数量,length: 实际长度，单位mm
        self.num_points = num_points
        self.length = length

        self.step = self.length / self.num_points  # 离散步长
        self.step_fft = 1 / self.length  # 频率步长
        self.half_num_points = self.num_points / 2

        self.vector = np.arange(1, self.num_points + 1)
        self.axis = -self.length / 2 + self.step / 2 + (self.vector - 1) * self.step  # 离散后实空间坐标轴
        self.axis_fft = -1 / (2 * self.step) + (self.vector - 1) / self.length  # 离散后频谱空间坐标轴

        self.d2_x, self.d2_y = np.meshgrid(self.axis, self.axis)  # 离散后的坐标矩阵
        self.d2_square = self.d2_x ** 2 + self.d2_y ** 2  # 离散后的坐标平方和
        self.d2_r = np.sqrt(self.d2_square)  # 离散后的极坐标半径

        self.d2_fft_x, self.d2_fft_y = np.meshgrid(self.axis_fft, self.axis_fft) # 离散后的频谱坐标矩阵
