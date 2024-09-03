"""
超透镜版图绘制
周王哲
2024.9.3
"""
import gdsfactory as gf
import numpy as np
from tqdm import tqdm


class LayerBinary2:
    """
    二元面版图绘制
    """

    def __init__(self, lens_radius, unit_period, unit_radius, mult_coef_a, diff_coef_m=1, norm_radius=1):
        """
        name: 版图名称
        lens_radius: 镜片半径 单位为mm
        unit_period: 单元结构周期 单位为mm
        unit_radius: 单元结构半径 单位为mm 数组 从低相位到高相位排布
        mult_coef_a: 二元面多项式系数 数组
        diff_coef_m： 二元面�系数
        """
        self.y_coordinate = np.array([],dtype=int)
        self.x_coordinate = np.array([],dtype=int)
        self.phase_number = np.array([],dtype=int)
        self.lens_radius = lens_radius
        self.unit_period = unit_period
        self.unit_radius = unit_radius
        self.mult_coef_a = mult_coef_a
        self.diff_coef_m = diff_coef_m
        self.norm_radius = norm_radius

    # 单元结构圆柱,正方形排布方式
    def cylinder_square(self):
        x_number = int(2 * self.lens_radius // self.unit_period)
        if x_number % 2 == 0:
            x_number = x_number + 1
        x_list = np.linspace(-(x_number - 1) / 2, (x_number - 1) / 2, x_number, endpoint=True, dtype=int)
        for x in x_list:
            y = np.sqrt(self.lens_radius ** 2 - (x * self.unit_period) ** 2)
            y_number = int(2 * y // self.unit_period)
            if y_number % 2 == 0:
                y_number = y_number + 1
            y_list = np.linspace(-(y_number - 1) / 2, (y_number - 1) / 2, y_number, endpoint=True, dtype=int)
            self.x_coordinate = np.concatenate((self.x_coordinate, np.full(y_number, x)))
            self.y_coordinate = np.concatenate((self.y_coordinate, y_list))
        phase = np.zeros_like(self.x_coordinate, dtype=float)
        for i, a_i in enumerate(self.mult_coef_a):  # enumerate组合为一个索引序列，同时列出数据和数据下标
            phase += self.diff_coef_m * a_i * (
                        (self.x_coordinate ** 2 + self.y_coordinate ** 2) / self.norm_radius) ** (2 * (i + 1))
            phase = np.mod(phase, 2 * np.pi)
        d = len(self.unit_radius)  # 计算相位离散份数
        interval = 2 * np.pi / d  # 计算相位离散间隔
        self.phase_number = np.round(np.floor(phase / interval)).astype(int)  # 选择柱子，对比Lens种此处无1/2是由于此处不是相位值，而是柱子在数组中的序号

    # 版图绘制
    def draw(self):
        c = gf.Component()
        for i in tqdm(range(len(self.x_coordinate))):
            cylinder = gf.components.circle(radius=self.unit_radius[self.phase_number[i]] * 1e3, angle_resolution=2.5,
                                            layer=(1, 0))
            unit = c << cylinder
            unit.dcenter = [self.x_coordinate[i] * 1e3, self.y_coordinate[i] * 1e3]
        return c


if __name__ == '__main__':
    mult_coef_a = [-1.101022155962093e3, 1.561111683794811e2, -4.370083784673994e2, 5.311254043289745,
                   1.319833645746903e4, -4.438157186306786e4]
    unit_radius = [48e-6, 56e-6, 59e-6, 61e-6, 63e-6, 64e-6, 67e-6, 76e-6]
    L = LayerBinary2(0.35,347e-6,unit_radius,mult_coef_a)
    L.cylinder_square()
    c = L.draw()
    c.write_gds("demo.gds")
