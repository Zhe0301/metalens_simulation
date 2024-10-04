"""
超透镜版图绘制
周王哲
2024.9.30
使用旧的mirror和center函数，以适配旧版 Klayout 0.26.12
gdsfactory == 7.27.1
"""
import os
import h5py
import gdsfactory as gf
import numpy as np
from multiprocessing import Process
from tqdm import tqdm


# 单元结构圆柱,正方形排布方式
def binary2_cylinder_square(save_path, name, lens_radius, unit_period, unit_radius, mult_coef_a, diff_coef_m=1,
                            norm_radius=1):
    """
    用于生成绘图所需的x，y坐标，以及对于的圆柱半径
    :param save_path: 储存目录
    :param name: 文件名称
    :param lens_radius: 镜片尺寸
    :param unit_period: 单元结构周期，单位mm
    :param unit_radius: 单元结构半径，一维数组，相位从小到大，单位mm
    :param mult_coef_a: zemax二元面2多项式系数，一维数组
    :param diff_coef_m: zemax二元面2衍射系数，默认为1
    :param norm_radius: zemax二元面2归一化半径，默认为1
    :return:x_coordinate, y_coordinate, radius_list:一维数组，x，y坐标，对应的圆柱半径
    """
    # 圆对称，仅画1/4 区域
    x_number = int(np.round(lens_radius / unit_period))  # 四舍五入
    x_list = np.arange(x_number + 1) * unit_period + unit_period / 2
    x_coordinate = np.array([])
    y_coordinate = np.array([])
    for x in x_list:
        if (lens_radius ** 2 - x ** 2) < 0:
            y = 0
        else:
            y = np.sqrt(lens_radius ** 2 - x ** 2)
        y_number = int(np.round(y / unit_period))  # 四舍五入
        y_list = np.arange(y_number + 1) * unit_period + unit_period / 2
        x_coordinate = np.concatenate((x_coordinate, np.full(y_number + 1, x)))
        y_coordinate = np.concatenate((y_coordinate, y_list))
    phase = np.zeros_like(x_coordinate, dtype=float)
    for i, a_i in enumerate(mult_coef_a):  # enumerate组合为一个索引序列，同时列出数据和数据下标
        phase += diff_coef_m * a_i * (
                (x_coordinate ** 2 + y_coordinate ** 2) / norm_radius) ** (i + 1)
    phase = np.mod(phase, 2 * np.pi)
    d = len(unit_radius)  # 计算相位离散份数
    interval = 2 * np.pi / d  # 计算相位离散间隔
    phase_number = np.round(np.floor(phase / interval)).astype(int)  # 选择柱子，对比Lens种此处无1/2是由于此处不是相位值，而是柱子在数组中的序号
    radius_list = np.zeros_like(phase_number, dtype=float)
    for i in range(d):
        radius_list[phase_number == i] = unit_radius[i]
    with h5py.File(save_path + name + '_phase_number.h5', 'w') as f:
        dset = f.create_dataset('phase_number', data=phase_number, compression='gzip', compression_opts=9)
    with h5py.File(save_path + name + '_radius_list.h5', 'w') as f:
        dset = f.create_dataset('radius_list', data=phase_number, compression='gzip', compression_opts=9)
    with h5py.File(save_path + name + '_x_coordinate.h5', 'w') as f:
        dset = f.create_dataset('x_coordinate', data=phase_number, compression='gzip', compression_opts=9)
    with h5py.File(save_path + name + '_y_coordinate.h5', 'w') as f:
        dset = f.create_dataset('y_coordinate', data=phase_number, compression='gzip', compression_opts=9)
    return x_coordinate, y_coordinate, radius_list

    # 版图绘制


def draw_circle(x_coordinate, y_coordinate, radius_list, name, save_path, id=0):
    """
    生成GDS文件
    :param x_coordinate: 一维数组，x坐标，单位mm
    :param y_coordinate: 一维数组，y坐标，单位mm
    :param radius_list: 一维数组，圆柱半径，单位mm
    :param name: 文件及器件名称
    :param save_path: 储存路径
    :param id: 多线程时线程号
    :return: 无
    """
    c = gf.Component()
    c.name = str(id)
    for i in tqdm(range(len(x_coordinate)), desc="Task_{}".format(id)):
        circle = gf.components.circle(radius=radius_list[i] * 1e3,layer=(1, 0))
        unit = c << circle
        unit.center = [x_coordinate[i] * 1e3, y_coordinate[i] * 1e3]
    # print(save_path + r"temp_{}_{}.gds".format(name, id))
    c.write_gds(save_path + r"temp_{}_{}.gds".format(name, id)) # 不支持奇特的字符，例如×乘号，使用字母和常规符号



if __name__ == '__main__':
    # mult_coef_a = [-5.848595615954e2, 3.20546768416e1, -1.422681034594e1, 2.984845643491] # Lens1
    # lens_radius = 0.75
    # name = "lens1"
    mult_coef_a = [3.157248960338e3, -2.062812665413e3, 1.207061495482e4, -6.173754564586e4]  # Lens2
    lens_radius = 0.31
    name = "lens2"
    # mult_coef_a = [-1.845376080337e3, 6.935299381526e1, -7.928934283067, 1.711027879901] # Lens3
    # lens_radius = 1.2
    # name = "lens3"
    unit_radius = [80e-6, 96e-6, 103e-6, 109e-6, 113e-6, 117e-6, 128e-6, 137e-6]
    unit_period = 500e-6
    save_path = r"E:/Research/WavePropagation/metalens_simulation/Zoom_6x/20240930_actual_cylinder_1064/"
    x_coordinate, y_coordinate, radius_list = binary2_cylinder_square(save_path, name, lens_radius, unit_period,
                                                                      unit_radius, mult_coef_a)
    if not os.path.exists(save_path + r"temp_GDS/"):
        # 如果不存在则创建临时文件夹
        os.makedirs(save_path + r"temp_GDS/")
    # c = L.draw()
    # c.write_gds("p1.gds")
    """并行绘制版图"""
    CPU_core = 8  # 运行的线程数量
    component_max = 50000  # 最大绘制组件数量
    unit_number = len(x_coordinate)
    print(" {} unit structures need to be drawn".format(unit_number))
    if unit_number >= component_max * CPU_core:
        n = np.ceil(unit_number / component_max).astype(int)  # 需要总轮次
    else:
        n = CPU_core
    print("{} tasks are required".format(n))  # 分区
    x = np.array_split(x_coordinate, n)
    y = np.array_split(y_coordinate, n)
    r = np.array_split(radius_list, n)
    i = 0  # 计数器
    while i < n:
        j = 0  # 进程计数器
        processes = []
        while j < CPU_core:
            p = Process(target=draw_circle, args=(x[i], y[i], r[i], name, save_path + r"temp_GDS/", i))
            p.start()
            processes.append(p)
            i = i + 1
            j = j + 1
            if i >= n:
                break
        for p in processes:
            p.join()

    c = gf.Component()
    c.name = 'quarter_lens'
    # 合并不同分区的计算数据
    for i in tqdm(range(n), desc="Merging"):
        s = gf.read.import_gds(save_path + r"temp_GDS/temp_{}_{}.gds".format(name, i))
        c.add_ref(s)
    # 镜像
    s_copy = gf.Component()
    s_copy.add_ref(c)
    s_copy.name = "quarter_lens_copy"
    m_y = s_copy.mirror((0, 0), (0, 1))
    m_y.name = "y_mirror"
    s_y = gf.Component()
    s_y.add_ref(s_copy)
    s_y.add_ref(m_y)
    s_y.name = "half_lens"
    m_x = s_y.mirror((0, 0), (1, 0))
    m_x.name = "x_mirror"
    c_all = gf.Component()
    c_all.name = "all"
    c_all.add_ref(m_y)
    c_all.add_ref(m_x)
    c_all.add_ref(s_copy)
    c_all.write_gds("p_m.gds")
    c_all.write_gds(save_path + "{}.gds".format(name))
