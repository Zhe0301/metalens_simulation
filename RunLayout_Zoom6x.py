"""
超透镜版图绘制
周王哲
2024.10.17
绘制6x变焦系统得版图，圆柱为八边形
使用旧的mirror和center函数，以适配旧版 Klayout 0.26.12
gdsfactory == 7.27.1
"""
from Layout_old import *

import gdsfactory as gf
import numpy as np
from multiprocessing import Process
from tqdm import tqdm


if __name__ == '__main__':
    mult_coef_a = [-5.848595615954e2, 3.20546768416e1, -1.422681034594e1, 2.984845643491]  # Lens1
    lens_radius = 0.75
    name = "lens1_old_octagon"
    # mult_coef_a = [3.157248960338e3, -2.062812665413e3, 1.207061495482e4, -6.173754564586e4]  # Lens2
    # lens_radius = 0.31
    # name = "lens2_old_octagon"
    # mult_coef_a = [-1.845376080337e3, 6.935299381526e1, -7.928934283067, 1.711027879901] # Lens3
    # lens_radius = 1.2
    # name = "lens3_old_octagon"

    unit_radius = [80e-6, 99e-6, 108e-6, 113e-6, 118e-6, 123e-6, 134e-6, 143e-6]
    unit_period = 500e-6
    save_path = r"E:/Research/WavePropagation/metalens_simulation/Zoom_6x/20241018_actual_cylinder_1064/"
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
            p = Process(target=draw_polygon, args=(x[i], y[i], r[i], 8, name, save_path + r"temp_GDS/", i))
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
    c_all.write_gds(save_path + "{}.gds".format(name))
