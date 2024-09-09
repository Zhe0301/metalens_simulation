"""
超透镜版图绘制
周王哲
2024.9.3
"""
import os

import gdsfactory as gf
import numpy as np
from multiprocessing import Process
from tqdm import tqdm



# 单元结构圆柱,正方形排布方式
def binary2_cylinder_square(save_path, name, lens_radius, unit_period, unit_radius, mult_coef_a, diff_coef_m=1, norm_radius=1):
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
    # 圆对称，仅画1/4区域
    x_number = int(np.round(lens_radius / unit_period)) # 四舍五入
    x_list = np.arange(x_number + 1) * unit_period + unit_period/2
    x_coordinate = np.array([])
    y_coordinate = np.array([])
    for x in x_list:
        if (lens_radius ** 2 - x ** 2) <0:
            y = 0
        else:
            y = np.sqrt(lens_radius ** 2 - x ** 2)
        y_number = int(np.round(y / unit_period)) # 四舍五入
        y_list = np.arange(y_number + 1) * unit_period + unit_period/2
        x_coordinate = np.concatenate((x_coordinate, np.full(y_number+1, x)))
        y_coordinate = np.concatenate((y_coordinate, y_list))
    phase = np.zeros_like(x_coordinate, dtype=float)
    for i, a_i in enumerate(mult_coef_a):  # enumerate组合为一个索引序列，同时列出数据和数据下标
        phase += diff_coef_m * a_i * (
                    (x_coordinate ** 2 + y_coordinate ** 2) / norm_radius) ** (i + 1)
    phase = np.mod(phase, 2 * np.pi)
    d = len(unit_radius)  # 计算相位离散份数
    interval = 2 * np.pi / d  # 计算相位离散间隔
    phase_number = np.round(np.floor(phase / interval)).astype(int)  # 选择柱子，对比Lens种此处无1/2是由于此处不是相位值，而是柱子在数组中的序号
    radius_list = np.zeros_like(phase_number,dtype=float)
    for i in range(d):
        radius_list[phase_number==i] = unit_radius[i]

    np.save(save_path + name + "_phase_number.npy",phase_number)
    np.save(save_path + name + "_radius_list.npy", radius_list)
    np.save(save_path + name + "_x_coordinate.npy", x_coordinate)
    np.save(save_path + name + "_y_coordinate.npy", y_coordinate)
    return x_coordinate, y_coordinate, radius_list

    # 版图绘制

def draw_circle(x_coordinate,y_coordinate,radius_list,name,save_path,id=0):
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
    c.name = id
    for i in tqdm(range(len(x_coordinate)),desc="Task_{}".format(id)):
        circle = gf.components.circle(radius_list[i] * 1e3,
                                            layer=(1, 0))
        unit = c << circle
        unit.dcenter = [x_coordinate[i] * 1e3,y_coordinate[i] * 1e3]
    c.write_gds(save_path+"temp_{}_{}.gds".format(name,id))



if __name__ == '__main__':
    mult_coef_a = [-1.134335348014e3, 1.997482968732e2, -7.744689756243e2, 2.350435931257e3, 3.245672141492e3, -2.507104417945e4] # Lens1
    lens_radius = 0.375
    name = "lens1"
    # mult_coef_a = [6.658448719319e3, -1.180564189085e4, 3.123599533268e5,-8.915821224249e6, 1.353360477873e8, -8.261794056944e8] # Lens2
    # lens_radius = 0.25
    # name = "lens2"
    # mult_coef_a = [-3.655457426687e3, 2.169008952537e2, -1.402136806599e2, 2.517459986023e2, -2.498340816648e2, 9.683857736021e1] # Lens3
    # lens_radius = 0.95
    # name = "lens3"
    unit_radius = [48e-6, 56e-6, 59e-6, 61e-6, 63e-6, 64e-6, 67e-6, 76e-6]
    save_path = r"E:/Research/WavePropagation/metalens_simulation/Zoom_6×/20240904_discrete_8/"
    x_coordinate, y_coordinate, radius_list = binary2_cylinder_square(save_path,name,lens_radius,347e-6,unit_radius,mult_coef_a)
    if not os.path.exists(save_path+r"temp_GDS/"):
        # 如果不存在则创建临时文件夹
        os.makedirs(save_path+r"temp_GDS/")
    # c = L.draw()
    # c.write_gds("p1.gds")
    """并行绘制版图"""
    CPU_core = 24 # 运行的线程数量
    component_max = 50000 # 最大绘制组件数量
    unit_number = len(x_coordinate)
    print(" {} unit structures need to be drawn".format(unit_number))
    if unit_number >= component_max*CPU_core:
        n = np.ceil(unit_number/component_max).astype(int) # 需要总轮次
    else:
        n = CPU_core
    print("{} tasks are required".format(n) )
    x = np.array_split(x_coordinate,n)
    y = np.array_split(y_coordinate,n)
    r = np.array_split(radius_list, n)
    i = 0 # 计数器
    while i < n:
        j = 0 # 进程计数器
        processes = []
        while j < CPU_core:
            p = Process(target=draw_circle,args=(x[i],y[i],r[i],name,save_path+r"temp_GDS/",i))
            p.start()
            processes.append(p)
            i = i + 1
            j = j + 1
            if i >= n:
                break
        for p in processes:
            p.join()

    c = gf.Component()
    c.name = 'all'
    # 合并不同核心的计算数据
    for i in tqdm(range(n),desc="Merging"):
        s = gf.read.import_gds(save_path+r"temp_GDS/temp_{}_{}.gds".format(name,i))
        c.add_ref(s)
    # 镜像
    m = c.dup()
    m.name = "y_mirror"
    m = c << m
    m.dmirror(p1=gf.kdb.DPoint(0, 0), p2=gf.kdb.DPoint(0, 1))  # y轴
    m = c.dup()
    m.name = "x_mirror"
    m = c << m
    m.dmirror(p1=gf.kdb.DPoint(0, 0), p2=gf.kdb.DPoint(1, 0))  # x轴
    c.write_gds(save_path+"{}.gds".format(name))


