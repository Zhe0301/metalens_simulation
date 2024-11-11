# 提取放大矩阵中心部分
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt


def extract_spot(intensity, G, center_x, center_y, max_radius=0, interp_factor=2, gpu_acceleration=False):
    """
    根据中心位置提取光斑，并适当插值
    参数:
    I: 光强矩阵
    G: 坐标网格对象，包含轴信息
    center_x, center_y: 圈入的中心位置
    max_radius: 圈入最大半径
    interp_factor: 插值因子
    返回值:
    x_interp, y_interp: 插值后的 x 和 y 轴坐标
    I_interp: 插值后的光强矩阵
    """
    if gpu_acceleration:
        xp = cp
        intensity = cp.asarray(intensity)
        axis = cp.asarray(G.axis)
        from cupyx.scipy.interpolate import RegularGridInterpolator
    else:
        xp = np
        axis = G.axis
        from scipy.interpolate import RegularGridInterpolator
    # 计算x和y坐标的最大范围
    x_max_dist = xp.max(xp.abs(axis - center_x))
    y_max_dist = xp.max(xp.abs(axis - center_y))
    radius_limit = xp.minimum(x_max_dist, y_max_dist)  # 最大有效半径

    # 检查最大半径的有效性
    if max_radius > radius_limit:
        print(f"invalid max_radius, radius_limit={radius_limit:.4f} mm")
        max_radius = radius_limit

    # 筛选需要计算的区域
    x_idx = (axis >= (center_x - max_radius - G.step)) & (axis <= (center_x + max_radius + G.step))
    y_idx = (axis >= (center_y - max_radius - G.step)) & (axis <= (center_y + max_radius + G.step))
    x_c = axis[x_idx]
    y_c = axis[y_idx]
    I_c = intensity[xp.ix_(y_idx, x_idx)]

    # 创建插值对象
    f = RegularGridInterpolator((x_c, y_c), I_c, method='linear')

    # 定义插值后的坐标轴
    x_interp = xp.linspace(xp.max(x_c), xp.min(y_c), xp.size(y_c) * interp_factor)
    y_interp = xp.linspace(xp.max(x_c), xp.min(y_c), xp.size(y_c) * interp_factor)

    # 生成网格点并进行插值
    X_interp, Y_interp = xp.meshgrid(x_interp, y_interp)
    I_interp = f((X_interp, Y_interp))
    if gpu_acceleration:
        x_interp = xp.asnumpy(x_interp)
        y_interp = xp.asnumpy(y_interp)
        I_interp = xp.asnumpy(I_interp)

    return x_interp, y_interp, I_interp


def calculate_enclosed_energy_ratio(intensity, G, center_x, center_y, max_radius=0, total_energy=0, min_points=256,
                                    gpu_acceleration=False):
    """
    计算不同半径下圈入能量占总能量的比,可用于非中心对称的光场，无法使用GPU加速

    param: intensity: 二维光强分布矩阵 w/mm
    param: G: 网格类
    return: center_x, center_y:  圈入圆的圆心坐标
    最后修改日期：2024年11月6日
    作者：周王哲
    """
    if gpu_acceleration:
        xp = cp
        from cupyx.scipy.interpolate import RegularGridInterpolator
        intensity = cp.asarray(intensity)
        axis = cp.asarray(G.axis)
    else:
        xp = np
        from scipy.interpolate import RegularGridInterpolator
        axis = G.axis
    # 计算总能量
    if max_radius == 0:
        max_radius = G.length / 2
    if total_energy == 0:
        total_energy = xp.trapz(xp.trapz(intensity, axis, axis=1), axis)
    # 计算x和y坐标的最大范围
    x_max_dist = xp.max(xp.abs(axis - center_x))
    y_max_dist = xp.max(xp.abs(axis - center_y))
    radius_limit = min(x_max_dist, y_max_dist)  # 最大有效半径
    if max_radius > radius_limit:
        print(f"invalid max_radius, radius_limit={radius_limit:.4f} mm")
        max_radius = radius_limit

    # 筛选需要计算的区域
    x_idx = (axis > (center_x - radius_limit - G.step)) & (axis < (center_x + max_radius + G.step))
    y_idx = (axis > (center_y - radius_limit - G.step)) & (axis < (center_y + max_radius + G.step))
    x_c = axis[x_idx]
    y_c = axis[y_idx]
    I_c = intensity[xp.ix_(y_idx, x_idx)]  # np.ix_两个一维索引转换为二维

    # 创建插值对象，用于计算非整数位置的光强
    f = RegularGridInterpolator((x_c, y_c), I_c, method='linear')

    # 初始化半径和能量比数组
    radius = xp.arange(0, max_radius + G.step, G.step)
    energy_ratio = xp.zeros(len(radius))

    # 计算每个半径下的圈入能量比
    for i, radii in enumerate(radius):
        if i == 0:
            energy_ratio[i] = 0
            continue

        if i > min_points / 2:
            interp_points = i
        else:
            interp_points = min_points

        x_interp = xp.linspace(center_x - radii, center_x + radii, interp_points)
        y_interp = xp.linspace(center_y - radii, center_y + radii, interp_points)
        X_interp, Y_interp = xp.meshgrid(x_interp, y_interp)
        distances = xp.sqrt((X_interp - center_x) ** 2 + (Y_interp - center_y) ** 2)

        I_c_interp = f((X_interp, Y_interp))
        I_c_interp[distances > radii] = 0
        encircled_energy = xp.trapz(xp.trapz(I_c_interp, x_interp, axis=1), y_interp)
        energy_ratio[i] = encircled_energy / total_energy
        if gpu_acceleration:
            energy_ratio = cp.asnumpy(energy_ratio)
            radius = cp.asnumpy(radius)
    return radius, energy_ratio


def calculate_fwhm(intensity, G, gpu_acceleration=False, centrosymmetry=False):
    """
    计算光强最强位置和半高全宽FWHM,可用于非重新对称光斑，需注意输出的不是FWHM而是各方向达到半高时与中心位置的差
    param: intensity: 二维光场强度分布矩阵 w/mm
    param: G: 网格类
    param: centrosymmetry: 是否中心对称
    :return: center_x, center_y: 最强光强位置
    :return: r_x_p, r_x_n, r_y_p, r_y_n: x,y正负方向半高全宽半径
    """
    if gpu_acceleration:
        xp = cp
        intensity = cp.asarray(intensity)
        d2_x = cp.asarray(G.d2_x)
        d2_y = cp.asarray(G.d2_y)
    else:
        xp = np
        d2_x = G.d2_x
        d2_y = G.d2_y

    # 找到最大光强及其位置
    max_idx = xp.argmax(intensity)
    max_row, max_col = xp.unravel_index(max_idx, intensity.shape)
    max_val = intensity[max_row, max_col]

    # 检查是否存在多个最大值
    max_all_idx = xp.where(abs(intensity - max_val) <= max_val * 1e-5)
    if xp.size(max_all_idx[0]) > 1:
        center_x = xp.mean(d2_x[max_all_idx])
        center_y = xp.mean(d2_y[max_all_idx])
    else:
        center_x = d2_x[max_row, max_col]
        center_y = d2_y[max_row, max_col]

    max_val /= 2  # 光强的一半

    # 如果中心对称，直接计算正方向的FWHM，所有方向相等
    if centrosymmetry and center_x == center_y:
        I_x_p = intensity[max_row, max_col:]
        x_p = d2_x[max_row, max_col:]

        position = xp.where(I_x_p < max_val)[0][0]
        r_x_p = xp.interp(max_val, I_x_p[:position + 1][::-1], x_p[:position + 1][::-1]) - center_x  # interp中x必须是单调递增的

        # 中心对称性假设，其他方向相等
        r_x_n = r_x_p
        r_y_p = r_x_p
        r_y_n = r_x_p

    else:
        # 非对称情况下，计算x和y方向的FWHM
        I_x_p = intensity[max_row, max_col:]
        x_p = d2_x[max_row, max_col:]
        I_x_n = xp.flip(intensity[max_row, :max_col + 1])
        x_n = xp.flip(d2_x[max_row, :max_col + 1])
        I_y_p = intensity[max_row:, max_col]
        y_p = d2_y[max_row:, max_col]
        I_y_n = xp.flip(intensity[:max_row + 1, max_col])
        y_n = xp.flip(d2_y[:max_row + 1, max_col])

        # x正方向
        position = xp.where(I_x_p < max_val)[0][0]
        r_x_p = xp.interp(max_val, I_x_p[:position + 1], x_p[:position + 1]) - center_x

        # x负方向
        position = xp.where(I_x_n < max_val)[0][0]
        r_x_n = center_x - xp.interp(max_val, I_x_n[:position + 1], x_n[:position + 1])

        # y正方向
        position = xp.where(I_y_p < max_val)[0][0]
        r_y_p = xp.interp(max_val, I_y_p[:position + 1], y_p[:position + 1]) - center_y

        # y负方向
        position = xp.where(I_y_n < max_val)[0][0]
        r_y_n = center_y - xp.interp(max_val, I_y_n[:position + 1], y_n[:position + 1])
    if gpu_acceleration:
        center_x = cp.asnumpy(center_x).item()
        center_y = cp.asnumpy(center_y).item()
        r_x_p = cp.asnumpy(r_x_p).item()
        r_x_n = cp.asnumpy(r_x_n).item()
        r_y_p = cp.asnumpy(r_y_p).item()
        r_y_n = cp.asnumpy(r_y_n).item()
    return center_x, center_y, r_x_p, r_x_n, r_y_p, r_y_n


"""根据psf推算mtf"""


def calculate_mtf(psf, step, zero_coef=1e5, gpu_acceleration=False):
    """
    计算MTF
    :param:psf: 点扩散函数
    :param:step: 步长 mm
    :param:zero_coef: 置零系数，认为小于最大值/zero_coef的值视为0
    :param:gpu_acceleration: 是否使用GPU加速计算
    :return: x,y方向调制传递函数，置零处理后的psf
    """
    xp = cp if gpu_acceleration else np
    if gpu_acceleration:
        psf = cp.asarray(psf)
    psf[psf < xp.max(psf) / zero_coef] = 0
    lsf_x = xp.trapz(psf, dx=step, axis=0)  #对y积分
    lsf_y = xp.trapz(psf, dx=step, axis=1)  # 对x积分
    mtf_x = xp.abs(xp.fft.fftshift(xp.fft.fft(lsf_x)))
    mtf_y = xp.abs(xp.fft.fftshift(xp.fft.fft(lsf_y)))
    N = len(mtf_x)
    mtf_x = mtf_x[N // 2:]
    mtf_x = mtf_x / xp.max(mtf_x)  # 归一化
    mtf_y = mtf_y[N // 2:]
    mtf_y = mtf_y / xp.max(mtf_y)
    if gpu_acceleration:
        mtf_x = cp.asnumpy(mtf_x)
        mtf_y = cp.asnumpy(mtf_y)
        psf = cp.asnumpy(psf)
    return mtf_x, mtf_y, psf
