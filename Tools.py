# 提取放大矩阵中心部分
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def extract_center(matrix, center_fraction):
    """
    从矩阵中提取中心部分。

    参数:
    - matrix: 输入矩阵
    - center_fraction: 中心部分的比例（0 到 1 之间），表示提取的部分大小

    返回:
    - 提取的中心部分矩阵
    """
    rows, cols = matrix.shape
    center_rows = int(rows * center_fraction)
    center_cols = int(cols * center_fraction)

    row_start = (rows - center_rows) // 2
    row_end = row_start + center_rows
    col_start = (cols - center_cols) // 2
    col_end = col_start + center_cols

    return matrix[row_start:row_end, col_start:col_end]


def calculate_enclosed_energy_ratio(r, intensity, gpu_acceleration=False):
    """
    计算不同半径下圈入能量占总能量的比,仅适用于中心对称场
    r: 径向坐标数组 mm
    intensity: 一维径向光场强度分布数组 w/mm
    return: 圈入能量占总能量的比数组
    """
    # 计算总能量
    xp = cp if gpu_acceleration else np  # 使用cupy进行GPU加速计算
    total_energy = xp.trapz(intensity * 2 * np.pi * r, r)

    # 计算不同半径下圈入的能量
    enclosed_energy = xp.zeros_like(r)
    for i in range(len(r)):
        if i == 0:
            if r[i] == 0:
                enclosed_energy[i] = 0
            else:
                enclosed_energy[i] = xp.pi * r[i] ** 2 * intensity[i]  #使用相同值进行外插，也即假设平顶分布
        else:
            enclosed_energy[i] = xp.trapz(intensity[:i + 1] * 2 * xp.pi * r[:i + 1], r[:i + 1])

    # 计算不同半径下圈入能量占总能量的比
    energy_ratio = enclosed_energy / total_energy

    return energy_ratio


def calculate_fwhm(r, intensity, gpu_acceleration=False):
    """
    计算光场强度达到半高宽 (FWHM) 的半径位置,仅适用于中心对称场
    r: 径向坐标数组 mm
    intensity: 光场强度分布数组 w/mm
    :return: 半高宽 (FWHM) 对应的半径位置
    """
    xp = cp if gpu_acceleration else np
    # 找到光场强度的最大值和一半最大值的位置
    I_max = xp.max(intensity)
    I_half_max = I_max / 2

    #  使用弦截法寻根
    intensity = intensity - I_half_max
    FWHM = r[0]
    FWHM0 = r[0]  # 启动参数
    FWHM1 = r[1]
    tolerance = 1e-7  # 容限
    i = 0
    while xp.abs(xp.interp(FWHM, r, intensity)) >= tolerance:
        if FWHM1 - FWHM0 == 0:
            FWHM1 = FWHM0 + (r[1] - r[0])
        FWHM = FWHM1 - xp.interp(FWHM1, r, intensity) * (FWHM1 - FWHM0) / (
                xp.interp(FWHM1, r, intensity) - xp.interp(FWHM0, r, intensity))
        FWHM0 = FWHM1
        FWHM1 = FWHM
        if i > 3000:
            print("More than 3000 iterations.FWHM calculation failed")
            break
        i = i + 1

    FWHM = 2 * FWHM
    return FWHM


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
    mtf_x = mtf_x[N//2:]
    mtf_x = mtf_x/xp.max(mtf_x) # 归一化
    mtf_y = mtf_y[N//2:]
    mtf_y = mtf_y / xp.max(mtf_y)
    if gpu_acceleration:
        mtf_x = cp.asnumpy(mtf_x)
        mtf_y = cp.asnumpy(mtf_y)
        psf = cp.asnumpy(psf)
    return mtf_x, mtf_y, psf
