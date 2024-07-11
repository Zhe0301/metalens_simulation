# 提取放大矩阵中心部分
import numpy as np
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


def calculate_enclosed_energy_ratio(r, intensity):
    """
    计算不同半径下圈入能量占总能量的比
    r: 径向坐标数组 mm
    intensity: 光场强度分布数组 w/mm
    return: 圈入能量占总能量的比数组
    """
    # 计算总能量
    total_energy = np.trapz(intensity * 2 * np.pi * r, r)

    # 计算不同半径下圈入的能量
    enclosed_energy = np.zeros_like(r)
    for i in range(len(r)):
        enclosed_energy[i] = np.trapz(intensity[:i + 1] * 2 * np.pi * r[:i + 1], r[:i + 1])

    # 计算不同半径下圈入能量占总能量的比
    energy_ratio = enclosed_energy / total_energy

    return energy_ratio


def calculate_fwhm(r, intensity):
    """
    计算光场强度达到半高宽 (FWHM) 的半径位置
    r: 径向坐标数组 mm
    intensity: 光场强度分布数组 w/mm
    :return: 半高宽 (FWHM) 对应的半径位置
    """
    # 找到光场强度的最大值和一半最大值的位置
    I_max = np.max(intensity)
    I_half_max = I_max / 2

    # 使用插值法找到光场强度为一半最大值的位置
    f_interp = interp1d(intensity, r, kind='linear', fill_value='extrapolate')

    try:
        r_half_max1 = f_interp(I_half_max)
        FWHM = 2 * r_half_max1
    except ValueError:
        FWHM = None

    return FWHM



