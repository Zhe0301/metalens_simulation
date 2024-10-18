"""
超透镜版图绘制
周王哲
2024.10.17
使用旧的mirror和center函数，以适配旧版 Klayout 0.26.12
gdsfactory == 7.27.1
"""
from Layout_old import *

import gdsfactory as gf
import numpy as np
from multiprocessing import Process
from tqdm import tqdm