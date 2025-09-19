# src/causal_discovery/wrappers/base.py
# 作者: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# 时间: 2025-09-19 16:00
# 描述： 因果发现算法基类 (Base Class for Causal Discovery Algorithms)

from typing import Union, Dict, Tuple

import numpy as np
import pandas as pd

from src.core import GlobalState


class BaseModel:
    def __init__(self, state: GlobalState, params: Dict):
        self.state = state
        # state.algorithm.gpu_available 可以获取GPU可用性
        self._params = {}
        self._params.update(params)

    def get_params(self):
        """作用：返回当前算法所有可用参数的字典（含默认+外部传入）。"""
        return self._params.copy()

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """作用：在数据上运行因果发现，返回邻接矩阵、信息字典和底层图对象/结果。"""
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()
