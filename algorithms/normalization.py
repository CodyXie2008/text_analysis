# -*- coding: utf-8 -*-
"""
归一化算法模块

包含常用的数据归一化和标准化算法
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseNormalizer(ABC):
    """归一化算法基类"""
    
    def __init__(self, feature_range: tuple = (0, 1)):
        """
        初始化归一化器
        
        Args:
            feature_range: 归一化后的数值范围，默认(0, 1)
        """
        self.feature_range = feature_range
        self.is_fitted = False
        self._params = {}
    
    @abstractmethod
    def fit(self, data: Union[np.ndarray, List, pd.Series]) -> 'BaseNormalizer':
        """拟合归一化器，计算归一化参数"""
        pass
    
    @abstractmethod
    def transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """应用归一化变换"""
        pass
    
    def fit_transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """拟合并变换数据"""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """逆变换，将归一化数据还原"""
        if not self.is_fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")
        return self._inverse_transform(data)
    
    @abstractmethod
    def _inverse_transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """逆变换的具体实现"""
        pass
    
    def _check_data(self, data: Union[np.ndarray, List, pd.Series, pd.DataFrame]) -> np.ndarray:
        """检查并转换输入数据格式"""
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, pd.DataFrame):
            data = data.values
        elif not isinstance(data, np.ndarray):
            raise TypeError("输入数据必须是numpy数组、列表、pandas Series或DataFrame")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return data


class MinMaxNormalizer(BaseNormalizer):
    """
    Min-Max归一化算法
    
    将数据线性变换到指定范围 [min, max]
    公式: X_norm = (X - X_min) / (X_max - X_min) * (max - min) + min
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        super().__init__(feature_range)
        self.min_vals = None
        self.max_vals = None
        self.data_min = None
        self.data_max = None
    
    def fit(self, data: Union[np.ndarray, List, pd.Series]) -> 'MinMaxNormalizer':
        """
        拟合归一化器
        
        Args:
            data: 输入数据
            
        Returns:
            self: 返回自身实例
        """
        data = self._check_data(data)
        
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
        
        # 避免除零错误
        self.data_range = self.data_max - self.data_min
        self.data_range[self.data_range == 0] = 1
        
        self.min_vals = self.feature_range[0]
        self.max_vals = self.feature_range[1]
        
        self.is_fitted = True
        self._params = {
            'data_min': self.data_min,
            'data_max': self.data_max,
            'data_range': self.data_range,
            'feature_range': self.feature_range
        }
        
        logger.info(f"MinMax归一化器拟合完成，数据范围: [{self.data_min}, {self.data_max}]")
        return self
    
    def transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """
        应用Min-Max归一化
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 归一化后的数据
        """
        if not self.is_fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")
        
        data = self._check_data(data)
        
        # 应用归一化公式
        normalized = (data - self.data_min) / self.data_range
        normalized = normalized * (self.max_vals - self.min_vals) + self.min_vals
        
        return normalized
    
    def _inverse_transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """逆变换"""
        data = self._check_data(data)
        
        # 逆变换公式
        normalized = (data - self.min_vals) / (self.max_vals - self.min_vals)
        original = normalized * self.data_range + self.data_min
        
        return original


class ZScoreNormalizer(BaseNormalizer):
    """
    Z-Score标准化算法
    
    将数据标准化为均值为0，标准差为1的分布
    公式: Z = (X - μ) / σ
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        super().__init__(feature_range)
        self.mean_vals = None
        self.std_vals = None
    
    def fit(self, data: Union[np.ndarray, List, pd.Series]) -> 'ZScoreNormalizer':
        """
        拟合标准化器
        
        Args:
            data: 输入数据
            
        Returns:
            self: 返回自身实例
        """
        data = self._check_data(data)
        
        self.mean_vals = np.mean(data, axis=0)
        self.std_vals = np.std(data, axis=0)
        
        # 避免除零错误
        self.std_vals[self.std_vals == 0] = 1
        
        self.is_fitted = True
        self._params = {
            'mean': self.mean_vals,
            'std': self.std_vals
        }
        
        logger.info(f"Z-Score标准化器拟合完成，均值: {self.mean_vals}, 标准差: {self.std_vals}")
        return self
    
    def transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """
        应用Z-Score标准化
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 标准化后的数据
        """
        if not self.is_fitted:
            raise ValueError("标准化器尚未拟合，请先调用fit()方法")
        
        data = self._check_data(data)
        
        # 应用标准化公式
        standardized = (data - self.mean_vals) / self.std_vals
        
        return standardized
    
    def _inverse_transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """逆变换"""
        data = self._check_data(data)
        
        # 逆变换公式
        original = data * self.std_vals + self.mean_vals
        
        return original


class RobustScaler(BaseNormalizer):
    """
    稳健缩放算法
    
    使用四分位数进行缩放，对异常值不敏感
    公式: X_scaled = (X - Q2) / (Q3 - Q1)
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        super().__init__(feature_range)
        self.median_vals = None
        self.q1_vals = None
        self.q3_vals = None
        self.iqr_vals = None
    
    def fit(self, data: Union[np.ndarray, List, pd.Series]) -> 'RobustScaler':
        """
        拟合稳健缩放器
        
        Args:
            data: 输入数据
            
        Returns:
            self: 返回自身实例
        """
        data = self._check_data(data)
        
        self.median_vals = np.median(data, axis=0)
        self.q1_vals = np.percentile(data, 25, axis=0)
        self.q3_vals = np.percentile(data, 75, axis=0)
        self.iqr_vals = self.q3_vals - self.q1_vals
        
        # 避免除零错误
        self.iqr_vals[self.iqr_vals == 0] = 1
        
        self.is_fitted = True
        self._params = {
            'median': self.median_vals,
            'q1': self.q1_vals,
            'q3': self.q3_vals,
            'iqr': self.iqr_vals
        }
        
        logger.info(f"稳健缩放器拟合完成，中位数: {self.median_vals}, IQR: {self.iqr_vals}")
        return self
    
    def transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """
        应用稳健缩放
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 缩放后的数据
        """
        if not self.is_fitted:
            raise ValueError("缩放器尚未拟合，请先调用fit()方法")
        
        data = self._check_data(data)
        
        # 应用稳健缩放公式
        scaled = (data - self.median_vals) / self.iqr_vals
        
        return scaled
    
    def _inverse_transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """逆变换"""
        data = self._check_data(data)
        
        # 逆变换公式
        original = data * self.iqr_vals + self.median_vals
        
        return original


class DecimalScaler(BaseNormalizer):
    """
    小数缩放算法
    
    将数据缩放到指定的小数位数
    公式: X_scaled = X / 10^n
    """
    
    def __init__(self, n_decimals: int = 2):
        """
        初始化小数缩放器
        
        Args:
            n_decimals: 小数位数，默认2位
        """
        super().__init__()
        self.n_decimals = n_decimals
        self.scale_factor = 10 ** n_decimals
    
    def fit(self, data: Union[np.ndarray, List, pd.Series]) -> 'DecimalScaler':
        """
        拟合小数缩放器
        
        Args:
            data: 输入数据
            
        Returns:
            self: 返回自身实例
        """
        data = self._check_data(data)
        
        # 计算最大绝对值，确定缩放因子
        max_abs = np.max(np.abs(data), axis=0)
        self.scale_factor = np.maximum(max_abs, 1)  # 避免除零
        
        self.is_fitted = True
        self._params = {
            'scale_factor': self.scale_factor,
            'n_decimals': self.n_decimals
        }
        
        logger.info(f"小数缩放器拟合完成，缩放因子: {self.scale_factor}")
        return self
    
    def transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """
        应用小数缩放
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 缩放后的数据
        """
        if not self.is_fitted:
            raise ValueError("缩放器尚未拟合，请先调用fit()方法")
        
        data = self._check_data(data)
        
        # 应用小数缩放公式
        scaled = data / self.scale_factor
        
        return scaled
    
    def _inverse_transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """逆变换"""
        data = self._check_data(data)
        
        # 逆变换公式
        original = data * self.scale_factor
        
        return original


class LogNormalizer(BaseNormalizer):
    """
    对数归一化算法
    
    对数据进行对数变换，适用于右偏分布数据
    公式: X_log = log(X + 1) 或 X_log = log(X + ε)
    """
    
    def __init__(self, epsilon: float = 1e-8, base: float = np.e):
        """
        初始化对数归一化器
        
        Args:
            epsilon: 避免log(0)的小常数，默认1e-8
            base: 对数底数，默认自然对数e
        """
        super().__init__()
        self.epsilon = epsilon
        self.base = base
    
    def fit(self, data: Union[np.ndarray, List, pd.Series]) -> 'LogNormalizer':
        """
        拟合对数归一化器
        
        Args:
            data: 输入数据
            
        Returns:
            self: 返回自身实例
        """
        data = self._check_data(data)
        
        # 检查数据是否全为正数
        if np.any(data < 0):
            logger.warning("数据包含负数，将对所有数据加上最小值的绝对值")
            min_val = np.min(data)
            if min_val < 0:
                self.offset = abs(min_val)
            else:
                self.offset = 0
        else:
            self.offset = 0
        
        self.is_fitted = True
        self._params = {
            'epsilon': self.epsilon,
            'base': self.base,
            'offset': self.offset
        }
        
        logger.info(f"对数归一化器拟合完成，偏移量: {self.offset}")
        return self
    
    def transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """
        应用对数归一化
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 对数归一化后的数据
        """
        if not self.is_fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")
        
        data = self._check_data(data)
        
        # 应用对数变换
        adjusted_data = data + self.offset + self.epsilon
        
        if self.base == np.e:
            log_data = np.log(adjusted_data)
        else:
            log_data = np.log(adjusted_data) / np.log(self.base)
        
        return log_data
    
    def _inverse_transform(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """逆变换"""
        data = self._check_data(data)
        
        # 逆变换公式
        if self.base == np.e:
            exp_data = np.exp(data)
        else:
            exp_data = self.base ** data
        
        original = exp_data - self.offset - self.epsilon
        
        return original


# 便捷函数
def min_max_normalize(data: Union[np.ndarray, List, pd.Series], 
                     feature_range: tuple = (0, 1)) -> np.ndarray:
    """Min-Max归一化便捷函数"""
    normalizer = MinMaxNormalizer(feature_range)
    return normalizer.fit_transform(data)


def z_score_normalize(data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
    """Z-Score标准化便捷函数"""
    normalizer = ZScoreNormalizer()
    return normalizer.fit_transform(data)


def robust_scale(data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
    """稳健缩放便捷函数"""
    scaler = RobustScaler()
    return scaler.fit_transform(data)


def decimal_scale(data: Union[np.ndarray, List, pd.Series], 
                 n_decimals: int = 2) -> np.ndarray:
    """小数缩放便捷函数"""
    scaler = DecimalScaler(n_decimals)
    return scaler.fit_transform(data)


def log_normalize(data: Union[np.ndarray, List, pd.Series], 
                 epsilon: float = 1e-8, base: float = np.e) -> np.ndarray:
    """对数归一化便捷函数"""
    normalizer = LogNormalizer(epsilon, base)
    return normalizer.fit_transform(data)
