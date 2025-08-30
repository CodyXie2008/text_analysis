# -*- coding: utf-8 -*-
"""
文本分析统计算法模块

包含常用的数据标准化、归一化、相似度计算等算法
"""

from .normalization import (
    MinMaxNormalizer,
    ZScoreNormalizer,
    RobustScaler,
    DecimalScaler,
    LogNormalizer
)

from .similarity import (
    CosineSimilarity,
    JaccardSimilarity,
    EuclideanDistance,
    ManhattanDistance,
    PearsonCorrelation
)

from .statistics import (
    DescriptiveStats,
    OutlierDetector,
    DistributionAnalyzer,
    CorrelationAnalyzer
)

from .scoring import (
    CompositeScore,
    WeightedScore,
    RankingScore,
    ThresholdClassifier
)

__all__ = [
    # 归一化算法
    'MinMaxNormalizer',
    'ZScoreNormalizer', 
    'RobustScaler',
    'DecimalScaler',
    'LogNormalizer',
    
    # 相似度算法
    'CosineSimilarity',
    'JaccardSimilarity',
    'EuclideanDistance',
    'ManhattanDistance',
    'PearsonCorrelation',
    
    # 统计分析
    'DescriptiveStats',
    'OutlierDetector',
    'DistributionAnalyzer',
    'CorrelationAnalyzer',
    
    # 评分算法
    'CompositeScore',
    'WeightedScore',
    'RankingScore',
    'ThresholdClassifier'
]
