# -*- coding: utf-8 -*-
"""
文本分析功能模块

包含数据清洗、情感分析、相似度分析、时间分析、点赞分析等功能
"""

# 核心分析模块
from .sentiment_analyzer_optimized import SentimentAnalyzer
from .time_analysis_optimized import TimeAnalysisAnalyzer
from .like_analysis_optimized import LikeAnalysisAnalyzer
from .data_cleaning_optimized import DataCleaningAnalyzer

# API管理器
from ..core.aliyun_api_manager import AliyunAPIManager, get_aliyun_api_manager, is_aliyun_api_available

__all__ = [
    # 核心分析模块
    'SentimentAnalyzer',
    'TimeAnalysisAnalyzer', 
    'LikeAnalysisAnalyzer',
    'DataCleaningAnalyzer',
    # API管理器
    'AliyunAPIManager',
    'get_aliyun_api_manager',
    'is_aliyun_api_available'
]