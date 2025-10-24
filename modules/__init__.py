# -*- coding: utf-8 -*-
"""
文本分析功能模块

包含数据清洗、情感分析、相似度分析、时间分析、点赞分析等功能
"""

# 核心分析模块
from .sentiment_conformity_analyzer import SentimentConformityAnalyzer
from .similarity_conformity_analyzer import SimilarityConformityAnalyzer
from .conformity_time_analyzer import ConformityTimeAnalyzer
from .like_conformity_analyzer import LikeConformityAnalyzer

# API管理器
from core.aliyun_api_manager import AliyunAPIManager, get_aliyun_api_manager, is_aliyun_api_available

__all__ = [
    # 核心分析模块
    'SentimentConformityAnalyzer',
    'SimilarityConformityAnalyzer',
    'ConformityTimeAnalyzer', 
    'LikeConformityAnalyzer',
    # API管理器
    'AliyunAPIManager',
    'get_aliyun_api_manager',
    'is_aliyun_api_available'
]