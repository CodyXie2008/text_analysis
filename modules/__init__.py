# -*- coding: utf-8 -*-
"""
从众心理分析功能模块

包含时间集中度分析、点赞互动分析、数据清洗、情感分析等功能
"""

# 旧版本模块（向后兼容，若不存在则跳过）
try:
    from .conformity_time_analysis import ConformityTimeAnalyzer  # type: ignore
except Exception:
    ConformityTimeAnalyzer = None  # type: ignore
try:
    from .like_interaction_analysis import LikeInteractionAnalyzer  # type: ignore
except Exception:
    LikeInteractionAnalyzer = None  # type: ignore
try:
    from .data_preparation_and_cleaning import CommentDataProcessor  # type: ignore
except Exception:
    CommentDataProcessor = None  # type: ignore

# 新版本优化模块
from .sentiment_analyzer_optimized import SentimentAnalyzer
from .time_analysis_optimized import TimeAnalysisAnalyzer
from .like_analysis_optimized import LikeAnalysisAnalyzer
from .data_cleaning_optimized import DataCleaningAnalyzer

__all__ = [name for name in [
    # 旧版本模块（仅导出存在的）
    'ConformityTimeAnalyzer' if ConformityTimeAnalyzer else None,
    'LikeInteractionAnalyzer' if LikeInteractionAnalyzer else None,
    'CommentDataProcessor' if CommentDataProcessor else None,
    # 新版本优化模块
    'SentimentAnalyzer',
    'TimeAnalysisAnalyzer',
    'LikeAnalysisAnalyzer',
    'DataCleaningAnalyzer'
] if name]