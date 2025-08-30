"""
数据存储路径配置文件
统一管理text_analysis模块的所有数据存储路径
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# 获取项目根目录 - 指向MediaCrawler项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据目录结构
DATA_DIRS = {
    'raw': PROJECT_ROOT / 'data' / 'raw',                    # 原始数据
    'processed': PROJECT_ROOT / 'data' / 'processed',        # 处理后数据
    'results': PROJECT_ROOT / 'data' / 'results',            # 分析结果
    'visualizations': PROJECT_ROOT / 'data' / 'visualizations', # 可视化图表
    'reports': PROJECT_ROOT / 'data' / 'reports',            # 分析报告
    'temp': PROJECT_ROOT / 'data' / 'temp'                   # 临时文件
}

# 确保目录存在
def ensure_directories():
    """确保所有数据目录都存在"""
    for dir_path in DATA_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)

# 获取文件路径
def get_data_path(category: str, filename: str) -> str:
    """获取指定类别的数据文件路径"""
    if category not in DATA_DIRS:
        raise ValueError(f"未知的数据类别: {category}")
    
    return str(DATA_DIRS[category] / filename)

 # 生成带时间戳和视频ID的文件名
def get_timestamped_filename(prefix: str, video_id: str = None, suffix: str = '.json') -> str:
    """生成带时间戳和视频ID的文件名，防止覆盖"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if video_id:
        return f"{prefix}_{video_id}_{timestamp}{suffix}"
    else:
        return f"{prefix}_{timestamp}{suffix}"

# 常用文件路径
COMMON_PATHS = {
    'processed_comments': get_data_path('processed', 'douyin_comments_processed.json'),
    'stopwords': PROJECT_ROOT / 'docs' / 'hit_stopwords.txt'
}

# 统一的分析模块路径管理器
class AnalysisPathManager:
    """统一的分析模块路径管理器"""
    
    def __init__(self, module_name: str, video_id: str = None):
        """
        初始化路径管理器
        
        Args:
            module_name: 模块名称 (sentiment, time, like, cleaning)
            video_id: 视频ID，用于文件命名
        """
        self.module_name = module_name
        self.video_id = video_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def get_results_paths(self) -> dict:
        """获取结果文件路径"""
        base_name = f"results_{self.module_name}"
        if self.video_id:
            base_name = f"{base_name}_{self.video_id}"
        return {
            'csv': get_data_path('results', f"{base_name}_{self.timestamp}.csv"),
            'json': get_data_path('results', f"{base_name}_{self.timestamp}.json")
        }
    
    def get_report_path(self) -> str:
        """获取报告文件路径"""
        base_name = f"reports_{self.module_name}"
        if self.video_id:
            base_name = f"{base_name}_{self.video_id}"
        return get_data_path('reports', f"{base_name}_{self.timestamp}.json")
    
    def get_visualization_path(self, chart_type: str = "main") -> str:
        """获取可视化文件路径"""
        base_name = f"visualizations_{self.module_name}"
        if self.video_id:
            base_name = f"{base_name}_{self.video_id}"
        return get_data_path('visualizations', f"{base_name}_{self.timestamp}.png")
    
    def get_cleaned_data_path(self) -> str:
        """获取清洗数据文件路径"""
        base_name = "processed_cleaning"
        if self.video_id:
            base_name = f"{base_name}_{self.video_id}"
        return get_data_path('processed', f"{base_name}_{self.timestamp}.json")


def resolve_latest_cleaned_data(video_id: Optional[str] = None) -> Optional[str]:
    """解析并返回最新的清洗数据文件路径。

    优先顺序：
    1) processed_cleaning_{video_id}_*.json（按修改时间倒序）
    2) processed_cleaning_*.json（按修改时间倒序）
    3) douyin_comments_processed_{video_id}.json（兼容别名）
    4) douyin_comments_processed.json（全局别名）

    Returns: 绝对路径字符串或 None
    """
    processed_dir: Path = DATA_DIRS['processed']

    # 1) video_id 定向匹配
    if video_id:
        candidates = sorted(
            processed_dir.glob(f"processed_cleaning_{video_id}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    # 2) 全局最新 cleaned_data_*.json
    candidates = sorted(
        processed_dir.glob("processed_cleaning_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])

    # 3) 兼容别名（带 video_id）
    if video_id:
        alias_with_id = processed_dir / f"douyin_comments_processed_{video_id}.json"
        if alias_with_id.exists():
            return str(alias_with_id)

    # 4) 全局别名
    alias_global = processed_dir / "douyin_comments_processed.json"
    if alias_global.exists():
        return str(alias_global)

    return None

# 模块特定的路径生成函数（保持向后兼容）
class PathManager:
    """路径管理器（向后兼容）"""
    
    @staticmethod
    def get_sentiment_analysis_paths(video_id: str = None):
        """获取情感分析模块的路径"""
        manager = AnalysisPathManager("sentiment", video_id)
        return {
            'results_csv': manager.get_results_paths()['csv'],
            'results_json': manager.get_results_paths()['json'],
            'report': manager.get_report_path(),
            'visualization': manager.get_visualization_path()
        }
    
    @staticmethod
    def get_time_analysis_paths(video_id: str = None):
        """获取时间分析模块的路径"""
        manager = AnalysisPathManager("time", video_id)
        return {
            'report': manager.get_report_path(),
            'visualization': manager.get_visualization_path()
        }
    
    @staticmethod
    def get_like_analysis_paths(video_id: str = None):
        """获取点赞分析模块的路径"""
        manager = AnalysisPathManager("like", video_id)
        return {
            'report': manager.get_report_path(),
            'visualization': manager.get_visualization_path()
        }
    
    @staticmethod
    def get_data_cleaning_paths(video_id: str = None):
        """获取数据清洗模块的路径"""
        manager = AnalysisPathManager("cleaning", video_id)
        return {
            'processed_data': manager.get_cleaned_data_path(),
            'cleaning_report': manager.get_report_path()
        }
    
    @staticmethod
    def get_conformity_analysis_paths(video_id: str = None):
        """获取从众心理综合分析路径"""
        manager = AnalysisPathManager("conformity", video_id)
        return {
            'comprehensive_report': manager.get_report_path(),
            'summary_visualization': manager.get_visualization_path("summary")
        }

# 便捷函数
def get_conformity_time_paths(video_id: str = None):
    """获取从众心理时间分析路径"""
    return PathManager.get_time_analysis_paths(video_id)

def get_like_interaction_paths(video_id: str = None):
    """获取点赞互动分析路径"""
    return PathManager.get_like_analysis_paths(video_id)

# 初始化时确保目录存在
ensure_directories() 