# -*- coding: utf-8 -*-
"""
从众心理时间分析模块

以父评论为舆论场，分析子评论的时间从众心理特征
子评论时间-父评论时间 时间越近 分数越高，然后归一化处理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import argparse
import os
import json

# 添加项目根目录到Python路径
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from text_analysis.core.base_analyzer import BaseAnalyzer, create_parser, parse_common_args
from text_analysis.algorithms.normalization import MinMaxNormalizer

logger = logging.getLogger(__name__)


class ConformityTimeAnalyzer(BaseAnalyzer):
    """
    从众心理时间分析器
    
    分析子评论相对于父评论的时间从众心理特征：
    1. 计算子评论与父评论的时间间隔
    2. 基于时间间隔计算从众心理分数（时间越近分数越高）
    3. 对分数进行归一化处理
    4. 生成时间从众心理分析报告
    """
    
    def __init__(self, module_name: str = "conformity_time_analyzer", output_dir: str = None, **kwargs):
        """初始化从众心理时间分析器"""
        super().__init__(module_name=module_name, **kwargs)
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output")
        self.normalizer = MinMaxNormalizer(feature_range=(0, 1))
        self.results = {}
        
    def calculate_time_conformity_score(self, data: Union[pd.DataFrame, List[Dict]]) -> Union[pd.DataFrame, List[Dict]]:
        """
        计算时间从众心理分数
        
        Args:
            data: 包含评论数据的DataFrame或字典列表，需要包含以下字段：
                - create_time: 评论创建时间（Unix时间戳）或 comment_time: 已转换的datetime
                - comment_id: 评论ID
                - is_parent: 父评论标识（可选，如果数据已通过数据清洗模块处理）
                
        Returns:
            Union[pd.DataFrame, List[Dict]]: 包含时间从众心理分数的数据
        """
        logger.info("开始计算时间从众心理分数...")
        
        # 处理输入数据格式
        if isinstance(data, list):
            # 如果是字典列表，转换为DataFrame
            df = pd.DataFrame(data)
            return_dict = True
        else:
            df = data.copy()
            return_dict = False
        
        # 检查数据是否已经过时间标准化处理
        if 'comment_time' in df.columns and 'is_parent' in df.columns:
            # 数据已经过数据清洗模块的时间标准化处理
            logger.info("检测到已处理的时间数据，直接使用")
            
            # 处理时间字段（可能是字符串格式）
            if df['comment_time'].dtype == 'object':
                df['comment_time'] = pd.to_datetime(df['comment_time'])
            
            parent_data = df[df['is_parent'] == True]
            if len(parent_data) > 0:
                parent_time = parent_data['comment_time'].iloc[0]
                parent_id = parent_data['comment_id'].iloc[0]
            else:
                raise ValueError("未找到父评论数据")
        else:
            # 需要手动处理时间数据
            logger.info("数据未经过时间标准化，开始处理...")
            
            # 确保时间列存在并转换为datetime类型
            if 'create_time' in df.columns:
                # 将Unix时间戳转换为datetime
                df['comment_time'] = pd.to_datetime(df['create_time'], unit='s')
            elif 'comment_time' in df.columns:
                df['comment_time'] = pd.to_datetime(df['comment_time'])
            else:
                raise ValueError("数据中缺少时间列，需要 'create_time' 或 'comment_time' 列")
            
            # 按时间排序，第一个是父评论
            df = df.sort_values('comment_time').reset_index(drop=True)
            
            # 获取父评论时间（第一个评论）
            parent_time = df['comment_time'].iloc[0]
            parent_id = df['comment_id'].iloc[0]
            
            # 添加父评论标识
            df['is_parent'] = df['comment_id'] == parent_id
        
        # 计算每个子评论的时间从众心理分数
        conformity_scores = []
        time_intervals = []
        
        # 首先计算所有时间间隔
        for idx, row in df.iterrows():
            comment_time = row['comment_time']
            time_diff = abs((comment_time - parent_time).total_seconds())
            time_intervals.append(time_diff)
        
        # 计算自适应衰减因子
        adaptive_decay_factor = self._calculate_adaptive_decay_factor(time_intervals)
        logger.info(f"自适应衰减因子: {adaptive_decay_factor:.0f}秒 ({adaptive_decay_factor/60:.1f}分钟)")
        
        # 使用自适应衰减因子计算分数
        for time_diff in time_intervals:
            # 计算从众心理分数（时间越近分数越高）
            # 使用指数衰减函数：score = exp(-time_diff / adaptive_decay_factor)
            raw_score = np.exp(-time_diff / adaptive_decay_factor)
            conformity_scores.append(raw_score)
        
        # 将结果添加到DataFrame
        df['time_interval_seconds'] = time_intervals
        df['raw_conformity_score'] = conformity_scores
        
        # 对分数进行归一化处理
        normalized_scores = self.normalizer.fit_transform(
            np.array(conformity_scores).reshape(-1, 1)
        ).flatten()
        
        df['normalized_conformity_score'] = normalized_scores
        
        # 添加时间间隔分类
        df['time_category'] = self._categorize_time_intervals(time_intervals)
        
        # 添加父评论标识
        df['is_parent'] = df['comment_id'] == parent_id
        
        logger.info(f"时间从众心理分数计算完成，共处理 {len(df)} 条评论")
        logger.info(f"父评论ID: {parent_id}, 子评论数量: {len(df) - 1}")
        
        if return_dict:
            return df.to_dict('records')
        else:
            return df
    
    def _categorize_time_intervals(self, time_intervals: List[float]) -> List[str]:
        """
        对时间间隔进行分类
        
        Args:
            time_intervals: 时间间隔列表（秒）
            
        Returns:
            List[str]: 时间分类列表
        """
        categories = []
        for interval in time_intervals:
            if interval <= 60:  # 1分钟内
                categories.append('立即响应')
            elif interval <= 300:  # 5分钟内
                categories.append('快速响应')
            elif interval <= 1800:  # 30分钟内
                categories.append('较快响应')
            elif interval <= 3600:  # 1小时内
                categories.append('一般响应')
            elif interval <= 86400:  # 24小时内
                categories.append('延迟响应')
            else:  # 超过24小时
                categories.append('滞后响应')
        
        return categories
    
    def _calculate_adaptive_decay_factor(self, time_intervals: List[float]) -> float:
        """
        根据数据特征自适应计算衰减因子
        
        Args:
            time_intervals: 时间间隔列表（秒）
            
        Returns:
            float: 自适应衰减因子（秒）
        """
        if not time_intervals:
            return 3600  # 默认1小时
        
        # 计算时间间隔的统计特征
        median_time = np.median(time_intervals)
        mean_time = np.mean(time_intervals)
        std_time = np.std(time_intervals)
        
        # 使用中位数时间差的1/4作为基础衰减因子
        # 这样能确保大部分评论的分数在合理范围内
        base_decay = median_time / 4
        
        # 考虑数据分布的标准差，如果分布很分散，适当增加衰减因子
        if std_time > median_time:
            # 数据分布很分散，增加衰减因子
            adaptive_decay = base_decay * 1.5
        else:
            # 数据分布相对集中，使用基础衰减因子
            adaptive_decay = base_decay
        
        # 设置合理的上下限
        min_decay = 300    # 5分钟
        max_decay = 86400  # 24小时
        
        # 确保衰减因子在合理范围内
        final_decay = np.clip(adaptive_decay, min_decay, max_decay)
        
        logger.info(f"时间间隔统计 - 中位数: {median_time:.0f}秒, 均值: {mean_time:.0f}秒, 标准差: {std_time:.0f}秒")
        logger.info(f"衰减因子计算 - 基础: {base_decay:.0f}秒, 自适应: {adaptive_decay:.0f}秒, 最终: {final_decay:.0f}秒")
        
        return final_decay
    
    def calculate_parent_conformity_score(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """
        计算父评论的总体从众心理分数
        
        Args:
            data: 包含评论数据的DataFrame或字典列表
            
        Returns:
            Dict: 父评论从众心理分数统计结果
        """
        logger.info("开始计算父评论从众心理分数...")
        
        # 计算时间从众心理分数
        result_data = self.calculate_time_conformity_score(data)
        
        # 转换为DataFrame格式进行处理
        if isinstance(result_data, list):
            df = pd.DataFrame(result_data)
        else:
            df = result_data
        
        # 分离父评论和子评论
        parent_data = df[df['is_parent'] == True]
        child_data = df[df['is_parent'] == False]
        
        if len(parent_data) == 0:
            raise ValueError("未找到父评论数据")
        
        parent_comment = parent_data.iloc[0]
        parent_id = parent_comment['comment_id']
        
        # 计算子评论的从众心理分数统计
        child_scores = child_data['normalized_conformity_score'].tolist()
        child_time_diffs = child_data['time_interval_seconds'].tolist()
        
        if len(child_scores) == 0:
            logger.warning("未找到子评论数据")
            return {
                'parent_comment_id': parent_id,
                'child_comment_count': 0,
                'parent_conformity_score': 0.0,
                'message': '无子评论数据'
            }
        
        # 基础统计
        avg_score = np.mean(child_scores)
        median_score = np.median(child_scores)
        max_score = np.max(child_scores)
        min_score = np.min(child_scores)
        std_score = np.std(child_scores)
        
        # 高从众心理比例（分数 > 0.7）
        high_conformity_count = sum(1 for score in child_scores if score > 0.7)
        high_conformity_ratio = high_conformity_count / len(child_scores)
        
        # 早期响应比例（1小时内）
        early_response_count = sum(1 for time_diff in child_time_diffs if time_diff <= 3600)
        early_response_ratio = early_response_count / len(child_time_diffs)
        
        # 快速响应比例（10分钟内）
        quick_response_count = sum(1 for time_diff in child_time_diffs if time_diff <= 600)
        quick_response_ratio = quick_response_count / len(child_time_diffs)
        
        result = {
            'parent_comment_id': parent_id,
            'child_comment_count': len(child_scores),
            'parent_conformity_score': avg_score,  # 主要分数：平均值
            'statistics': {
                'mean_score': avg_score,
                'median_score': median_score,
                'max_score': max_score,
                'min_score': min_score,
                'std_score': std_score
            },
            'conformity_distribution': {
                'high_conformity_count': high_conformity_count,
                'high_conformity_ratio': high_conformity_ratio,
                'early_response_count': early_response_count,
                'early_response_ratio': early_response_ratio,
                'quick_response_count': quick_response_count,
                'quick_response_ratio': quick_response_ratio
            },
            'time_analysis': {
                'avg_time_diff': np.mean(child_time_diffs),
                'median_time_diff': np.median(child_time_diffs),
                'min_time_diff': np.min(child_time_diffs),
                'max_time_diff': np.max(child_time_diffs)
            },
            'parent_info': {
                'create_time': parent_comment['create_time'],
                'comment_time': parent_comment['comment_time'].isoformat() if 'comment_time' in parent_comment else None
            }
        }
        
        logger.info(f"父评论从众心理分数计算完成: {parent_id}, 分数: {avg_score:.4f}")
        
        return result
    
    def analyze_conformity_patterns(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        分析从众心理模式
        
        Args:
            data: 包含时间从众心理分数的DataFrame
            
        Returns:
            Dict: 从众心理模式分析结果
        """
        logger.info("开始分析从众心理模式...")
        
        analysis_results = {}
        
        # 1. 整体统计
        analysis_results['overall_stats'] = {
            'total_comments': len(data),
            'unique_parents': data['parent_comment_id'].nunique(),
            'avg_conformity_score': data['normalized_conformity_score'].mean(),
            'std_conformity_score': data['normalized_conformity_score'].std(),
            'median_conformity_score': data['normalized_conformity_score'].median(),
            'min_conformity_score': data['normalized_conformity_score'].min(),
            'max_conformity_score': data['normalized_conformity_score'].max()
        }
        
        # 2. 时间分类统计
        time_category_stats = data.groupby('time_category').agg({
            'normalized_conformity_score': ['count', 'mean', 'std'],
            'time_interval_seconds': ['mean', 'median']
        }).round(4)
        
        analysis_results['time_category_stats'] = time_category_stats
        
        # 3. 父评论级别的从众心理分析
        parent_level_stats = data.groupby('parent_comment_id').agg({
            'normalized_conformity_score': ['count', 'mean', 'std'],
            'time_interval_seconds': ['mean', 'min', 'max']
        }).round(4)
        
        analysis_results['parent_level_stats'] = parent_level_stats
        
        # 4. 从众心理强度分布
        conformity_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        conformity_labels = ['很低', '低', '中等', '高', '很高']
        data['conformity_level'] = pd.cut(
            data['normalized_conformity_score'], 
            bins=conformity_bins, 
            labels=conformity_labels, 
            include_lowest=True
        )
        
        conformity_distribution = data['conformity_level'].value_counts().sort_index()
        analysis_results['conformity_distribution'] = conformity_distribution
        
        # 5. 时间间隔与从众心理的相关性
        correlation = data['time_interval_seconds'].corr(data['normalized_conformity_score'])
        analysis_results['time_conformity_correlation'] = correlation
        
        # 6. 高从众心理评论的特征分析
        high_conformity_threshold = data['normalized_conformity_score'].quantile(0.8)
        high_conformity_data = data[data['normalized_conformity_score'] >= high_conformity_threshold]
        
        analysis_results['high_conformity_analysis'] = {
            'threshold': high_conformity_threshold,
            'count': len(high_conformity_data),
            'percentage': len(high_conformity_data) / len(data) * 100,
            'avg_time_interval': high_conformity_data['time_interval_seconds'].mean(),
            'time_category_distribution': high_conformity_data['time_category'].value_counts()
        }
        
        logger.info("从众心理模式分析完成")
        
        return analysis_results
    
    def generate_time_conformity_report(self, data: pd.DataFrame, 
                                      analysis_results: Dict[str, any]) -> str:
        """
        生成时间从众心理分析报告
        
        Args:
            data: 包含分析结果的DataFrame
            analysis_results: 分析结果字典
            
        Returns:
            str: Markdown格式的分析报告
        """
        logger.info("生成时间从众心理分析报告...")
        
        report = f"""# 从众心理时间分析报告

## 分析概述
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总评论数**: {analysis_results['overall_stats']['total_comments']}
- **父评论数**: {analysis_results['overall_stats']['unique_parents']}
- **平均从众心理分数**: {analysis_results['overall_stats']['avg_conformity_score']:.4f}

## 主要发现

### 1. 整体从众心理特征
- **平均从众心理分数**: {analysis_results['overall_stats']['avg_conformity_score']:.4f}
- **标准差**: {analysis_results['overall_stats']['std_conformity_score']:.4f}
- **中位数**: {analysis_results['overall_stats']['median_conformity_score']:.4f}
- **分数范围**: [{analysis_results['overall_stats']['min_conformity_score']:.4f}, {analysis_results['overall_stats']['max_conformity_score']:.4f}]

### 2. 时间间隔与从众心理相关性
- **相关系数**: {analysis_results['time_conformity_correlation']:.4f}
- **解释**: {'强负相关' if analysis_results['time_conformity_correlation'] < -0.5 else '中等负相关' if analysis_results['time_conformity_correlation'] < -0.3 else '弱负相关' if analysis_results['time_conformity_correlation'] < 0 else '弱正相关' if analysis_results['time_conformity_correlation'] < 0.3 else '中等正相关' if analysis_results['time_conformity_correlation'] < 0.5 else '强正相关'}（时间间隔越短，从众心理越强）

### 3. 时间分类统计
"""
        
        # 添加时间分类统计表格
        time_stats = analysis_results['time_category_stats']
        for category in time_stats.index:
            count = time_stats.loc[category, ('normalized_conformity_score', 'count')]
            mean_score = time_stats.loc[category, ('normalized_conformity_score', 'mean')]
            avg_interval = time_stats.loc[category, ('time_interval_seconds', 'mean')]
            report += f"- **{category}**: {count}条评论，平均分数{mean_score:.4f}，平均间隔{avg_interval:.1f}秒\n"
        
        report += f"""
### 4. 从众心理强度分布
"""
        
        # 添加从众心理分布
        distribution = analysis_results['conformity_distribution']
        for level, count in distribution.items():
            percentage = count / len(data) * 100
            report += f"- **{level}**: {count}条评论 ({percentage:.1f}%)\n"
        
        report += f"""
### 5. 高从众心理评论分析
- **高从众心理阈值**: {analysis_results['high_conformity_analysis']['threshold']:.4f}
- **高从众心理评论数**: {analysis_results['high_conformity_analysis']['count']}条
- **占比**: {analysis_results['high_conformity_analysis']['percentage']:.1f}%
- **平均时间间隔**: {analysis_results['high_conformity_analysis']['avg_time_interval']:.1f}秒

### 6. 高从众心理评论的时间分布
"""
        
        # 添加高从众心理评论的时间分布
        time_dist = analysis_results['high_conformity_analysis']['time_category_distribution']
        for category, count in time_dist.items():
            percentage = count / analysis_results['high_conformity_analysis']['count'] * 100
            report += f"- **{category}**: {count}条评论 ({percentage:.1f}%)\n"
        
        report += """
## 结论与建议

### 主要结论
1. **时间效应明显**: 评论时间越接近父评论，从众心理越强
2. **响应速度影响**: 快速响应（5分钟内）的评论表现出更强的从众心理
3. **分布特征**: 从众心理分数呈现{'正态分布' if abs(analysis_results['overall_stats']['avg_conformity_score'] - 0.5) < 0.1 else '偏态分布'}特征

### 建议
1. **内容监控**: 重点关注高从众心理区域的评论内容
2. **时间窗口**: 在父评论发布后的前30分钟内加强监控
3. **用户教育**: 提醒用户避免盲目跟风评论
4. **算法优化**: 在推荐算法中考虑时间从众心理因素

---
*本报告由从众心理时间分析系统自动生成*
"""
        
        return report
    
    def analyze(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """
        实现BaseAnalyzer的analyze方法
        """
        logger.info("开始从众心理时间分析...")
        
        # 计算时间从众心理分数
        processed_data = self.calculate_time_conformity_score(data)
        if isinstance(processed_data, list):
            processed_df = pd.DataFrame(processed_data)
        else:
            processed_df = processed_data
        
        # 计算父评论总体时间从众心理分数
        parent_overall_result = self.calculate_parent_conformity_score(processed_df)
        
        # 分析时间从众心理模式
        pattern_results = self.analyze_conformity_patterns(processed_df)
        
        # 生成报告
        report_content = self.generate_time_conformity_report(processed_df, pattern_results)
        
        # 准备输出结果
        output_result = {
            "analysis_metadata": {
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analyzer_version": "v1.0.0",
                "data_source": getattr(self, 'data_source', 'unknown'),
                "total_comments": len(processed_df)
            },
            "parent_comment_info": {
                "comment_id": parent_overall_result.get('parent_comment_id', 'N/A'),
                "content": processed_df[processed_df['comment_id'] == parent_overall_result.get('parent_comment_id', 'N/A')]['content'].iloc[0] if not processed_df[processed_df['comment_id'] == parent_overall_result.get('parent_comment_id', 'N/A')].empty else 'N/A',
            },
            "parent_environment_time_analysis": parent_overall_result,
            "time_classification": pattern_results.get('time_category_distribution', {}),
            "top_high_conformity_comments": processed_df.nlargest(10, 'normalized_conformity_score').to_dict('records') if not processed_df.empty else []
        }
        
        logger.info("从众心理时间分析完成")
        return output_result
    
    def run_analysis(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        运行完整的从众心理时间分析
        
        Args:
            data: 包含评论数据的DataFrame
            
        Returns:
            Dict: 完整的分析结果
        """
        logger.info("开始从众心理时间分析...")
        
        # 1. 计算时间从众心理分数
        result_data = self.calculate_time_conformity_score(data)
        
        # 2. 分析从众心理模式
        analysis_results = self.analyze_conformity_patterns(result_data)
        
        # 3. 生成报告
        report = self.generate_time_conformity_report(result_data, analysis_results)
        
        # 4. 保存结果
        self._save_results(result_data, analysis_results, report)
        
        # 5. 保存到实例变量
        self.results = {
            'data': result_data,
            'analysis': analysis_results,
            'report': report
        }
        
        logger.info("从众心理时间分析完成")
        
        return self.results
    
    def _save_results(self, data: pd.DataFrame, analysis_results: Dict[str, any], report: str):
        """保存分析结果"""
        logger.info("保存分析结果...")
        
        # 创建输出目录
        output_dir = os.path.join(self.output_dir, "conformity_time_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细数据
        data_path = os.path.join(output_dir, f"conformity_time_data_{timestamp}.csv")
        data.to_csv(data_path, index=False, encoding='utf-8-sig')
        
        # 保存分析结果
        analysis_path = os.path.join(output_dir, f"conformity_time_analysis_{timestamp}.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存报告
        report_path = os.path.join(output_dir, f"conformity_time_report_{timestamp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"分析结果已保存到: {output_dir}")


def main():
    """主函数"""
    parser = create_parser("conformity_time_analyzer", "从众心理时间分析")
    parser.add_argument('--parent-id-col', default='parent_comment_id',
                       help='父评论ID列名')
    parser.add_argument('--time-col', default='comment_time',
                       help='评论时间列名')
    parser.add_argument('--comment-id-col', default='comment_id',
                       help='评论ID列名')
    
    args = parser.parse_args()
    args = parse_common_args(parser, args)
    
    # 创建分析器
    analyzer = ConformityTimeAnalyzer(
        module_name="conformity_time_analyzer",
        video_id=args.video_id
    )
    
    # 加载数据
    if args.use_cleaned_data:
        logger.info(f"使用清洗后的数据: {args.cleaned_data_path}")
        data = analyzer.load_data(use_cleaned_data=True, cleaned_data_path=args.cleaned_data_path)
    else:
        logger.info("从数据库加载数据...")
        data = analyzer.load_data(limit=args.limit)
    
    # 确保必要的列存在
    required_columns = [args.parent_id_col, args.time_col, args.comment_id_col]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")
    
    # 运行分析
    results = analyzer.run_analysis(data)
    
    # 打印摘要
    print("\n=== 从众心理时间分析摘要 ===")
    print(f"总评论数: {results['analysis']['overall_stats']['total_comments']}")
    print(f"父评论数: {results['analysis']['overall_stats']['unique_parents']}")
    print(f"平均从众心理分数: {results['analysis']['overall_stats']['avg_conformity_score']:.4f}")
    print(f"时间相关性: {results['analysis']['time_conformity_correlation']:.4f}")
    
    logger.info("从众心理时间分析完成")


if __name__ == "__main__":
    main()
