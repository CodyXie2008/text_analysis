#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感从众心理分析模块

基于父评论和子评论的情感差异分析从众心理特征
通过阿里云API计算情感值，分析情感相似性
"""

import os
import sys
import json
import time
import threading
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from text_analysis.core.base_analyzer import BaseAnalyzer, create_parser, parse_common_args
from text_analysis.core.aliyun_api_manager import get_aliyun_api_manager, is_aliyun_api_available
from text_analysis.algorithms.normalization import MinMaxNormalizer

import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentConformityAnalyzer(BaseAnalyzer):
    """情感从众心理分析器"""
    
    def __init__(self, module_name: str = "sentiment_conformity_analyzer", 
                 video_id: Optional[str] = None, output_dir: Optional[str] = None, **kwargs):
        """初始化情感从众心理分析器"""
        super().__init__(module_name=module_name, **kwargs)
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output")
        self.normalizer = MinMaxNormalizer(feature_range=(0, 1))
        self.results = {}
        
        # 初始化阿里云API管理器
        self.api_manager = get_aliyun_api_manager()
        if not self.api_manager:
            raise ValueError("阿里云API配置缺失，请设置环境变量")
        
        # 并发控制参数
        self.sa_concurrency = kwargs.get('sa_concurrency', 8)
        self.sa_batch_size = kwargs.get('sa_batch_size', 200)
        self.sa_throttle_ms = kwargs.get('sa_throttle_ms', 0)
        self._stats_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_analyzed': 0,
            'parent_comments': 0,
            'child_comments': 0,
            'api_calls': 0,
            'api_errors': 0,
            'total_confidence': 0.0,
            'total_sentiment_score': 0.0
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        使用阿里云API分析文本情感
        
        Args:
            text: 待分析的文本
            
        Returns:
            Dict: 包含情感分析结果的字典
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'method': 'aliyun'
            }
        
        try:
            result = self.api_manager.analyze_sentiment(text)
            
            with self._stats_lock:
                self.stats['api_calls'] += 1
                self.stats['total_confidence'] += result.get('confidence', 0.0)
                self.stats['total_sentiment_score'] += result.get('score', 0.0)
            
            return {
                'sentiment': result['sentiment'],
                'score': result['score'],
                'confidence': result['confidence'],
                'method': 'aliyun'
            }
        except Exception as e:
            logger.error(f"阿里云情感分析失败: {e}")
            with self._stats_lock:
                self.stats['api_errors'] += 1
            
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'error': str(e),
                'method': 'aliyun'
            }
    
    def _analyze_texts_concurrent(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """并发批量情感分析"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        n = len(texts)
        if n == 0:
            return []
        
        # 去重映射
        unique_index: Dict[str, int] = {}
        order_to_text: Dict[int, str] = {}
        for i, t in enumerate(texts):
            if t not in unique_index:
                unique_index[t] = len(unique_index)
            order_to_text[i] = t
        
        unique_list: List[str] = [None] * len(unique_index)
        for t, u in unique_index.items():
            unique_list[u] = t
        
        unique_results: List[Dict[str, Union[str, float]]] = [None] * len(unique_list)
        
        def _process_range(start: int, end: int):
            logger.info(f"情感API {start+1}-{end}/{len(unique_list)}")
            with ThreadPoolExecutor(max_workers=self.sa_concurrency) as ex:
                futures = {}
                for idx in range(start, end):
                    # 可选节流
                    if self.sa_throttle_ms > 0 and (idx - start) % self.sa_concurrency == 0:
                        time.sleep(self.sa_throttle_ms / 1000.0)
                    futures[ex.submit(self.analyze_sentiment, unique_list[idx])] = idx
                
                for fut in as_completed(futures):
                    uid = futures[fut]
                    try:
                        unique_results[uid] = fut.result()
                    except Exception as e:
                        unique_results[uid] = {
                            'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0,
                            'error': str(e), 'method': 'aliyun'
                        }
        
        # 分批处理
        for i in range(0, len(unique_list), self.sa_batch_size):
            _process_range(i, min(i + self.sa_batch_size, len(unique_list)))
        
        # 回填结果
        result_map: Dict[str, Dict[str, Union[str, float]]] = {
            t: unique_results[u] for t, u in unique_index.items()
        }
        return [result_map[order_to_text[i]] for i in range(n)]
    
    def calculate_sentiment_conformity_score(self, data: Union[pd.DataFrame, List[Dict]]) -> Union[pd.DataFrame, List[Dict]]:
        """
        计算情感从众心理分数
        
        Args:
            data: 包含评论数据的DataFrame或字典列表，需要包含以下字段：
                - content: 评论内容
                - comment_id: 评论ID
                - is_parent: 父评论标识（可选，如果数据已通过数据清洗模块处理）
                
        Returns:
            Union[pd.DataFrame, List[Dict]]: 包含情感从众心理分数的数据
        """
        logger.info("开始计算情感从众心理分数...")
        
        # 处理输入数据格式
        if isinstance(data, list):
            df = pd.DataFrame(data)
            return_dict = True
        else:
            df = data.copy()
            return_dict = False
        
        # 检查数据是否已经过处理
        if 'is_parent' in df.columns:
            logger.info("检测到已处理的数据，直接使用")
        else:
            logger.info("数据未经过处理，开始处理...")
            
            # 确保必要的列存在
            if 'content' not in df.columns:
                raise ValueError("数据中缺少content字段")
            if 'comment_id' not in df.columns:
                raise ValueError("数据中缺少comment_id字段")
            
            # 假设第一个评论是父评论
            df['is_parent'] = False
            df.loc[0, 'is_parent'] = True
            logger.info("假设第一个评论为父评论")
        
        # 分离父评论和子评论
        parent_data = df[df['is_parent'] == True]
        child_data = df[df['is_parent'] == False]
        
        if len(parent_data) == 0:
            raise ValueError("未找到父评论数据")
        
        parent_comment = parent_data.iloc[0]
        parent_id = parent_comment['comment_id']
        parent_content = parent_comment['content']
        
        logger.info(f"父评论ID: {parent_id}")
        logger.info(f"子评论数量: {len(child_data)}")
        
        # 收集所有需要分析的文本
        all_texts = [parent_content]
        for _, row in child_data.iterrows():
            all_texts.append(row['content'])
        
        # 并发进行情感分析
        logger.info("开始并发情感分析...")
        sentiment_results = self._analyze_texts_concurrent(all_texts)
        
        # 提取父评论情感结果
        parent_sentiment = sentiment_results[0]
        parent_score = parent_sentiment['score']
        parent_sentiment_type = parent_sentiment['sentiment']
        
        logger.info(f"父评论情感: {parent_sentiment_type}, 分数: {parent_score:.4f}")
        
        # 计算每个子评论的情感从众心理分数
        sentiment_differences = []
        conformity_scores = []
        
        for i, (_, row) in enumerate(child_data.iterrows()):
            child_sentiment = sentiment_results[i + 1]  # +1 因为第一个是父评论
            child_score = child_sentiment['score']
            
            # 计算情感差异的绝对值
            sentiment_diff = abs(parent_score - child_score)
            sentiment_differences.append(sentiment_diff)
        
        # 计算自适应归一化参数
        if sentiment_differences:
            # 使用中位数作为参考点
            median_diff = np.median(sentiment_differences)
            max_diff = np.max(sentiment_differences)
            
            # 如果差异范围很大，使用对数缩放
            if max_diff > median_diff * 10:
                # 使用对数缩放处理大范围差异
                log_differences = [np.log1p(diff) for diff in sentiment_differences]
                normalized_scores = self.normalizer.fit_transform(
                    np.array(log_differences).reshape(-1, 1)
                ).flatten()
                conformity_scores = 1 - normalized_scores
                logger.info("使用对数缩放处理大范围情感差异")
            else:
                # 直接归一化差异
                normalized_scores = self.normalizer.fit_transform(
                    np.array(sentiment_differences).reshape(-1, 1)
                ).flatten()
                conformity_scores = 1 - normalized_scores
                logger.info("使用直接归一化处理情感差异")
        else:
            logger.warning("未找到子评论数据")
            conformity_scores = []
        
        # 为父评论添加分数（设为1.0，表示完全从众）
        parent_conformity_score = 1.0
        
        # 将结果添加到DataFrame
        df['sentiment_difference'] = 0.0  # 父评论的差异为0
        df['raw_sentiment_conformity_score'] = 0.0  # 父评论的原始分数为0
        df['normalized_sentiment_conformity_score'] = 0.0  # 父评论的归一化分数为0
        df['sentiment_type'] = 'neutral'  # 默认情感类型
        df['sentiment_score'] = 0.0  # 默认情感分数
        df['sentiment_confidence'] = 0.0  # 默认置信度
        
        # 更新父评论的情感信息
        parent_idx = parent_data.index[0]
        df.loc[parent_idx, 'sentiment_type'] = parent_sentiment_type
        df.loc[parent_idx, 'sentiment_score'] = parent_score
        df.loc[parent_idx, 'sentiment_confidence'] = parent_sentiment['confidence']
        df.loc[parent_idx, 'normalized_sentiment_conformity_score'] = parent_conformity_score
        
        # 更新子评论的分数
        child_indices = child_data.index
        for i, idx in enumerate(child_indices):
            child_sentiment = sentiment_results[i + 1]
            df.loc[idx, 'sentiment_difference'] = sentiment_differences[i]
            df.loc[idx, 'raw_sentiment_conformity_score'] = sentiment_differences[i]
            df.loc[idx, 'normalized_sentiment_conformity_score'] = conformity_scores[i]
            df.loc[idx, 'sentiment_type'] = child_sentiment['sentiment']
            df.loc[idx, 'sentiment_score'] = child_sentiment['score']
            df.loc[idx, 'sentiment_confidence'] = child_sentiment['confidence']
        
        # 添加情感分类
        df['sentiment_category'] = self._categorize_sentiment_differences(df['sentiment_difference'].tolist())
        
        # 更新统计信息
        with self._stats_lock:
            self.stats['total_analyzed'] += len(df)
            self.stats['parent_comments'] += 1
            self.stats['child_comments'] += len(child_data)
        
        logger.info(f"情感从众心理分数计算完成，共处理 {len(df)} 条评论")
        logger.info(f"父评论情感: {parent_sentiment_type}, 子评论数量: {len(child_data)}")
        
        if return_dict:
            return df.to_dict('records')
        else:
            return df
    
    def _categorize_sentiment_differences(self, sentiment_differences: List[float]) -> List[str]:
        """
        对情感差异进行分类
        
        Args:
            sentiment_differences: 情感差异列表
            
        Returns:
            List[str]: 情感分类列表
        """
        categories = []
        for diff in sentiment_differences:
            if diff == 0:  # 父评论
                categories.append('父评论')
            elif diff <= 0.1:  # 差异很小
                categories.append('高度情感从众')
            elif diff <= 0.3:  # 差异较小
                categories.append('中度情感从众')
            elif diff <= 0.5:  # 差异中等
                categories.append('轻度情感从众')
            elif diff <= 0.8:  # 差异较大
                categories.append('低度情感从众')
            else:  # 差异很大
                categories.append('非情感从众')
        
        return categories
    
    def calculate_parent_sentiment_conformity_score(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """
        计算父评论的总体情感从众心理分数
        
        Args:
            data: 包含评论数据的DataFrame或字典列表
            
        Returns:
            Dict: 父评论情感从众心理分数统计结果
        """
        logger.info("开始计算父评论情感从众心理分数...")
        
        # 计算情感从众心理分数
        result_data = self.calculate_sentiment_conformity_score(data)
        
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
        parent_sentiment_score = parent_comment['sentiment_score']
        parent_sentiment_type = parent_comment['sentiment_type']
        
        # 计算子评论的情感从众心理分数统计
        child_scores = child_data['normalized_sentiment_conformity_score'].tolist()
        child_sentiment_diffs = child_data['sentiment_difference'].tolist()
        
        if len(child_scores) == 0:
            logger.warning("未找到子评论数据")
            return {
                'parent_comment_id': parent_id,
                'parent_sentiment_score': parent_sentiment_score,
                'child_comment_count': 0,
                'parent_sentiment_conformity_score': 0.0,
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
        
        # 情感差异统计
        avg_sentiment_diff = np.mean(child_sentiment_diffs)
        median_sentiment_diff = np.median(child_sentiment_diffs)
        min_sentiment_diff = np.min(child_sentiment_diffs)
        max_sentiment_diff = np.max(child_sentiment_diffs)
        
        # 情感分类统计
        sentiment_categories = child_data['sentiment_category'].value_counts()
        
        result = {
            'parent_comment_id': parent_id,
            'parent_sentiment_score': parent_sentiment_score,
            'parent_sentiment_type': parent_sentiment_type,
            'child_comment_count': len(child_scores),
            'parent_sentiment_conformity_score': avg_score,  # 主要分数：平均值
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
                'sentiment_category_distribution': sentiment_categories.to_dict()
            },
            'sentiment_analysis': {
                'avg_sentiment_difference': avg_sentiment_diff,
                'median_sentiment_difference': median_sentiment_diff,
                'min_sentiment_difference': min_sentiment_diff,
                'max_sentiment_difference': max_sentiment_diff
            },
            'parent_info': {
                'sentiment_score': parent_sentiment_score,
                'sentiment_type': parent_sentiment_type,
                'comment_id': parent_id
            }
        }
        
        logger.info(f"父评论情感从众心理分数计算完成: {parent_id}, 分数: {avg_score:.4f}")
        
        return result
    
    def analyze_sentiment_conformity_patterns(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        分析情感从众心理模式
        
        Args:
            data: 包含情感从众心理分数的DataFrame
            
        Returns:
            Dict: 情感从众心理模式分析结果
        """
        logger.info("开始分析情感从众心理模式...")
        
        analysis_results = {}
        
        # 1. 整体统计
        analysis_results['overall_stats'] = {
            'total_comments': len(data),
            'parent_comments': len(data[data['is_parent'] == True]),
            'child_comments': len(data[data['is_parent'] == False]),
            'avg_sentiment_conformity_score': float(data['normalized_sentiment_conformity_score'].mean()),
            'median_sentiment_conformity_score': float(data['normalized_sentiment_conformity_score'].median()),
            'std_sentiment_conformity_score': float(data['normalized_sentiment_conformity_score'].std())
        }
        
        # 2. 情感从众心理相关性分析
        if len(data) > 1:
            sentiment_scores = data['sentiment_score'].values
            conformity_scores = data['normalized_sentiment_conformity_score'].values
            correlation = np.corrcoef(sentiment_scores, conformity_scores)[0, 1]
            analysis_results['sentiment_conformity_correlation'] = float(correlation)
        else:
            analysis_results['sentiment_conformity_correlation'] = 0.0
        
        # 3. 高从众心理分析
        high_conformity_threshold = 0.7
        high_conformity_data = data[data['normalized_sentiment_conformity_score'] >= high_conformity_threshold]
        analysis_results['high_conformity_analysis'] = {
            'count': len(high_conformity_data),
            'percentage': len(high_conformity_data) / len(data) * 100 if len(data) > 0 else 0,
            'avg_sentiment_score': float(high_conformity_data['sentiment_score'].mean()) if len(high_conformity_data) > 0 else 0.0
        }
        
        # 4. 情感类型分布
        sentiment_type_dist = data['sentiment_type'].value_counts()
        analysis_results['sentiment_type_distribution'] = sentiment_type_dist.to_dict()
        
        # 5. 情感分类分布
        sentiment_category_dist = data['sentiment_category'].value_counts()
        analysis_results['sentiment_category_distribution'] = sentiment_category_dist.to_dict()
        
        logger.info("情感从众心理模式分析完成")
        
        return analysis_results
    
    def generate_sentiment_conformity_report(self, data: pd.DataFrame, 
                                           analysis_results: Dict[str, any]) -> str:
        """
        生成情感从众心理分析报告
        
        Args:
            data: 包含分析结果的DataFrame
            analysis_results: 分析结果字典
            
        Returns:
            str: Markdown格式的分析报告
        """
        logger.info("生成情感从众心理分析报告...")
        
        # 获取父评论信息
        parent_data = data[data['is_parent'] == True]
        if len(parent_data) > 0:
            parent_comment = parent_data.iloc[0]
            parent_id = parent_comment['comment_id']
            parent_content = parent_comment['content']
            parent_sentiment = parent_comment['sentiment_type']
            parent_score = parent_comment['sentiment_score']
        else:
            parent_id = "未知"
            parent_content = "未知"
            parent_sentiment = "未知"
            parent_score = 0.0
        
        # 生成报告
        report = f"""# 情感从众心理分析报告

## 📊 分析概览

- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **父评论ID**: {parent_id}
- **父评论内容**: {parent_content}
- **父评论情感**: {parent_sentiment} (分数: {parent_score:.4f})
- **总评论数**: {analysis_results['overall_stats']['total_comments']}
- **子评论数**: {analysis_results['overall_stats']['child_comments']}

## 🎯 情感从众心理分析结果

### 整体统计
- **平均情感从众心理分数**: {analysis_results['overall_stats']['avg_sentiment_conformity_score']:.4f}
- **中位数情感从众心理分数**: {analysis_results['overall_stats']['median_sentiment_conformity_score']:.4f}
- **标准差**: {analysis_results['overall_stats']['std_sentiment_conformity_score']:.4f}

### 高从众心理分析
- **高从众评论数量**: {analysis_results['high_conformity_analysis']['count']}
- **高从众比例**: {analysis_results['high_conformity_analysis']['percentage']:.1f}%
- **高从众评论平均情感分数**: {analysis_results['high_conformity_analysis']['avg_sentiment_score']:.4f}

### 情感相关性
- **情感分数与从众心理相关性**: {analysis_results['sentiment_conformity_correlation']:.4f}

## 📈 情感类型分布

"""
        
        # 添加情感类型分布
        for sentiment_type, count in analysis_results['sentiment_type_distribution'].items():
            percentage = count / analysis_results['overall_stats']['total_comments'] * 100
            report += f"- **{sentiment_type}**: {count} 条 ({percentage:.1f}%)\n"
        
        report += "\n## 🏷️ 情感从众心理分类分布\n\n"
        
        # 添加情感分类分布
        for category, count in analysis_results['sentiment_category_distribution'].items():
            percentage = count / analysis_results['overall_stats']['total_comments'] * 100
            report += f"- **{category}**: {count} 条 ({percentage:.1f}%)\n"
        
        # 添加前5名高从众心理评论
        high_conformity_comments = data[data['is_parent'] == False].nlargest(5, 'normalized_sentiment_conformity_score')
        if len(high_conformity_comments) > 0:
            report += "\n## ⭐ 前5名高情感从众心理评论\n\n"
            for i, (_, row) in enumerate(high_conformity_comments.iterrows(), 1):
                report += f"{i}. **评论ID**: {row['comment_id']}\n"
                report += f"   - **内容**: {row['content']}\n"
                report += f"   - **情感从众心理分数**: {row['normalized_sentiment_conformity_score']:.4f}\n"
                report += f"   - **情感类型**: {row['sentiment_type']}\n"
                report += f"   - **情感分数**: {row['sentiment_score']:.4f}\n"
                report += f"   - **情感差异**: {row['sentiment_difference']:.4f}\n\n"
        
        report += f"\n---\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['api_calls'] > 0:
            stats['avg_confidence'] = stats['total_confidence'] / stats['api_calls']
            stats['avg_sentiment_score'] = stats['total_sentiment_score'] / stats['api_calls']
        else:
            stats['avg_confidence'] = 0.0
            stats['avg_sentiment_score'] = 0.0
        return stats
    
    def analyze(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """
        实现BaseAnalyzer的analyze方法
        """
        logger.info("开始情感从众心理分析...")
        
        # 计算情感从众心理分数
        processed_data = self.calculate_sentiment_conformity_score(data)
        if isinstance(processed_data, list):
            processed_df = pd.DataFrame(processed_data)
        else:
            processed_df = processed_data
        
        # 计算父评论总体情感从众心理分数
        parent_overall_result = self.calculate_parent_sentiment_conformity_score(processed_df)
        
        # 分析情感从众心理模式
        pattern_results = self.analyze_sentiment_conformity_patterns(processed_df)
        
        # 生成报告
        report_content = self.generate_sentiment_conformity_report(processed_df, pattern_results)
        
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
            "parent_environment_sentiment_analysis": parent_overall_result,
            "sentiment_classification": pattern_results.get('sentiment_category_distribution', {}),
            "top_high_sentiment_comments": processed_df.nlargest(10, 'normalized_sentiment_conformity_score').to_dict('records') if not processed_df.empty else []
        }
        
        logger.info("情感从众心理分析完成")
        return output_result

def main():
    """主函数"""
    parser = create_parser("sentiment_conformity_analyzer", "情感从众心理分析")
    parser.add_argument('--sa-concurrency', type=int, default=8, help='情感API并发数，默认8')
    parser.add_argument('--sa-batch-size', type=int, default=200, help='情感API批大小，默认200')
    parser.add_argument('--sa-throttle-ms', type=int, default=0, help='情感API节流毫秒，默认0=不限制')
    
    args = parser.parse_args()
    args = parse_common_args(parser, args)
    
    # 检查阿里云API配置
    if not is_aliyun_api_available():
        logger.error("阿里云API配置缺失，请设置环境变量")
        return
    
    # 创建分析器
    analyzer = SentimentConformityAnalyzer(
        module_name="sentiment_conformity_analyzer",
        video_id=args.video_id,
        sa_concurrency=args.sa_concurrency,
        sa_batch_size=args.sa_batch_size,
        sa_throttle_ms=args.sa_throttle_ms
    )
    
    # 加载数据
    if args.use_cleaned_data:
        logger.info(f"使用清洗后的数据: {args.cleaned_data_path}")
        data = analyzer.load_data(use_cleaned_data=True, cleaned_data_path=args.cleaned_data_path)
    else:
        logger.info("从数据库加载数据...")
        data = analyzer.load_data(limit=args.limit)
    
    if data is None or len(data) == 0:
        logger.error("没有找到数据")
        return
    
    # 运行分析
    try:
        # 计算情感从众心理分数
        result_data = analyzer.calculate_sentiment_conformity_score(data)
        
        # 计算父评论情感从众心理分数
        parent_result = analyzer.calculate_parent_sentiment_conformity_score(data)
        
        # 分析模式
        analysis_results = analyzer.analyze_sentiment_conformity_patterns(result_data)
        
        # 生成报告
        report = analyzer.generate_sentiment_conformity_report(result_data, analysis_results)
        
        # 保存结果
        analyzer.save_results(result_data, parent_result, analysis_results, report)
        
        logger.info("情感从众心理分析完成")
        
    except Exception as e:
        logger.error(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
