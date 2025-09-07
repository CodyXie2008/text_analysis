#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相似度从众心理分析模块
基于阿里云文本向量API计算父子评论相似度，分析从众心理
通过计算父评论与子评论的文本相似度，相似度越高，从众心理分数越高
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入基础分析器和API管理器
from text_analysis.core.base_analyzer import BaseAnalyzer
from text_analysis.core.aliyun_api_manager import get_aliyun_api_manager
from text_analysis.algorithms.normalization import MinMaxNormalizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityConformityAnalyzer(BaseAnalyzer):
    """相似度从众心理分析器"""
    
    def __init__(self, module_name: str = "similarity_conformity_analyzer", 
                 video_id: Optional[str] = None, **kwargs):
        super().__init__(module_name=module_name, video_id=video_id)
        
        # 初始化阿里云API管理器
        self.aliyun_api_manager = get_aliyun_api_manager()
        if not self.aliyun_api_manager:
            raise ValueError("阿里云API配置缺失，请设置 ALIYUN_ACCESS_KEY_ID 和 ALIYUN_ACCESS_KEY_SECRET")
        
        # 初始化归一化器
        self.normalizer = MinMaxNormalizer(feature_range=(0, 1))
        
        # 分析结果存储
        self.results = {}
        
        # 统计信息
        self._stats_lock = threading.Lock()
        self.stats = {
            'api_calls': 0,
            'api_errors': 0,
            'total_similarity': 0.0,
            'total_confidence': 0.0,
            'processed_comments': 0
        }
        
        logger.info("✅ 相似度从众心理分析器初始化成功")
    
    def get_text_vector(self, text: str) -> List[float]:
        """
        使用阿里云API获取文本向量
        
        Args:
            text: 待向量化的文本
            
        Returns:
            List[float]: 文本向量
        """
        if not text or not text.strip():
            return []
        
        try:
            vector = self.aliyun_api_manager.get_text_vector(text)
            
            with self._stats_lock:
                self.stats['api_calls'] += 1
            
            return vector
        except Exception as e:
            logger.error(f"获取文本向量失败: {e}")
            with self._stats_lock:
                self.stats['api_errors'] += 1
            return []
    
    def _get_vectors_concurrent(self, texts: List[str], batch_size: int = 50, 
                              concurrency: int = 8, throttle_ms: int = 0) -> List[List[float]]:
        """
        并发获取文本向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            concurrency: 并发数
            throttle_ms: 节流毫秒数
            
        Returns:
            List[List[float]]: 向量列表
        """
        vectors = [None] * len(texts)
        
        def _process_range(start: int, end: int):
            """处理指定范围的文本"""
            for i in range(start, end):
                if i < len(texts):
                    vectors[i] = self.get_text_vector(texts[i])
                    if throttle_ms > 0:
                        time.sleep(throttle_ms / 1000.0)
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                future = executor.submit(_process_range, i, min(i + batch_size, len(texts)))
                futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"向量化处理失败: {e}")
        
        return vectors
    
    def calculate_similarity_score(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            float: 相似度分数 [0, 1]
        """
        if not vector1 or not vector2 or len(vector1) != len(vector2):
            return 0.0
        
        try:
            # 转换为numpy数组
            v1 = np.array(vector1).reshape(1, -1)
            v2 = np.array(vector2).reshape(1, -1)
            
            # 计算余弦相似度
            similarity = cosine_similarity(v1, v2)[0][0]
            
            # 确保相似度在[0, 1]范围内
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def calculate_similarity_conformity_score(self, data: Union[pd.DataFrame, List[Dict]]) -> Union[pd.DataFrame, List[Dict]]:
        """
        计算相似度从众心理分数
        
        Args:
            data: 评论数据
            
        Returns:
            包含相似度从众心理分数的数据
        """
        logger.info("开始计算相似度从众心理分数...")
        
        # 转换为DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        if df.empty:
            logger.warning("数据为空，无法计算相似度从众心理分数")
            return df.to_dict('records') if isinstance(data, list) else df
        
        # 识别父评论和子评论
        parent_comments = df[df['parent_comment_id'] == '0']
        child_comments = df[df['parent_comment_id'] != '0']
        
        if parent_comments.empty:
            logger.warning("未找到父评论")
            return df.to_dict('records') if isinstance(data, list) else df
        
        # 获取父评论
        parent_comment = parent_comments.iloc[0]
        parent_content = parent_comment.get('content', '')
        parent_comment_id = parent_comment.get('comment_id', '')
        
        logger.info(f"父评论ID: {parent_comment_id}")
        logger.info(f"子评论数量: {len(child_comments)}")
        
        # 获取父评论向量
        logger.info("获取父评论向量...")
        parent_vector = self.get_text_vector(parent_content)
        if not parent_vector:
            logger.error("父评论向量获取失败")
            return df.to_dict('records') if isinstance(data, list) else df
        
        # 获取所有子评论的文本
        child_texts = child_comments['content'].tolist()
        
        # 并发获取子评论向量
        logger.info(f"并发获取 {len(child_texts)} 个子评论向量...")
        child_vectors = self._get_vectors_concurrent(child_texts, batch_size=50, concurrency=8)
        
        # 计算相似度分数
        logger.info("计算相似度分数...")
        similarity_scores = []
        for i, child_vector in enumerate(child_vectors):
            if child_vector:
                similarity = self.calculate_similarity_score(parent_vector, child_vector)
                similarity_scores.append(similarity)
            else:
                similarity_scores.append(0.0)
        
        # 归一化相似度分数
        if similarity_scores:
            normalized_scores = self.normalizer.fit_transform(
                np.array(similarity_scores).reshape(-1, 1)
            ).flatten()
            logger.info("相似度分数归一化完成")
        else:
            normalized_scores = []
        
        # 将结果添加到DataFrame
        df['similarity_score'] = 0.0
        df['normalized_similarity_conformity_score'] = 0.0
        
        # 设置父评论的相似度分数为1.0（与自己的相似度）
        parent_mask = df['comment_id'] == parent_comment_id
        df.loc[parent_mask, 'similarity_score'] = 1.0
        df.loc[parent_mask, 'normalized_similarity_conformity_score'] = 1.0
        
        # 设置子评论的相似度分数
        for i, (idx, row) in enumerate(child_comments.iterrows()):
            if i < len(normalized_scores):
                df.loc[idx, 'similarity_score'] = similarity_scores[i]
                df.loc[idx, 'normalized_similarity_conformity_score'] = normalized_scores[i]
        
        logger.info(f"相似度从众心理分数计算完成，共处理 {len(df)} 条评论")
        
        return df.to_dict('records') if isinstance(data, list) else df
    
    def calculate_parent_similarity_conformity_score(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """
        计算父评论的总体相似度从众心理分数
        
        Args:
            data: 包含相似度分数的评论数据
            
        Returns:
            Dict: 父评论相似度从众心理分析结果
        """
        logger.info("计算父评论相似度从众心理分数...")
        
        # 转换为DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        if df.empty:
            return {}
        
        # 确保 'normalized_similarity_conformity_score' 列存在
        if 'normalized_similarity_conformity_score' not in df.columns:
            logger.info("缺少相似度从众心理分数列，先计算相似度从众心理分数...")
            df = self.calculate_similarity_conformity_score(df)
            if isinstance(df, list):
                df = pd.DataFrame(df)
        
        # 识别父评论和子评论
        parent_comments = df[df['parent_comment_id'] == '0']
        child_comments = df[df['parent_comment_id'] != '0']
        
        if parent_comments.empty:
            return {}
        
        parent_comment = parent_comments.iloc[0]
        parent_comment_id = parent_comment.get('comment_id', '')
        
        # 获取子评论的相似度分数
        child_similarity_scores = child_comments['normalized_similarity_conformity_score'].tolist()
        
        if not child_similarity_scores:
            return {
                'parent_comment_id': parent_comment_id,
                'parent_similarity_conformity_score': 0.0,
                'child_comment_count': 0,
                'statistics': {},
                'similarity_distribution': {},
                'similarity_analysis': {}
            }
        
        # 计算统计信息
        mean_score = np.mean(child_similarity_scores)
        median_score = np.median(child_similarity_scores)
        std_score = np.std(child_similarity_scores)
        min_score = np.min(child_similarity_scores)
        max_score = np.max(child_similarity_scores)
        
        # 计算高相似度比例（相似度 > 0.7）
        high_similarity_count = sum(1 for score in child_similarity_scores if score > 0.7)
        high_similarity_ratio = high_similarity_count / len(child_similarity_scores)
        
        # 计算父评论总体相似度从众心理分数（子评论相似度分数的平均值）
        parent_similarity_conformity_score = mean_score
        
        # 相似度分布统计
        similarity_distribution = {
            'high_similarity_count': high_similarity_count,
            'high_similarity_ratio': high_similarity_ratio,
            'medium_similarity_count': sum(1 for score in child_similarity_scores if 0.3 <= score <= 0.7),
            'low_similarity_count': sum(1 for score in child_similarity_scores if score < 0.3)
        }
        
        # 相似度分析
        similarity_analysis = {
            'mean_similarity': mean_score,
            'median_similarity': median_score,
            'std_similarity': std_score,
            'min_similarity': min_score,
            'max_similarity': max_score
        }
        
        result = {
            'parent_comment_id': parent_comment_id,
            'parent_similarity_conformity_score': parent_similarity_conformity_score,
            'child_comment_count': len(child_comments),
            'statistics': {
                'mean_score': mean_score,
                'median_score': median_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score
            },
            'similarity_distribution': similarity_distribution,
            'similarity_analysis': similarity_analysis
        }
        
        logger.info(f"父评论相似度从众心理分数计算完成: {parent_comment_id}, 分数: {parent_similarity_conformity_score:.4f}")
        
        return result
    
    def _categorize_similarity_scores(self, similarity_scores: List[float]) -> List[str]:
        """
        根据相似度分数对评论进行分类
        
        Args:
            similarity_scores: 相似度分数列表
            
        Returns:
            List[str]: 分类结果列表
        """
        categories = []
        for score in similarity_scores:
            if score >= 0.8:
                categories.append("高度相似")
            elif score >= 0.6:
                categories.append("中度相似")
            elif score >= 0.4:
                categories.append("轻度相似")
            elif score >= 0.2:
                categories.append("低度相似")
            else:
                categories.append("非相似")
        return categories
    
    def analyze_similarity_conformity_patterns(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        分析相似度从众心理模式
        
        Args:
            data: 包含相似度分数的数据
            
        Returns:
            Dict: 相似度从众心理模式分析结果
        """
        logger.info("开始分析相似度从众心理模式...")
        
        if data.empty:
            return {}
        
        # 获取子评论的相似度分数
        child_comments = data[data['parent_comment_id'] != '0']
        if child_comments.empty:
            return {}
        
        similarity_scores = child_comments['normalized_similarity_conformity_score'].tolist()
        
        # 分类统计
        categories = self._categorize_similarity_scores(similarity_scores)
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # 计算分类分布
        total_count = len(categories)
        category_distribution = {k: v / total_count for k, v in category_counts.items()}
        
        result = {
            'similarity_category_distribution': category_distribution,
            'similarity_category_counts': category_counts,
            'total_analyzed_comments': total_count
        }
        
        logger.info("相似度从众心理模式分析完成")
        return result
    
    def generate_similarity_conformity_report(self, data: pd.DataFrame, 
                                            pattern_results: Dict[str, any]) -> str:
        """
        生成相似度从众心理分析报告
        
        Args:
            data: 分析数据
            pattern_results: 模式分析结果
            
        Returns:
            str: 分析报告
        """
        logger.info("生成相似度从众心理分析报告...")
        
        # 获取父评论信息
        parent_comments = data[data['parent_comment_id'] == '0']
        if parent_comments.empty:
            return "未找到父评论"
        
        parent_comment = parent_comments.iloc[0]
        parent_comment_id = parent_comment.get('comment_id', '')
        parent_content = parent_comment.get('content', '')
        
        # 计算父评论总体分数
        parent_result = self.calculate_parent_similarity_conformity_score(data)
        parent_score = parent_result.get('parent_similarity_conformity_score', 0.0)
        
        # 生成报告
        report = f"""
# 相似度从众心理分析报告

## 基本信息
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **父评论ID**: {parent_comment_id}
- **父评论内容**: {parent_content[:100]}...
- **子评论数量**: {parent_result.get('child_comment_count', 0)}

## 父评论相似度从众心理分数
- **总体分数**: {parent_score:.4f}
- **从众心理水平**: {"高从众" if parent_score > 0.7 else "中等从众" if parent_score > 0.4 else "低从众"}

## 统计信息
- **平均相似度**: {parent_result.get('statistics', {}).get('mean_score', 0.0):.4f}
- **中位数相似度**: {parent_result.get('statistics', {}).get('median_score', 0.0):.4f}
- **标准差**: {parent_result.get('statistics', {}).get('std_score', 0.0):.4f}
- **最小相似度**: {parent_result.get('statistics', {}).get('min_score', 0.0):.4f}
- **最大相似度**: {parent_result.get('statistics', {}).get('max_score', 0.0):.4f}

## 相似度分布
- **高相似度比例**: {parent_result.get('similarity_distribution', {}).get('high_similarity_ratio', 0.0):.1%}
- **高相似度数量**: {parent_result.get('similarity_distribution', {}).get('high_similarity_count', 0)}
- **中等相似度数量**: {parent_result.get('similarity_distribution', {}).get('medium_similarity_count', 0)}
- **低相似度数量**: {parent_result.get('similarity_distribution', {}).get('low_similarity_count', 0)}

## 分类分布
"""
        
        # 添加分类分布信息
        category_distribution = pattern_results.get('similarity_category_distribution', {})
        for category, ratio in category_distribution.items():
            report += f"- **{category}**: {ratio:.1%}\n"
        
        report += f"""
## 分析结论
基于文本相似度分析，该父评论环境的从众心理特征如下：

1. **相似度水平**: {parent_score:.4f} ({'高从众' if parent_score > 0.7 else '中等从众' if parent_score > 0.4 else '低从众'})
2. **一致性程度**: {parent_result.get('similarity_distribution', {}).get('high_similarity_ratio', 0.0):.1%} 的子评论与父评论高度相似
3. **环境影响**: {'强影响' if parent_result.get('similarity_distribution', {}).get('high_similarity_ratio', 0.0) > 0.6 else '中等影响' if parent_result.get('similarity_distribution', {}).get('high_similarity_ratio', 0.0) > 0.3 else '弱影响'}

## 技术统计
- **API调用次数**: {self.stats['api_calls']}
- **API错误次数**: {self.stats['api_errors']}
- **处理评论数**: {self.stats['processed_comments']}
"""
        
        logger.info("相似度从众心理分析报告生成完成")
        return report
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self._stats_lock:
            return self.stats.copy()

    def analyze(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """
        实现BaseAnalyzer的analyze方法
        """
        logger.info("开始相似度从众心理分析...")
        
        # 计算相似度从众心理分数
        processed_data = self.calculate_similarity_conformity_score(data)
        if isinstance(processed_data, list):
            processed_df = pd.DataFrame(processed_data)
        else:
            processed_df = processed_data

        # 计算父评论总体相似度从众心理分数
        parent_overall_result = self.calculate_parent_similarity_conformity_score(processed_df)

        # 分析相似度从众心理模式
        pattern_results = self.analyze_similarity_conformity_patterns(processed_df)

        # 生成报告
        report_content = self.generate_similarity_conformity_report(processed_df, pattern_results)

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
            "parent_environment_similarity_analysis": parent_overall_result,
            "similarity_classification": pattern_results.get('similarity_category_distribution', {}),
            "top_high_similarity_comments": processed_df.nlargest(10, 'normalized_similarity_conformity_score').to_dict('records') if not processed_df.empty else []
        }

        logger.info("相似度从众心理分析完成")
        return output_result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='相似度从众心理分析')
    parser.add_argument('--data-file', type=str, help='数据文件路径')
    parser.add_argument('--output-dir', type=str, default='data', help='输出目录')
    parser.add_argument('--vector-concurrency', type=int, default=8, help='向量API并发数，默认8')
    parser.add_argument('--vector-batch-size', type=int, default=50, help='向量API批大小，默认50')
    parser.add_argument('--vector-throttle-ms', type=int, default=0, help='向量API节流毫秒，默认0=不限制')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = SimilarityConformityAnalyzer(
        output_dir=args.output_dir,
        vector_concurrency=args.vector_concurrency,
        vector_batch_size=args.vector_batch_size,
        vector_throttle_ms=args.vector_throttle_ms
    )
    
    # 加载数据
    if args.data_file:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 使用默认测试数据
        data = [
            {
                'comment_id': '1',
                'parent_comment_id': '0',
                'content': '这是一个测试父评论',
                'create_time': 1234567890
            },
            {
                'comment_id': '2',
                'parent_comment_id': '1',
                'content': '这是一个相似的子评论',
                'create_time': 1234567891
            }
        ]
    
    # 运行分析
    result_data = analyzer.calculate_similarity_conformity_score(data)
    parent_result = analyzer.calculate_parent_similarity_conformity_score(result_data)
    
    # 分析模式
    df = pd.DataFrame(result_data) if isinstance(result_data, list) else result_data
    pattern_results = analyzer.analyze_similarity_conformity_patterns(df)
    
    # 生成报告
    report = analyzer.generate_similarity_conformity_report(df, pattern_results)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, f'similarity_conformity_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    result = {
        'parent_result': parent_result,
        'pattern_results': pattern_results,
        'report': report,
        'stats': analyzer.get_stats()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"分析完成，结果保存到: {output_file}")
    print(f"父评论相似度从众心理分数: {parent_result.get('parent_similarity_conformity_score', 0.0):.4f}")


if __name__ == "__main__":
    main()
