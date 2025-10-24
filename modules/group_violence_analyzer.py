#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络暴力强度分析模块
根据建模思路构建GroupViolence指数，采用词典法计算评论暴力程度
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

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.base_analyzer import BaseAnalyzer, create_parser, parse_common_args
from core.data_paths import AnalysisPathManager, PROJECT_ROOT, DATA_DIRS
from algorithms.normalization import MinMaxNormalizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GroupViolenceAnalyzer(BaseAnalyzer):
    """网络暴力强度分析器"""
    
    def __init__(self, video_id: Optional[str] = None):
        """
        初始化网络暴力强度分析器
        
        Args:
            video_id: 可选的视频ID
        """
        super().__init__("group_violence", video_id)
        
        # 初始化输出目录
        self.output_dir = DATA_DIRS['reports']
        
        # 初始化归一化器
        self.normalizer = MinMaxNormalizer(feature_range=(0, 1))
        
        # 从文件加载网络暴力词库
        self.violence_words = self._load_violence_words()
        logger.info(f"✅ 已加载 {len(self.violence_words)} 个暴力词汇")
        
        # 并发控制参数
        self.concurrency = 8
        self.batch_size = 200
        self._stats_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_threads': 0,
            'processed_threads': 0,
            'average_group_violence': 0.0,
            'max_group_violence': 0.0,
            'min_group_violence': 1.0
        }
        
        # 结果存储
        self.results = {}
        
        # 预处理数据目录
        self.processed_dir = DATA_DIRS['processed']
        logger.info("✅ 网络暴力强度分析器初始化成功")
    
    def _load_violence_words(self) -> List[str]:
        """
        从txt文件加载暴力词汇
        
        Returns:
            List[str]: 去重后的暴力词汇列表
        """
        # 获取词汇文件路径
        words_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'violence_words.txt')
        violence_words = []
        
        try:
            with open(words_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # 去除注释和空行
                    line = line.strip()
                    if line and not line.startswith('#'):
                        violence_words.append(line)
            
            # 去重
            violence_words = list(set(violence_words))
            logger.info(f"从文件 {words_file} 加载了 {len(violence_words)} 个暴力词汇")
        except Exception as e:
            logger.error(f"加载暴力词汇文件失败: {e}")
            # 使用默认词汇作为后备
            violence_words = [
                "傻逼", "垃圾", "恶心", "滚", "死", "废物", "蠢", "脑残", "闭嘴", "操",
                "靠", "草", "妈的", "智障", "傻逼玩意儿", "垃圾东西", "去死", "滚蛋", "混蛋", "畜生"
            ]
        
        return violence_words
    
    def load_processed_data(self, video_id: Optional[str] = None) -> List[Dict]:
        """
        从processed目录加载预处理后的数据
        
        Args:
            video_id: 可选的视频ID
            
        Returns:
            List[Dict]: 加载的评论数据列表
        """
        logger.info(f"从 {self.processed_dir} 加载预处理数据...")
        
        all_data = []
        
        # 获取processed目录下的所有JSON文件
        file_pattern = f"*_{video_id}_*.json" if video_id else "*.json"
        try:
            json_files = list(self.processed_dir.glob(file_pattern))
            logger.info(f"找到 {len(json_files)} 个预处理数据文件")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'root' in data and isinstance(data['root'], list):
                            all_data.extend(data['root'])
                            logger.info(f"加载文件 {json_file.name}，包含 {len(data['root'])} 条根评论")
                except Exception as e:
                    logger.error(f"加载文件 {json_file.name} 失败: {e}")
        except Exception as e:
            logger.error(f"加载预处理数据失败: {e}")
        
        logger.info(f"成功加载 {len(all_data)} 条评论集")
        return all_data
    
    def get_violence_score(self, text: str, tokens: List[str] = None) -> float:
        """
        使用词典法计算单条评论的暴力程度
        优先使用tokens进行匹配，提高准确性
        公式：V_i = min(匹配到的暴力词数量 / 3, 1.0)
        
        Args:
            text: 评论文本
            tokens: 分词后的词汇列表（可选）
            
        Returns:
            float: 暴力程度分数 [0, 1]
        """
        try:
            # 优先使用tokens进行匹配
            if tokens and isinstance(tokens, list):
                # 计算匹配到的暴力词数量（使用集合交集）
                token_set = set(tokens)
                violence_words_set = set(self.violence_words)
                # 计算交集大小
                count = len(token_set & violence_words_set)
            elif text and text.strip():
                # 回退到使用原始文本
                count = sum([text.count(w) for w in self.violence_words])
            else:
                return 0.0
            
            # 应用公式，上限为1
            return min(count / 3.0, 1.0)
        except Exception as e:
            logger.error(f"计算暴力分数失败: {e}")
            return 0.0
    
    def calculate_group_violence(self, comment_thread: Dict) -> Dict:
        """
        计算单个评论集的暴力强度
        公式：GroupViolence = Σ(V_i × (1 + log(1 + like_i))) / N
        
        Args:
            comment_thread: 评论集数据
            
        Returns:
            Dict: 包含暴力强度和详细信息的字典
        """
        parent_comment = comment_thread
        sub_comments = parent_comment.get('children', [])
        
        # 收集所有评论
        all_comments = [parent_comment]
        all_comments.extend(sub_comments)
        
        # 计算每个评论的暴力分数和点赞权重
        violence_scores = []
        likes = []
        comment_violence_details = []
        
        for comment in all_comments:
            content = comment.get('content', '')
            tokens = comment.get('tokens', [])
            like_count = comment.get('like_count', 0)
            
            # 计算暴力分数，优先使用tokens
            violence_score = self.get_violence_score(content, tokens)
            
            violence_scores.append(violence_score)
            likes.append(like_count)
            
            # 收集详细信息
            comment_detail = {
                'comment_id': comment.get('comment_id'),
                'content': content,
                'tokens': tokens,
                'like_count': like_count,
                'violence_score': violence_score,
                'is_parent': comment == parent_comment
            }
            comment_violence_details.append(comment_detail)
        
        # 计算加权暴力强度
        if violence_scores:
            # 计算权重：1 + log(1 + like_count)
            weights = np.array([1 + np.log1p(like) for like in likes])
            
            # 计算加权和
            weighted_sum = np.sum(np.array(violence_scores) * weights)
            
            # 加权平均
            group_violence = weighted_sum / len(violence_scores)
        else:
            group_violence = 0.0
        
        # 计算额外统计信息
        avg_violence = np.mean(violence_scores) if violence_scores else 0.0
        max_violence = np.max(violence_scores) if violence_scores else 0.0
        
        result = {
            'comment_id': parent_comment.get('comment_id'),
            'group_violence': group_violence,
            'average_violence': avg_violence,
            'max_violence': max_violence,
            'total_comments': len(all_comments),
            'parent_likes': parent_comment.get('like_count', 0),
            'sub_comments_count': len(sub_comments),
            'parent_violence_score': self.get_violence_score(parent_comment.get('content', '')),
            'comment_violence_details': comment_violence_details
        }
        
        return result
    
    def analyze(self, data: List[Dict]) -> List[Dict]:
        """
        分析所有评论集的暴力强度
        
        Args:
            data: 评论集数据列表
            
        Returns:
            List[Dict]: 分析结果列表
        """
        logger.info("开始分析网络暴力强度...")
        
        results = []
        total_groups = len(data)
        self.stats['total_threads'] = total_groups
        
        # 过滤掉parent_likes为0且sub_comments_count为0的父评论
        filtered_data = []
        for thread in data:
            parent_likes = thread.get('like_count', 0)
            sub_comments_count = len(thread.get('children', []))
            if not (parent_likes == 0 and sub_comments_count == 0):
                filtered_data.append(thread)
        
        logger.info(f"过滤后的评论集数量: {len(filtered_data)}/{total_groups}")
        
        for i, thread in enumerate(filtered_data):
            try:
                if i % 10 == 0:
                    logger.info(f"处理进度: {i+1}/{len(filtered_data)}")
                
                result = self.calculate_group_violence(thread)
                results.append(result)
                
                with self._stats_lock:
                    self.stats['processed_threads'] += 1
                    self.stats['average_group_violence'] += result['group_violence']
                    self.stats['max_group_violence'] = max(self.stats['max_group_violence'], result['group_violence'])
                    self.stats['min_group_violence'] = min(self.stats['min_group_violence'], result['group_violence'])
                    
            except Exception as e:
                logger.error(f"处理评论集失败: {e}")
        
        # 计算平均暴力强度
        if results:
            self.stats['average_group_violence'] /= len(results)
        
        logger.info(f"网络暴力强度分析完成，共处理 {len(results)} 个评论集")
        logger.info(f"平均暴力强度: {self.stats['average_group_violence']:.4f}")
        logger.info(f"最大暴力强度: {self.stats['max_group_violence']:.4f}")
        logger.info(f"最小暴力强度: {self.stats['min_group_violence']:.4f}")
        
        return results
    
    def save_results(self, results: List[Dict]):
        """
        保存分析结果
        
        Args:
            results: 分析结果列表
        """
        # 确保输出目录存在
        output_dir = self.output_dir / "group_violence"
        os.makedirs(output_dir, exist_ok=True)
        
        # 收集包含暴力词汇的评论信息
        violence_instances = []
        for result in results:
            for comment_detail in result.get('comment_violence_details', []):
                if comment_detail.get('violence_score', 0) > 0:
                    content = comment_detail.get('content', '')
                    tokens = comment_detail.get('tokens', [])
                    
                    # 优先使用tokens查找暴力词汇
                    found_words = []
                    if tokens and isinstance(tokens, list):
                        token_set = set(tokens)
                        found_words = list(token_set & set(self.violence_words))
                    elif content:
                        # 回退到使用原始文本
                        for word in self.violence_words:
                            if word in content:
                                found_words.append(word)
                    
                    if found_words:
                        violence_instances.append({
                            'comment_id': comment_detail.get('comment_id'),
                            'violence_words': found_words,
                            'violence_score': comment_detail.get('violence_score'),
                            'is_parent': comment_detail.get('is_parent', False)
                        })
        
        # 限制样本数量，避免结果文件过大
        violence_words_sample = violence_instances[:20]  # 只保留前20个实例
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"group_violence_results_{timestamp}.json"
        
        output_data = {
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'module_name': 'group_violence',
            'video_id': self.video_id,
            'stats': self.stats,
            'violence_words_count': len(self.violence_words),
            'violence_words_sample': violence_words_sample,
            'total_violence_instances': len(violence_instances),
            'results': results
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已保存到: {results_file}")
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
    
    def run_analysis(self, limit: Optional[int] = None, **kwargs):
        """
        运行完整的分析流程
        
        Args:
            limit: 可选的处理数量限制
            **kwargs: 其他参数
            
        Returns:
            Dict: 包含分析结果的字典
        """
        logger.info("开始运行网络暴力强度分析流程...")
        start_time = time.time()
        
        try:
            # 加载数据
            data = self.load_processed_data(self.video_id)
            
            # 限制处理数量
            if limit and isinstance(limit, int) and limit > 0:
                data = data[:limit]
                logger.info(f"限制处理数量为: {limit}")
            
            # 分析数据
            results = self.analyze(data)
            
            # 保存结果
            self.save_results(results)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"分析流程完成，耗时: {duration:.2f} 秒")
            
            return {
                'status': 'success',
                'results': results,
                'stats': self.stats,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"分析流程失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

def main():
    """
    主函数，用于命令行运行
    """
    parser = create_parser("group_violence", "网络暴力强度分析模块")
    # 添加特定参数
    parser.add_argument('--concurrency', type=int, default=8, help='并发数')
    parser.add_argument('--batch-size', type=int, default=200, help='批处理大小')
    
    args = parser.parse_args()
    args = parse_common_args(parser, args)
    
    # 创建分析器
    analyzer = GroupViolenceAnalyzer(args.video_id)
    
    # 设置并发参数
    analyzer.concurrency = args.concurrency
    analyzer.batch_size = args.batch_size
    
    # 运行分析
    analyzer.run_analysis(limit=args.limit)

if __name__ == "__main__":
    main()