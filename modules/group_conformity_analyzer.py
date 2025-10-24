#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
群体从众心理指数分析模块
根据建模思路构建GroupConformity指数，综合考虑语义趋同、情绪一致性和点赞集中度
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
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.base_analyzer import BaseAnalyzer, create_parser, parse_common_args
from core.aliyun_api_manager import AliyunAPIManager, get_aliyun_api_manager
from core.data_paths import AnalysisPathManager, PROJECT_ROOT, DATA_DIRS
from algorithms.normalization import MinMaxNormalizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GroupConformityAnalyzer(BaseAnalyzer):
    """群体从众心理指数分析器"""
    
    def __init__(self, video_id: Optional[str] = None):
        """
        初始化群体从众心理指数分析器
        
        Args:
            video_id: 可选的视频ID
        """
        super().__init__("group_conformity", video_id)
        
        # 初始化输出目录
        self.output_dir = DATA_DIRS['reports']
        
        # 初始化阿里云API管理器
        self.api_manager = get_aliyun_api_manager()
        if not self.api_manager:
            raise ValueError("阿里云API配置缺失，请设置环境变量")
        
        # 初始化归一化器
        self.normalizer = MinMaxNormalizer(feature_range=(0, 1))
        
        # 并发控制参数
        # 并发线程数 - 控制同时运行的线程数量，建议根据CPU核心数和API限制调整
        # 阿里云API有调用限制，建议值：4-16，设置过低会降低处理速度
        self.concurrency = 8
        
        # 批量处理大小 - 控制单次处理的数据量，影响内存使用和处理效率
        # 数据量大时可增大，但会占用更多内存；数据量小时可减小提高实时性
        # 建议值：100-500
        self.batch_size = 200
        
        # API调用间隔时间（毫秒）- 控制API调用频率，避免触发限流
        # 阿里云API一般建议调用频率不超过10次/秒
        # 如果仍出现API错误，可适当增大此值；如果处理速度太慢，可尝试减小
        self.throttle_ms = 50
        self._stats_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_threads': 0,
            'processed_threads': 0,
            'api_calls': 0,
            'api_errors': 0,
            'average_group_conformity': 0.0
        }
        
        # 结果存储
        self.results = {}
        
        # 预处理数据目录
        self.processed_dir = DATA_DIRS['processed']
        logger.info("✅ 群体从众心理指数分析器初始化成功")
    
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
            vector = self.api_manager.get_text_vector(text)
            
            with self._stats_lock:
                self.stats['api_calls'] += 1
            
            return vector
        except Exception as e:
            logger.error(f"获取文本向量失败: {e}")
            with self._stats_lock:
                self.stats['api_errors'] += 1
            return []
    
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
                'confidence': 0.0
            }
        
        try:
            result = self.api_manager.analyze_sentiment(text)
            
            with self._stats_lock:
                self.stats['api_calls'] += 1
            
            return result
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            with self._stats_lock:
                self.stats['api_errors'] += 1
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0
            }
    
    def calculate_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
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
            v1 = np.array(vector1).reshape(1, -1)
            v2 = np.array(vector2).reshape(1, -1)
            similarity = cosine_similarity(v1, v2)[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def calculate_herfindahl_index(self, likes: List[int]) -> float:
        """
        计算Herfindahl指数（点赞集中度）
        
        Args:
            likes: 点赞数列表
            
        Returns:
            float: Herfindahl指数 [0, 1]
        """
        if not likes or sum(likes) == 0:
            return 0.0
        
        total_likes = sum(likes)
        p_list = [like / total_likes for like in likes]
        herfindahl = sum(p ** 2 for p in p_list)
        
        return herfindahl
    
    def calculate_semantic_similarity(self, comments: List[Dict]) -> float:
        """
        计算语义趋同度（只计算子评论与父评论之间的相似度）
        
        Args:
            comments: 评论列表（包含父评论和子评论）
            
        Returns:
            float: 语义趋同度分数 [0, 1]
        """
        if len(comments) < 2:
            return 0.0
        
        # 父评论是第一个，子评论是其余的
        parent_comment = comments[0]
        sub_comments = comments[1:]
        
        if not parent_comment.get('content') or len(sub_comments) == 0:
            return 0.0
        
        # 并发获取文本向量
        vectors = []
        texts = [parent_comment.get('content', '')] + [comment.get('content', '') for comment in sub_comments if comment.get('content')]
        
        def _process_text(text: str) -> List[float]:
            vector = self.get_text_vector(text)
            if self.throttle_ms > 0:
                time.sleep(self.throttle_ms / 1000.0)
            return vector
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(_process_text, text): text for text in texts}
            for future in as_completed(futures):
                vector = future.result()
                vectors.append(vector)
        
        # 确保有父评论向量和至少一个子评论向量
        if len(vectors) < 2:
            return 0.0
        
        # 计算父评论与每个子评论的相似度
        parent_vector = vectors[0]
        similarities = []
        
        for i in range(1, len(vectors)):
            if vectors[i]:  # 确保子评论向量有效
                sim = self.calculate_cosine_similarity(parent_vector, vectors[i])
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        # 返回子评论与父评论相似度的平均值
        return np.mean(similarities)
    
    def calculate_emotion_align(self, comments: List[Dict]) -> float:
        """
        计算情绪一致性（只计算子评论与父评论之间的情绪一致率）
        
        Args:
            comments: 评论列表（包含父评论和子评论）
            
        Returns:
            float: 情绪一致性分数 [0, 1]
        """
        if len(comments) < 2:
            return 0.0
        
        # 父评论是第一个，子评论是其余的
        parent_comment = comments[0]
        sub_comments = comments[1:]
        
        if not parent_comment.get('content') or len(sub_comments) == 0:
            return 0.0
        
        # 并发分析情感
        all_comments_to_analyze = [parent_comment] + [c for c in sub_comments if c.get('content')]
        sentiments = []
        
        def _process_comment(comment: Dict) -> Dict:
            text = comment.get('content', '')
            result = self.analyze_sentiment(text)
            if self.throttle_ms > 0:
                time.sleep(self.throttle_ms / 1000.0)
            return result
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(_process_comment, comment): comment for comment in all_comments_to_analyze}
            for future in as_completed(futures):
                sentiment = future.result()
                sentiments.append(sentiment)
        
        if len(sentiments) < 2:
            return 0.0
        
        # 获取父评论的情感极性
        parent_score = sentiments[0].get('score', 0.0)
        parent_sign = np.sign(parent_score)
        
        # 计算与父评论情感极性一致的子评论数量
        aligned_count = 0
        valid_sub_comments = 0
        
        for i in range(1, len(sentiments)):
            sub_score = sentiments[i].get('score', 0.0)
            sub_sign = np.sign(sub_score)
            
            # 跳过情感极性为0的子评论
            if sub_sign == 0:
                continue
            
            valid_sub_comments += 1
            # 计算与父评论的情感极性一致率（考虑父评论极性为0的情况）
            if parent_sign == 0:
                # 如果父评论极性为0，我们认为任何非零的子评论都不一致
                continue
            elif sub_sign == parent_sign:
                aligned_count += 1
        
        if valid_sub_comments == 0:
            return 0.0
        
        # 返回情感一致性比率
        return aligned_count / valid_sub_comments
    
    def calculate_emotion_conformity(self, comments: List[Dict]) -> float:
        """
        计算情绪一致性（基于情感值差值的绝对值的平均值）
        公式：SC = 1 - (1/n) * Σ|S_i - S_p|
        
        Args:
            comments: 评论列表（包含父评论和子评论）
            
        Returns:
            float: 情绪一致性分数 [0, 1]，越接近1表示情感越趋同
        """
        if len(comments) < 2:
            return 0.0
        
        # 父评论是第一个，子评论是其余的
        parent_comment = comments[0]
        sub_comments = comments[1:]
        
        if not parent_comment.get('content') or len(sub_comments) == 0:
            return 0.0
        
        # 并发分析情感
        all_comments_to_analyze = [parent_comment] + [c for c in sub_comments if c.get('content')]
        sentiments = []
        
        def _process_comment(comment: Dict) -> Dict:
            text = comment.get('content', '')
            # 保留原始API返回值
            original_result = self.analyze_sentiment(text)
            if self.throttle_ms > 0:
                time.sleep(self.throttle_ms / 1000.0)
            return original_result
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(_process_comment, comment): comment for comment in all_comments_to_analyze}
            for future in as_completed(futures):
                # 保存完整的情感分析结果
                sentiment = future.result()
                sentiments.append(sentiment)
        
        if len(sentiments) < 2:
            return 0.0
        
        # 获取父评论的情感值
        parent_score = sentiments[0].get('score', 0.0)
        
        # 计算子评论与父评论情感值差值的绝对值之和
        total_diff = 0.0
        valid_sub_comments = 0
        
        for i in range(1, len(sentiments)):
            # 从完整结果中获取score字段用于计算
            sub_score = sentiments[i].get('score', 0.0)
            total_diff += abs(sub_score - parent_score)
            valid_sub_comments += 1
        
        if valid_sub_comments == 0:
            return 0.0
        
        # 计算平均差值
        avg_diff = total_diff / valid_sub_comments
        
        # 应用公式：SC = 1 - (1/n) * Σ|S_i - S_p|
        # 由于情感值范围在[-1,1]，差值范围在[0,2]，确保结果在[0,1]范围内
        emotion_conformity = 1 - avg_diff / 2.0
        
        # 确保结果在有效范围内
        return max(0.0, min(1.0, emotion_conformity))
    
    def calculate_group_conformity(self, comment_thread: Dict) -> Dict:
        """
        计算单个评论集的从众心理指数
        
        Args:
            comment_thread: 评论集数据
            
        Returns:
            Dict: 包含从众心理指数和各维度分数的字典
        """
        parent_comment = comment_thread
        sub_comments = parent_comment.get('children', [])
        
        # 收集所有评论
        all_comments = [parent_comment]
        all_comments.extend(sub_comments)
        
        # 提取点赞数
        likes = [comment.get('like_count', 0) for comment in all_comments]
        
        # 计算三个维度的分数
        semantic_sim = self.calculate_semantic_similarity(all_comments)
        emotion_align = self.calculate_emotion_align(all_comments)  # 原有情感一致率算法
        emotion_conformity = self.calculate_emotion_conformity(all_comments)  # 新的情感一致性算法
        like_concentration = self.calculate_herfindahl_index(likes)
        
        # 加权计算总从众心理指数
        # 权重：语义趋同(0.4)、情绪一致性(0.3)、点赞集中度(0.3)
        # 使用新的情感一致性算法进行计算
        group_conformity = 0.4 * semantic_sim + 0.3 * emotion_conformity + 0.3 * like_concentration
        
        # 收集每个评论的详细情感分析数据（包含官方文档定义的原始字段）
        comments_sentiment_details = []
        for comment in all_comments:
            if comment.get('content'):
                sentiment_result = self.analyze_sentiment(comment.get('content', ''))
                # 确保包含官方文档定义的原始字段
                sentiment_detail = {
                    'comment_id': comment.get('comment_id'),
                    'sentiment': sentiment_result.get('sentiment'),
                    'positive_prob': sentiment_result.get('positive_prob'),
                    'negative_prob': sentiment_result.get('negative_prob'),
                    'neutral_prob': sentiment_result.get('neutral_prob'),
                    'score': sentiment_result.get('score'),
                    'confidence': sentiment_result.get('confidence')
                }
                comments_sentiment_details.append(sentiment_detail)
        
        result = {
            'comment_id': parent_comment.get('comment_id'),
            'group_conformity': group_conformity,
            'semantic_similarity': semantic_sim,
            'emotion_alignment': emotion_align,  # 保留原有算法结果
            'emotion_conformity': emotion_conformity,  # 添加新算法结果
            'like_concentration': like_concentration,
            'total_comments': len(all_comments),
            'parent_likes': parent_comment.get('like_count', 0),
            'sub_comments_count': len(sub_comments),
            'comments_sentiment_details': comments_sentiment_details  # 添加详细的情感分析数据
        }
        
        return result
    
    def analyze(self, data: List[Dict]) -> List[Dict]:
        """
        分析所有评论集的从众心理指数
        
        Args:
            data: 评论集数据列表
            
        Returns:
            List[Dict]: 分析结果列表
        """
        logger.info("开始分析群体从众心理指数...")
        
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
                
                result = self.calculate_group_conformity(thread)
                results.append(result)
                
                with self._stats_lock:
                    self.stats['processed_threads'] += 1
                    self.stats['average_group_conformity'] += result['group_conformity']
                    
            except Exception as e:
                logger.error(f"处理评论集失败: {e}")
        
        # 计算平均从众心理指数
        if results:
            self.stats['average_group_conformity'] /= len(results)
        
        logger.info(f"群体从众心理分析完成，共处理 {len(results)} 个评论集")
        logger.info(f"平均从众心理指数: {self.stats['average_group_conformity']:.4f}")
        
        return results
    
    def save_results(self, results: List[Dict]):
        """
        保存分析结果
        
        Args:
            results: 分析结果列表
        """
        # 确保输出目录存在
        output_dir = self.output_dir / "group_conformity"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"group_conformity_results_{timestamp}.json"
        
        output_data = {
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'module_name': 'group_conformity',
            'video_id': self.video_id,
            'stats': self.stats,
            'results': results
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {results_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def run_analysis(self, limit: Optional[int] = None, **kwargs):
        """
        运行完整的分析流程
        
        Args:
            limit: 限制处理的评论集数量
            **kwargs: 其他参数
        """
        # 加载预处理数据
        data = self.load_processed_data(self.video_id)
        
        # 如果指定了限制，只处理前N个
        if limit and len(data) > limit:
            data = data[:limit]
            logger.info(f"限制处理前 {limit} 个评论集")
        
        # 执行分析
        results = self.analyze(data)
        
        # 保存结果
        self.save_results(results)
        
        return {
            'total_processed': len(results),
            'average_group_conformity': self.stats['average_group_conformity'],
            'api_calls': self.stats['api_calls'],
            'api_errors': self.stats['api_errors']
        }

def main():
    """主函数"""
    parser = create_parser("group_conformity", "群体从众心理指数分析工具")
    # 添加特定参数
    parser.add_argument('--concurrency', type=int, default=8, help='API并发数')
    parser.add_argument('--batch-size', type=int, default=200, help='批处理大小')
    
    args = parser.parse_args()
    args = parse_common_args(parser, args)
    
    # 创建分析器
    analyzer = GroupConformityAnalyzer(args.video_id)
    
    # 设置并发参数
    analyzer.concurrency = args.concurrency
    analyzer.batch_size = args.batch_size
    
    # 运行分析
    analyzer.run_analysis(limit=args.limit)

if __name__ == "__main__":
    main()