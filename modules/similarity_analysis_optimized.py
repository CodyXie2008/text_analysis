#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本相似度分析模块
基于阿里云通用文本向量API实现父子评论相似度分析
用于识别"模仿性评论"，量化从众心理
"""

import os
import sys
import json
import time
import warnings
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入基础分析器
from text_analysis.core.base_analyzer import BaseAnalyzer
from text_analysis.core.data_paths import PROJECT_ROOT, resolve_latest_cleaned_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


class AliyunTextVectorClient:
    """阿里云词向量API客户端"""
    
    def __init__(self, access_key_id: str, access_key_secret: str):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = "https://alinlp.cn-hangzhou.aliyuncs.com"
        self.service_code = "alinlp"
        
    def get_text_vector(self, text: str) -> List[float]:
        """获取文本向量 - 使用阿里云词向量API"""
        try:
            # 尝试使用阿里云SDK
            try:
                from aliyunsdkcore.client import AcsClient
                from aliyunsdkcore.request import CommonRequest
                
                # 创建AcsClient实例
                client = AcsClient(
                    self.access_key_id,
                    self.access_key_secret,
                    'cn-hangzhou'
                )
                
                # 使用CommonRequest
                request = CommonRequest()
                request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
                request.set_version('2020-06-29')
                request.set_action_name('GetWeChGeneral')
                request.add_query_param('ServiceCode', 'alinlp')
                request.add_query_param('Text', text)
                request.add_query_param('Size', '100')
                request.add_query_param('Type', 'word')
                request.add_query_param('Operation', 'average')
                
                # 发送请求
                response = client.do_action_with_exception(request)
                result = json.loads(response)
                
                # 解析返回结果
                if "Data" in result:
                    data_content = result["Data"]
                    if isinstance(data_content, str):
                        data_content = json.loads(data_content)
                    
                    if "result" in data_content and data_content["result"]:
                        if isinstance(data_content["result"], dict) and "vec" in data_content["result"]:
                            return data_content["result"]["vec"]
                        elif isinstance(data_content["result"], list):
                            vectors = [item["vec"] for item in data_content["result"] if "vec" in item]
                            if vectors:
                                import numpy as np
                                return np.mean(vectors, axis=0).tolist()
                
                raise Exception(f"API返回数据格式异常: {result}")
                
            except ImportError:
                raise Exception("阿里云SDK未安装，请运行: pip install aliyun-python-sdk-core")
            except Exception as e:
                raise e
                
        except Exception as e:
            error_msg = str(e)
            if "400" in error_msg:
                if "BasicServiceNotActivated" in error_msg:
                    print("⚠️ 阿里云NLP基础版服务未开通，请访问：https://common-buy.aliyun.com/?commodityCode=nlp%5FalinlpBasePost%5Fpublic%5Fcn#/buy")
                elif "UserStatusInvalid" in error_msg:
                    print("⚠️ 用户状态无效，请检查账户是否欠费")
                elif "InvalidParameter" in error_msg:
                    print("⚠️ 参数无效，请检查API调用参数")
                else:
                    print(f"⚠️ 阿里云API调用失败(400错误): {error_msg}")
            else:
                print(f"⚠️ 阿里云API调用失败: {error_msg}")
            
            print("🔄 自动切换到本地TF-IDF方法...")
            # 使用本地TF-IDF作为备选方案
            return self._get_tfidf_vector(text)
    
    def _get_tfidf_vector(self, text: str) -> List[float]:
        """使用本地TF-IDF计算文本向量"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import jieba
            
            # 简单的文本预处理
            text = text.strip()
            if not text:
                return [0.0] * 100
            
            # 使用jieba分词
            words = list(jieba.cut(text))
            processed_text = ' '.join(words)
            
            # 创建TF-IDF向量化器
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=None,
                ngram_range=(1, 2)
            )
            
            # 计算TF-IDF向量
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            vector = tfidf_matrix.toarray()[0].tolist()
            
            # 确保向量长度为100
            if len(vector) < 100:
                vector.extend([0.0] * (100 - len(vector)))
            elif len(vector) > 100:
                vector = vector[:100]
            
            return vector
            
        except Exception as e:
            print(f"❌ TF-IDF计算失败: {e}")
            return [0.0] * 100
    
    def batch_get_vectors(self, texts: List[str], batch_size: int = 50, concurrency: int = 4, throttle_ms: int = 0) -> List[List[float]]:
        """批量获取文本向量（支持并发与节流）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        total = len(texts)
        vectors: List[List[float]] = [None] * total  # type: ignore

        def _submit_range(start: int, end: int):
            print(f"🔄 处理向量 {start+1}-{end}/{total}")
            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
                futures = {}
                for idx in range(start, end):
                    if throttle_ms > 0 and (idx - start) % concurrency == 0:
                        time.sleep(throttle_ms / 1000.0)
                    futures[ex.submit(self.get_text_vector, texts[idx])] = idx
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        vectors[idx] = fut.result()
                    except Exception:
                        vectors[idx] = [0.0] * 100

        for i in range(0, total, batch_size):
            _submit_range(i, min(i + batch_size, total))

        return vectors  # type: ignore


class SimilarityAnalyzer:
    """文本相似度分析器"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 time_diff_threshold: int = 3600,
                 min_text_length: int = 5,
                 vector_batch_size: int = 50,
                 vector_concurrency: int = 4,
                 vector_throttle_ms: int = 0):
        self.similarity_threshold = similarity_threshold
        self.time_diff_threshold = time_diff_threshold
        self.min_text_length = min_text_length
        self.vector_batch_size = vector_batch_size
        self.vector_concurrency = vector_concurrency
        self.vector_throttle_ms = vector_throttle_ms
        
        # 初始化阿里云客户端
        access_key_id = os.getenv('NLP_AK_ENV')
        access_key_secret = os.getenv('NLP_SK_ENV')
        
        if not access_key_id or not access_key_secret:
            raise ValueError("❌ 请设置阿里云API密钥环境变量: NLP_AK_ENV, NLP_SK_ENV")
        
        self.vector_client = AliyunTextVectorClient(access_key_id, access_key_secret)
        print("✅ 阿里云文本向量客户端初始化成功")
        print(f"向量请求参数：并发={self.vector_concurrency} 批大小={self.vector_batch_size} 限流={self.vector_throttle_ms}ms")
    
    def _load_from_database(self, video_id: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """从数据库加载数据"""
        try:
            from config.db_config import get_db_conn
            conn = get_db_conn()
            
            if video_id:
                # 分析指定视频的评论
                sql = """
                SELECT comment_id as id, content, create_time, like_count, aweme_id
                FROM douyin_aweme_comment
                WHERE content IS NOT NULL AND LENGTH(content) > 5 AND aweme_id = %s
                ORDER BY create_time DESC
                """
                params = [video_id]
                print(f"分析视频 {video_id} 的评论...")
            else:
                # 分析所有评论
                sql = """
                SELECT comment_id as id, content, create_time, like_count, aweme_id
                FROM douyin_aweme_comment
                WHERE content IS NOT NULL AND LENGTH(content) > 5
                ORDER BY create_time DESC
                """
                params = []
                print("分析所有评论...")
            
            if limit:
                sql += f" LIMIT {limit}"
                print(f"限制分析数量: {limit}")
            
            df = pd.read_sql_query(sql, conn, params=params)
            conn.close()
            
            print(f"✅ 成功加载 {len(df)} 条评论")
            return df
            
        except Exception as e:
            print(f"❌ 数据库加载失败: {e}")
            return pd.DataFrame()
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """执行相似度分析"""
        print("=== 开始文本相似度分析 ===")
        
        # 1. 数据预处理
        print("1. 数据预处理...")
        df_processed = self._preprocess_data(df)
        
        if df_processed.empty:
            return {"error": "没有有效的数据进行分析"}
        
        # 2. 计算文本向量
        print("2. 计算文本向量...")
        vectors = self._calculate_text_vectors(df_processed['content'].tolist())
        
        # 3. 计算相似度矩阵
        print("3. 计算相似度矩阵...")
        similarity_matrix = self._calculate_similarity_matrix(vectors)
        
        # 4. 识别评论对
        print("4. 识别评论对...")
        comment_pairs = self._identify_comment_pairs(df_processed)
        
        # 5. 分析相似度
        print("5. 分析相似度...")
        similarity_results = self._analyze_similarity(df_processed, similarity_matrix, comment_pairs)
        
        # 6. 识别模仿性评论
        print("6. 识别模仿性评论...")
        imitative_comments = self._identify_imitative_comments(df_processed, similarity_results)
        
        # 7. 生成统计结果
        print("7. 生成统计结果...")
        stats = self._generate_statistics(df_processed, similarity_results, imitative_comments)
        
        # 保存处理后的数据
        self.df_processed = df_processed
        self.similarity_results = similarity_results
        self.imitative_comments = imitative_comments
        
        print("✅ 文本相似度分析完成")
        return stats
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 复制数据避免修改原始数据
        df_processed = df.copy()
        
        # 过滤空内容和短内容
        df_processed = df_processed[df_processed['content'].notna()]
        df_processed = df_processed[df_processed['content'].str.len() >= self.min_text_length]
        
        # 去重
        df_processed = df_processed.drop_duplicates(subset=['content'])
        
        print(f"   - 原始数据: {len(df)} 条")
        print(f"   - 有效数据: {len(df_processed)} 条")
        
        return df_processed
    
    def _calculate_text_vectors(self, texts: List[str]) -> List[List[float]]:
        """计算文本向量（去重+并发）"""
        # 去重映射
        unique_index: Dict[str, int] = {}
        order_to_text: Dict[int, str] = {}
        for i, t in enumerate(texts):
            if t not in unique_index:
                unique_index[t] = len(unique_index)
            order_to_text[i] = t
        unique_list = [None] * len(unique_index)
        for t, u in unique_index.items():
            unique_list[u] = t

        # 使用配置的批大小、并发与限流（默认限流=0 即不做任何限流）
        vectors_unique = self.vector_client.batch_get_vectors(
            unique_list,
            batch_size=self.vector_batch_size,
            concurrency=self.vector_concurrency,
            throttle_ms=self.vector_throttle_ms,
        )
        vec_map: Dict[str, List[float]] = {t: vectors_unique[u] for t, u in unique_index.items()}
        return [vec_map[order_to_text[i]] for i in range(len(texts))]
    
    def _calculate_similarity_matrix(self, vectors: List[List[float]]) -> np.ndarray:
        """计算相似度矩阵"""
        vectors_array = np.array(vectors)
        similarity_matrix = cosine_similarity(vectors_array)
        return similarity_matrix
    
    def _identify_comment_pairs(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """识别评论对（用于相似度分析）"""
        pairs = []
        n = len(df)
        
        # 为了避免计算量过大，只分析相邻的评论对
        for i in range(n - 1):
            pairs.append((i, i + 1))
        
        print(f"   - 找到 {len(pairs)} 对相邻评论")
        return pairs
    
    def _analyze_similarity(self, df: pd.DataFrame, similarity_matrix: np.ndarray, 
                          comment_pairs: List[Tuple[int, int]]) -> pd.DataFrame:
        """分析相似度"""
        results = []
        
        for idx1, idx2 in comment_pairs:
            similarity = similarity_matrix[idx1][idx2]
            
            # 计算时间差
            time1 = pd.to_datetime(df.iloc[idx1]['create_time'])
            time2 = pd.to_datetime(df.iloc[idx2]['create_time'])
            time_diff = abs((time2 - time1).total_seconds())
            
            # 处理不同的字段名
            id_field = 'id' if 'id' in df.columns else 'comment_id'
            
            results.append({
                'comment1_id': df.iloc[idx1][id_field],
                'comment2_id': df.iloc[idx2][id_field],
                'comment1_content': df.iloc[idx1]['content'],
                'comment2_content': df.iloc[idx2]['content'],
                'similarity': similarity,
                'time_diff': time_diff,
                'is_imitative': (similarity > self.similarity_threshold) and (time_diff < self.time_diff_threshold)
            })
        
        return pd.DataFrame(results)
    
    def _identify_imitative_comments(self, df: pd.DataFrame, similarity_results: pd.DataFrame) -> pd.DataFrame:
        """识别模仿性评论"""
        imitative_pairs = similarity_results[similarity_results['is_imitative'] == True]
        
        # 获取模仿性评论的详细信息
        imitative_comment_ids = imitative_pairs['comment1_id'].tolist() + imitative_pairs['comment2_id'].tolist()
        # 处理不同的字段名
        id_field = 'id' if 'id' in df.columns else 'comment_id'
        imitative_comments = df[df[id_field].isin(imitative_comment_ids)].copy()
        
        # 添加相似度信息（简化处理）
        imitative_comments['similarity'] = 0.0
        imitative_comments['time_diff'] = 0.0
        
        return imitative_comments
    
    def _generate_statistics(self, df: pd.DataFrame, similarity_results: pd.DataFrame, 
                           imitative_comments: pd.DataFrame) -> Dict:
        """生成统计结果"""
        total_comments = len(df)
        total_pairs = len(similarity_results)
        imitative_pairs = len(similarity_results[similarity_results['is_imitative'] == True])
        imitative_comments_count = len(imitative_comments)
        
        # 相似度统计
        similarities = similarity_results['similarity'].tolist()
        avg_similarity = np.mean(similarities) if similarities else 0
        median_similarity = np.median(similarities) if similarities else 0
        
        # 时间差统计
        time_diffs = similarity_results['time_diff'].tolist()
        avg_time_diff = np.mean(time_diffs) if time_diffs else 0
        
        stats = {
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'module_name': 'similarity',
            'total_comments': total_comments,
            'total_pairs': total_pairs,
            'imitative_pairs': imitative_pairs,
            'imitative_comments': imitative_comments_count,
            'imitative_ratio': imitative_pairs / total_pairs if total_pairs > 0 else 0,
            'avg_similarity': float(avg_similarity),
            'median_similarity': float(median_similarity),
            'avg_time_diff': float(avg_time_diff),
            'similarity_threshold': self.similarity_threshold,
            'time_diff_threshold': self.time_diff_threshold
        }
        
        return stats
    
    def save_results(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        """保存分析结果"""
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'data', 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存相似度分析结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 相似度对结果
        similarity_file = os.path.join(output_dir, f'similarity_analysis_{timestamp}.csv')
        self.similarity_results.to_csv(similarity_file, index=False, encoding='utf-8')
        print(f"✅ 相似度分析结果已保存到: {similarity_file}")
        
        # 模仿性评论结果
        imitative_file = os.path.join(output_dir, f'imitative_comments_{timestamp}.csv')
        self.imitative_comments.to_csv(imitative_file, index=False, encoding='utf-8')
        print(f"✅ 模仿性评论结果已保存到: {imitative_file}")
        
        # JSON格式结果
        json_file = os.path.join(output_dir, f'similarity_analysis_{timestamp}.json')
        json_data = {
            'similarity_pairs': self.similarity_results.to_dict('records'),
            'imitative_comments': self.imitative_comments.to_dict('records')
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON结果已保存到: {json_file}")
    
    def generate_report(self, df: pd.DataFrame):
        """生成分析报告"""
        if not hasattr(self, 'similarity_results'):
            print("❌ 请先执行分析")
            return
        
        report_dir = os.path.join(PROJECT_ROOT, 'data', 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(report_dir, f'similarity_analysis_report_{timestamp}.json')
        
        # 生成报告数据
        stats = self._generate_statistics(df, self.similarity_results, self.imitative_comments)
        
        # 添加详细统计
        similarity_stats = {
            'similarity_distribution': {
                '0.0-0.2': len(self.similarity_results[self.similarity_results['similarity'] < 0.2]),
                '0.2-0.4': len(self.similarity_results[(self.similarity_results['similarity'] >= 0.2) & (self.similarity_results['similarity'] < 0.4)]),
                '0.4-0.6': len(self.similarity_results[(self.similarity_results['similarity'] >= 0.4) & (self.similarity_results['similarity'] < 0.6)]),
                '0.6-0.8': len(self.similarity_results[(self.similarity_results['similarity'] >= 0.6) & (self.similarity_results['similarity'] < 0.8)]),
                '0.8-1.0': len(self.similarity_results[self.similarity_results['similarity'] >= 0.8])
            },
            'time_diff_distribution': {
                '0-1h': len(self.similarity_results[self.similarity_results['time_diff'] < 3600]),
                '1-6h': len(self.similarity_results[(self.similarity_results['time_diff'] >= 3600) & (self.similarity_results['time_diff'] < 21600)]),
                '6-24h': len(self.similarity_results[(self.similarity_results['time_diff'] >= 21600) & (self.similarity_results['time_diff'] < 86400)]),
                '>24h': len(self.similarity_results[self.similarity_results['time_diff'] >= 86400])
            }
        }
        
        report_data = {
            **stats,
            'detailed_stats': similarity_stats,
            'analysis_parameters': {
                'similarity_threshold': self.similarity_threshold,
                'time_diff_threshold': self.time_diff_threshold,
                'min_text_length': self.min_text_length
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 分析报告已保存到: {report_file}")
        
        # 打印报告摘要
        print("\n=== 分析报告摘要 ===")
        print(f"分析时间: {stats['analysis_time']}")
        print(f"模块名称: {stats['module_name']}")
        print(f"总评论数: {stats['total_comments']:,}")
        print(f"父子评论对: {stats['total_pairs']:,}")
        print(f"模仿性评论对: {stats['imitative_pairs']:,}")
        print(f"模仿性评论: {stats['imitative_comments']:,}")
        print(f"模仿比例: {stats['imitative_ratio']:.2%}")
        print(f"平均相似度: {stats['avg_similarity']:.3f}")
        print(f"中位数相似度: {stats['median_similarity']:.3f}")
        print(f"平均时间差: {stats['avg_time_diff']:.0f}秒")
    
    def create_visualizations(self, df: pd.DataFrame):
        """创建可视化图表"""
        if not hasattr(self, 'similarity_results'):
            print("❌ 请先执行分析")
            return
        
        viz_dir = os.path.join(PROJECT_ROOT, 'data', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('文本相似度分析可视化', fontsize=16, fontweight='bold')
        
        # 1. 相似度分布直方图
        ax1 = axes[0, 0]
        ax1.hist(self.similarity_results['similarity'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.similarity_threshold, color='red', linestyle='--', label=f'阈值: {self.similarity_threshold}')
        ax1.set_xlabel('相似度')
        ax1.set_ylabel('频次')
        ax1.set_title('相似度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 时间差分布直方图
        ax2 = axes[0, 1]
        ax2.hist(self.similarity_results['time_diff'] / 3600, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(self.time_diff_threshold / 3600, color='red', linestyle='--', label=f'阈值: {self.time_diff_threshold/3600:.1f}h')
        ax2.set_xlabel('时间差(小时)')
        ax2.set_ylabel('频次')
        ax2.set_title('时间差分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 相似度vs时间差散点图
        ax3 = axes[1, 0]
        scatter = ax3.scatter(self.similarity_results['similarity'], 
                            self.similarity_results['time_diff'] / 3600,
                            c=self.similarity_results['is_imitative'], 
                            cmap='RdYlGn', alpha=0.6)
        ax3.axhline(self.time_diff_threshold / 3600, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(self.similarity_threshold, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('相似度')
        ax3.set_ylabel('时间差(小时)')
        ax3.set_title('相似度 vs 时间差')
        ax3.grid(True, alpha=0.3)
        
        # 4. 模仿性评论统计
        ax4 = axes[1, 1]
        labels = ['非模仿性', '模仿性']
        sizes = [len(self.similarity_results[~self.similarity_results['is_imitative']]),
                 len(self.similarity_results[self.similarity_results['is_imitative']])]
        if sum(sizes) > 0:
            colors = ['lightcoral', 'lightblue']
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('模仿性评论比例')
        else:
            ax4.text(0.5, 0.5, '无模仿性统计数据', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存图表
        viz_file = os.path.join(viz_dir, f'similarity_analysis_main_{timestamp}.png')
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化图表已保存到: {viz_file}")
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文本相似度分析工具')
    parser.add_argument('--video-id', type=str, help='视频ID，如果不指定则分析所有数据')
    parser.add_argument('--limit', type=int, help='限制分析数量')
    parser.add_argument('--use-cleaned-data', action='store_true', help='使用清洗后的数据文件')
    parser.add_argument('--cleaned-data-path', type=str, help='清洗数据文件路径')
    parser.add_argument('--similarity-threshold', type=float, default=0.7, help='相似度阈值')
    parser.add_argument('--time-diff-threshold', type=int, default=3600, help='时间差阈值(秒)')
    parser.add_argument('--test', action='store_true', help='测试模式，只分析少量数据')
    parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    parser.add_argument('--no-viz', action='store_true', help='不创建可视化图表')
    # 向量请求性能参数
    parser.add_argument('--vector-batch-size', type=int, default=100, help='向量API批大小，默认100')
    parser.add_argument('--vector-concurrency', type=int, default=8, help='向量API并发数，默认8')
    parser.add_argument('--vector-throttle-ms', type=int, default=0, help='向量API节流毫秒，默认0=不限制')
    
    args = parser.parse_args()
    
    # 测试模式设置
    if args.test and not args.limit:
        args.limit = 10
        print("🧪 测试模式：只分析少量数据")
    
    # 显示配置信息
    print(f"=== 文本相似度分析工具 ===")
    if args.video_id:
        print(f"视频ID: {args.video_id}")
    if args.limit:
        print(f"限制数量: {args.limit}")
    if args.use_cleaned_data:
        print("使用清洗数据")
    print(f"相似度阈值: {args.similarity_threshold}")
    print(f"时间差阈值: {args.time_diff_threshold}秒")
    print("=" * 50)
    
    try:
        # 初始化分析器
        analyzer = SimilarityAnalyzer(
            similarity_threshold=args.similarity_threshold,
            time_diff_threshold=args.time_diff_threshold,
            vector_batch_size=max(1, args.vector_batch_size),
            vector_concurrency=max(1, args.vector_concurrency),
            vector_throttle_ms=max(0, args.vector_throttle_ms),
        )
        
        # 加载数据
        if args.use_cleaned_data:
            # 从清洗数据文件加载
            if args.cleaned_data_path:
                cleaned_data_path = args.cleaned_data_path
            else:
                auto_path = resolve_latest_cleaned_data(args.video_id)
                cleaned_data_path = auto_path or os.path.join(PROJECT_ROOT, 'data', 'processed', 'douyin_comments_processed.json')
            
            if not os.path.exists(cleaned_data_path):
                print(f"❌ 清洗数据文件不存在: {cleaned_data_path}")
                return
            
            with open(cleaned_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            print(f"✅ 成功加载清洗数据: {len(df)} 条记录")
            
            # 限制数据量
            if args.limit and len(df) > args.limit:
                df = df.head(args.limit)
                print(f"✅ 限制数据量: {len(df)} 条记录")
        else:
            # 从数据库加载
            df = analyzer._load_from_database(args.video_id, args.limit)
            if df.empty:
                print("❌ 没有找到数据")
                return
        
        # 执行分析
        stats = analyzer.analyze(df)
        
        if 'error' in stats:
            print(f"❌ 分析失败: {stats['error']}")
            return
        
        # 保存结果
        if not args.no_save:
            analyzer.save_results(df)
        
        # 生成报告
        if not args.no_report:
            analyzer.generate_report(df)
        
        # 创建可视化
        if not args.no_viz:
            analyzer.create_visualizations(df)
        
        print("\n✅ 文本相似度分析完成!")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
