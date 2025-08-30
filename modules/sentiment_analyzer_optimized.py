#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版情感分析模块
支持本地词典和阿里云API两种分析方式
统一执行入口，支持视频ID分析
"""

import os
import sys
import re
import json
import time
import threading
import argparse
import logging
import hmac
import hashlib
import base64
from datetime import datetime
from typing import Dict, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入PROJECT_ROOT
from text_analysis.core.data_paths import PROJECT_ROOT, resolve_latest_cleaned_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.db_config import get_db_conn

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv未安装，无法自动加载.env文件")

class DictionaryAnalyzer:
    """本地词典情感分析器"""
    
    def __init__(self):
        self.sentiment_dict = self._load_sentiment_dict()
        self.negation_words = {'不', '没', '无', '非', '未', '别', '莫', '勿', '毋', '弗', '否', '反'}
        self.intensifier_words = {
            '非常': 2.0, '特别': 2.0, '极其': 2.0, '十分': 2.0, '很': 1.5, '挺': 1.3,
            '比较': 1.2, '有点': 0.8, '稍微': 0.7, '略微': 0.7, '太': 1.8, '真': 1.5,
            '确实': 1.3, '真的': 1.5, '绝对': 2.0, '完全': 2.0,
        }
    
    def _load_sentiment_dict(self) -> Dict[str, float]:
        """加载情感词典"""
        return {
            # 正向情感词
            '好': 1.0, '棒': 1.0, '赞': 1.0, '优秀': 1.0, '完美': 1.0, '精彩': 1.0,
            '喜欢': 1.0, '爱': 1.0, '支持': 1.0, '推荐': 1.0, '满意': 1.0, '开心': 1.0,
            '高兴': 1.0, '快乐': 1.0, '兴奋': 1.0, '激动': 1.0, '感动': 1.0, '温暖': 1.0,
            '美好': 1.0, '漂亮': 1.0, '帅气': 1.0, '可爱': 1.0, '有趣': 1.0, '搞笑': 1.0,
            '厉害': 1.0, '强大': 1.0, '专业': 1.0, '高质量': 1.0, '超赞': 1.5, '太棒了': 1.5,
            '666': 1.0, '牛': 1.0, '神': 1.0, '绝了': 1.5, '无敌': 1.5, '爆赞': 1.5,
            '加油': 0.8, '好样': 0.8, '棒棒': 1.0, '棒棒哒': 1.2, '棒极了': 1.3,
            '太棒': 1.2, '很棒': 1.1, '非常好': 1.2, '特别好': 1.2, '超级棒': 1.4,
            '太赞': 1.2, '很赞': 1.1, '超赞': 1.3, '赞赞': 1.0, '赞赞赞': 1.2,
            '喜欢': 1.0, '很喜欢': 1.2, '超喜欢': 1.3, '爱了': 1.1, '爱了爱了': 1.3,
            
            # 负向情感词
            '差': -1.0, '烂': -1.0, '垃圾': -1.0, '糟糕': -1.0, '恶心': -1.0, '讨厌': -1.0,
            '恨': -1.0, '烦': -1.0, '生气': -1.0, '愤怒': -1.0, '失望': -1.0, '伤心': -1.0,
            '难过': -1.0, '痛苦': -1.0, '绝望': -1.0, '崩溃': -1.0, '无语': -1.0, '坑': -1.0,
            '骗': -1.0, '假': -1.0, '水': -1.0, '无聊': -1.0, '没意思': -1.0, '浪费时间': -1.0,
            '太差了': -1.5, '太烂了': -1.5, '太恶心了': -1.5, '太坑了': -1.5, '太假了': -1.5,
            '太无聊了': -1.5, '太失望了': -1.5, '太伤心了': -1.5, '太痛苦了': -1.5,
            '很差': -1.1, '非常差': -1.3, '特别差': -1.3, '超级差': -1.4,
            '太烂': -1.2, '很烂': -1.1, '超烂': -1.3, '烂透了': -1.4,
            '讨厌': -1.0, '很讨厌': -1.2, '超讨厌': -1.3, '太讨厌': -1.2,
            '失望': -1.0, '很失望': -1.2, '太失望': -1.3, '非常失望': -1.4,
            
            # 中性情感词
            '一般': 0.0, '还行': 0.0, '凑合': 0.0, '普通': 0.0, '正常': 0.0, '标准': 0.0,
            '可以': 0.0, '不错': 0.0, '还好': 0.0, '马马虎虎': 0.0, '过得去': 0.0,
            '还行': 0.0, '凑合': 0.0, '一般般': 0.0, '马马虎虎': 0.0, '过得去': 0.0,
        }
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """分析文本情感"""
        if not text or not text.strip():
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        # 文本预处理
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        words = text.split()
        
        if not words:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        total_score = 0.0
        word_count = 0
        
        for i, word in enumerate(words):
            word_score = 0.0
            
            # 检查情感词典
            if word in self.sentiment_dict:
                word_score = self.sentiment_dict[word]
            
            # 检查程度副词
            if i > 0 and words[i-1] in self.intensifier_words:
                word_score *= self.intensifier_words[words[i-1]]
            
            # 检查否定词
            if i > 0 and words[i-1] in self.negation_words:
                word_score *= -1
            
            total_score += word_score
            word_count += 1
        
        # 计算平均分数
        if word_count > 0:
            score = total_score / word_count
        else:
            score = 0.0
        
        # 确定情感极性
        if score >= 0.3:
            sentiment = 'positive'
        elif score <= -0.3:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # 计算置信度
        confidence = min(abs(score) * 2, 0.9) if abs(score) > 0 else 0.3
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence,
            'method': 'dictionary'
        }

class AliyunAnalyzer:
    """阿里云NLP情感分析器"""
    
    def __init__(self):
        self.access_key_id = os.getenv('NLP_AK_ENV')
        self.access_key_secret = os.getenv('NLP_SK_ENV')
        self.region_id = os.getenv('NLP_REGION_ENV', 'cn-hangzhou')
        self.endpoint = f"https://nlp.{self.region_id}.aliyuncs.com"
        
        if not self.access_key_id or not self.access_key_secret:
            raise ValueError("阿里云AccessKey未配置，请设置环境变量NLP_AK_ENV和NLP_SK_ENV")
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """分析文本情感"""
        try:
            # 优先使用SDK
            return self._analyze_with_sdk(text)
        except Exception as e:
            logger.warning(f"SDK分析失败，尝试HTTP请求: {e}")
            try:
                return self._analyze_with_http(text)
            except Exception as e2:
                logger.error(f"HTTP请求也失败: {e2}")
                # 返回默认结果
                return {
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'confidence': 0.0,
                    'error': f"API连接失败: {e2}",
                    'method': 'aliyun'
                }
    
    def _analyze_with_sdk(self, text: str) -> Dict[str, Union[str, float]]:
        """使用SDK分析"""
        try:
            from aliyunsdkcore.client import AcsClient
            from aliyunsdkcore.request import CommonRequest
            
            # 创建AcsClient实例
            client = AcsClient(
                self.access_key_id,
                self.access_key_secret,
                self.region_id
            )
            
            # 使用CommonRequest
            request = CommonRequest()
            request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
            request.set_version('2020-06-29')
            request.set_action_name('GetSaChGeneral')
            request.add_query_param('ServiceCode', 'alinlp')
            request.add_query_param('Text', text)
            
            # 发送请求
            response = client.do_action_with_exception(request)
            result = json.loads(response)
            try:
                rid = result.get('RequestId') or result.get('RequestID')
                logger.info(f"[Aliyun SA SDK] RequestId={rid}")
            except Exception:
                pass
            
            return self._parse_response(result)
            
        except ImportError:
            raise Exception("阿里云SDK未安装，请运行: pip install aliyun-python-sdk-core")
        except Exception as e:
            raise e
    
    def _analyze_with_http(self, text: str) -> Dict[str, Union[str, float]]:
        """使用HTTP请求分析"""
        import requests
        
        params = {
            'Action': 'SentimentAnalysis',
            'Version': '2018-04-08',
            'Format': 'JSON',
            'Timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'SignatureMethod': 'HMAC-SHA1',
            'SignatureVersion': '1.0',
            'SignatureNonce': str(int(time.time() * 1000)),
            'AccessKeyId': self.access_key_id,
            'Text': text,
        }
        
        # 生成签名
        signature = self._generate_signature('POST', '/', params)
        params['Signature'] = signature
        
        try:
            response = requests.post(self.endpoint, data=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            try:
                rid = result.get('RequestId') or result.get('RequestID')
                logger.info(f"[Aliyun SA HTTP] RequestId={rid}")
            except Exception:
                pass
            return self._parse_response(result)
        except Exception as e:
            raise e
    
    def _generate_signature(self, method: str, path: str, params: Dict) -> str:
        """生成签名"""
        canonicalized_query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        string_to_sign = f"{method}\n{path}\n{canonicalized_query_string}\n"
        
        signature = hmac.new(
            self.access_key_secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha1
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')
    
    def _parse_response(self, result: Dict) -> Dict[str, Union[str, float]]:
        """解析响应"""
        try:
            # 解析Data字段
            data_str = result.get('Data', '{}')
            if isinstance(data_str, str):
                data = json.loads(data_str)
            else:
                data = data_str
            
            # 获取结果
            result_data = data.get('result', {})
            sentiment_zh = result_data.get('sentiment', '')
            positive_prob = float(result_data.get('positive_prob', 0))
            negative_prob = float(result_data.get('negative_prob', 0))
            neutral_prob = float(result_data.get('neutral_prob', 0))
            
            # 映射情感
            sentiment_map = {
                'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral',
                '正向': 'positive', '负向': 'negative', '中性': 'neutral',
                '正面': 'positive', '负面': 'negative',
            }
            
            sentiment = sentiment_map.get(sentiment_zh.lower(), 'neutral')
            
            # 计算分数和置信度
            if sentiment == 'positive':
                score = positive_prob
                confidence = positive_prob
            elif sentiment == 'negative':
                score = -negative_prob
                confidence = negative_prob
            else:
                score = 0.0
                confidence = neutral_prob
            
            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': confidence,
                'positive_prob': positive_prob,
                'negative_prob': negative_prob,
                'neutral_prob': neutral_prob,
                'method': 'aliyun'
            }
        except Exception as e:
            raise e

class SentimentAnalyzer:
    """统一情感分析器"""
    
    def __init__(self, analyzer_type: str = "dictionary", video_id: Optional[str] = None,
                 sa_concurrency: int = 8, sa_batch_size: int = 200, sa_throttle_ms: int = 0):
        """
        初始化情感分析器
        
        Args:
            analyzer_type: 分析器类型 ("dictionary" 或 "aliyun")
        """
        self.analyzer_type = analyzer_type
        self.video_id = video_id
        self.sa_concurrency = max(1, sa_concurrency)
        self.sa_batch_size = max(1, sa_batch_size)
        self.sa_throttle_ms = max(0, sa_throttle_ms)
        self._stats_lock = threading.Lock()
        
        if analyzer_type == "dictionary":
            self.analyzer = DictionaryAnalyzer()
        elif analyzer_type == "aliyun":
            self.analyzer = AliyunAnalyzer()
        else:
            raise ValueError(f"不支持的分析器类型: {analyzer_type}")
        
        self.stats = {
            'total_analyzed': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'total_confidence': 0.0,
            'total_score': 0.0,
            'errors': 0
        }
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """分析单个文本"""
        try:
            result = self.analyzer.analyze_text(text)
            self._update_stats(result)
            return result
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            self.stats['errors'] += 1
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'error': str(e),
                'method': self.analyzer_type
            }
    
    def analyze_comments(self, conn, video_id: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        分析评论数据
        
        Args:
            conn: 数据库连接
            video_id: 视频ID，如果为None则分析所有评论
            limit: 限制分析数量，如果为None则分析所有评论
        """
        print("=== 从数据库加载评论数据 ===")
        
        if video_id:
            # 分析指定视频的评论
            sql = """
            SELECT comment_id, content, create_time, like_count, aweme_id
            FROM douyin_aweme_comment
            WHERE content IS NOT NULL AND LENGTH(content) > 5 AND aweme_id = %s
            ORDER BY create_time DESC
            """
            params = [video_id]
            print(f"分析视频 {video_id} 的评论...")
        else:
            # 分析所有评论
            sql = """
            SELECT comment_id, content, create_time, like_count, aweme_id
            FROM douyin_aweme_comment
            WHERE content IS NOT NULL AND LENGTH(content) > 5
            ORDER BY create_time DESC
            """
            params = []
            print("分析所有评论...")
        
        if limit:
            sql += f" LIMIT {limit}"
            print(f"限制分析数量: {limit}")
        
        try:
            df = pd.read_sql_query(sql, conn, params=params)
            print(f"[OK] 成功加载 {len(df)} 条评论")
            
            if df.empty:
                print("[WARN] 没有找到符合条件的评论")
                return df
            
            # 并发进行情感分析
            print("=== 开始情感分析（并发） ===")
            df = self.analyze_dataframe(df)
            
            print("[OK] 情感分析完成")
            return df
            
        except Exception as e:
            print(f"[ERR] 数据库分析失败: {e}")
            return pd.DataFrame()
    
    def _update_stats(self, result: Dict[str, Union[str, float]]):
        """更新统计信息"""
        with self._stats_lock:
            self.stats['total_analyzed'] += 1
            self.stats['total_confidence'] += result.get('confidence', 0.0)
            self.stats['total_score'] += result.get('score', 0.0)
            sentiment = result.get('sentiment', 'neutral')
            if sentiment == 'positive':
                self.stats['positive_count'] += 1
            elif sentiment == 'negative':
                self.stats['negative_count'] += 1
            else:
                self.stats['neutral_count'] += 1

    def _analyze_texts_concurrent(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """并发批量情感分析（去重+线程池）"""
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
        unique_list: List[str] = [None] * len(unique_index)  # type: ignore
        for t, u in unique_index.items():
            unique_list[u] = t

        unique_results: List[Dict[str, Union[str, float]]] = [None] * len(unique_list)  # type: ignore

        def _process_range(start: int, end: int):
            print(f"[RUN] 情感API {start+1}-{end}/{len(unique_list)}")
            with ThreadPoolExecutor(max_workers=self.sa_concurrency) as ex:
                futures = {}
                for idx in range(start, end):
                    # 可选节流（默认0=不限制）
                    if self.sa_throttle_ms > 0 and (idx - start) % self.sa_concurrency == 0:
                        time.sleep(self.sa_throttle_ms / 1000.0)
                    futures[ex.submit(self.analyze_text, unique_list[idx])] = idx
                for fut in as_completed(futures):
                    uid = futures[fut]
                    try:
                        unique_results[uid] = fut.result()
                    except Exception as e:
                        unique_results[uid] = {
                            'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0,
                            'error': str(e), 'method': self.analyzer_type
                        }

        for i in range(0, len(unique_list), self.sa_batch_size):
            _process_range(i, min(i + self.sa_batch_size, len(unique_list)))

        # 回填结果
        result_map: Dict[str, Dict[str, Union[str, float]]] = {
            t: unique_results[u] for t, u in unique_index.items()
        }
        return [result_map[order_to_text[i]] for i in range(n)]

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """对DataFrame的content列并发执行情感分析并填充结果列"""
        if 'content' not in df.columns:
            return df
        texts = df['content'].astype(str).tolist()
        results = self._analyze_texts_concurrent(texts)
        df = df.copy()
        df['sentiment'] = [r.get('sentiment', 'neutral') for r in results]
        df['sentiment_score'] = [r.get('score', 0.0) for r in results]
        df['sentiment_confidence'] = [r.get('confidence', 0.0) for r in results]
        return df
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['total_analyzed'] > 0:
            stats['avg_confidence'] = stats['total_confidence'] / stats['total_analyzed']
            stats['avg_score'] = stats['total_score'] / stats['total_analyzed']
        else:
            stats['avg_confidence'] = 0.0
            stats['avg_score'] = 0.0
        return stats
    
    def save_results(self, df: pd.DataFrame, output_dir: str = None):
        """保存分析结果"""
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'data', 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名（results_sentiment_[videoId]_timestamp）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.video_id:
            base_filename = f"results_sentiment_{self.video_id}_{timestamp}"
        else:
            base_filename = f"results_sentiment_{timestamp}"
        
        # 保存为CSV
        csv_file = os.path.join(output_dir, f"{base_filename}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"[OK] 结果已保存到: {csv_file}")
        
        # 保存为JSON
        json_file = os.path.join(output_dir, f"{base_filename}.json")
        results = {
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analyzer_type': self.analyzer_type,
            'total_comments': len(df),
            'stats': self.get_stats(),
            'data': df.to_dict('records')
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[OK] 结果已保存到: {json_file}")
        
        return csv_file, json_file
    
    def generate_report(self, df: pd.DataFrame, output_dir: str = None):
        """生成分析报告"""
        print("=== 生成分析报告 ===")
        
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'data', 'reports')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        stats = self.get_stats()
        sentiment_counts = df['sentiment'].value_counts()
        
        # 生成报告
        report = {
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analyzer_type': self.analyzer_type,
            'summary': {
                'total_comments': len(df),
                'positive_count': int(sentiment_counts.get('positive', 0)),
                'negative_count': int(sentiment_counts.get('negative', 0)),
                'neutral_count': int(sentiment_counts.get('neutral', 0)),
                'avg_confidence': float(stats['avg_confidence']),
                'avg_score': float(stats['avg_score'])
            },
            'sentiment_distribution': sentiment_counts.to_dict(),
            'top_positive_comments': df[df['sentiment'] == 'positive'].nlargest(5, 'sentiment_score')[['content', 'sentiment_score']].to_dict('records'),
            'top_negative_comments': df[df['sentiment'] == 'negative'].nsmallest(5, 'sentiment_score')[['content', 'sentiment_score']].to_dict('records'),
        }
        
        # 保存报告（reports_sentiment_[videoId]_timestamp.json）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.video_id:
            report_file = os.path.join(output_dir, f"reports_sentiment_{self.video_id}_{timestamp}.json")
        else:
            report_file = os.path.join(output_dir, f"reports_sentiment_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] 分析报告已保存到: {report_file}")
        
        # 打印报告摘要
        print("\n=== 分析报告摘要 ===")
        print(f"分析时间: {report['analysis_time']}")
        print(f"分析器类型: {report['analyzer_type']}")
        print(f"总评论数: {report['summary']['total_comments']:,}")
        print(f"正向评论: {report['summary']['positive_count']:,} ({report['summary']['positive_count']/report['summary']['total_comments']*100:.1f}%)")
        print(f"负向评论: {report['summary']['negative_count']:,} ({report['summary']['negative_count']/report['summary']['total_comments']*100:.1f}%)")
        print(f"中性评论: {report['summary']['neutral_count']:,} ({report['summary']['neutral_count']/report['summary']['total_comments']*100:.1f}%)")
        print(f"平均置信度: {report['summary']['avg_confidence']:.3f}")
        print(f"平均情感分数: {report['summary']['avg_score']:.3f}")
        
        return report_file
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str = None):
        """创建可视化图表"""
        print("=== 创建可视化图表 ===")
        
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'data', 'visualizations')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('情感分析结果可视化', fontsize=16, fontweight='bold')
        
        # 1. 情感分布饼图
        ax1 = axes[0, 0]
        if 'sentiment' in df.columns and not df['sentiment'].empty:
            sentiment_counts = df['sentiment'].value_counts()
            if sentiment_counts.sum() > 0:
                colors = ['lightgreen', 'lightcoral', 'lightblue']
                wedges, texts, autotexts = ax1.pie(sentiment_counts.values, labels=sentiment_counts.index,
                                                  autopct='%1.1f%%', colors=colors, startangle=90)
                ax1.set_title('情感分布')
            else:
                ax1.text(0.5, 0.5, '无情感分布数据', ha='center', va='center', transform=ax1.transAxes)
                ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, '无情感分布数据', ha='center', va='center', transform=ax1.transAxes)
            ax1.axis('off')
        
        # 2. 情感分数分布直方图
        ax2 = axes[0, 1]
        if 'sentiment_score' in df.columns and df['sentiment_score'].notna().any():
            ax2.hist(df['sentiment_score'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('情感分数')
            ax2.set_ylabel('频次')
            ax2.set_title('情感分数分布')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无情感分数数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        # 3. 置信度分布
        ax3 = axes[1, 0]
        if 'confidence' in df.columns and df['confidence'].notna().any():
            ax3.hist(df['confidence'].dropna(), bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_xlabel('置信度')
            ax3.set_ylabel('频次')
            ax3.set_title('置信度分布')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无置信度数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        # 4. 情感分数vs置信度散点图
        ax4 = axes[1, 1]
        if all(col in df.columns for col in ['sentiment_score', 'confidence']) and (
            df['sentiment_score'].notna().any() and df['confidence'].notna().any()):
            scatter = ax4.scatter(df['sentiment_score'], df['confidence'],
                                  c=df['sentiment_score'], cmap='RdYlGn', alpha=0.6)
            ax4.set_xlabel('情感分数')
            ax4.set_ylabel('置信度')
            ax4.set_title('情感分数 vs 置信度')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4)
        else:
            ax4.text(0.5, 0.5, '无散点图数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存图表（visualizations_sentiment_[videoId]_timestamp.png）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.video_id:
            output_file = os.path.join(output_dir, f'visualizations_sentiment_{self.video_id}_{timestamp}.png')
        else:
            output_file = os.path.join(output_dir, f'visualizations_sentiment_{timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[OK] 可视化图表已保存到: {output_file}")
        
        # 关闭图表，不显示
        plt.close()
        
        return output_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化版情感分析工具")
    parser.add_argument('--type', choices=['local', 'aliyun'], 
                       default='local', help='分析器类型：local(本地词典) 或 aliyun(阿里云API)')
    parser.add_argument('--video-id', type=str, help='视频ID，如果不指定则分析所有评论')
    parser.add_argument('--limit', type=int, help='限制分析数量')
    parser.add_argument('--use-cleaned-data', action='store_true', help='使用清洗后的数据文件')
    parser.add_argument('--cleaned-data-path', type=str, help='清洗数据文件路径')
    parser.add_argument('--test', action='store_true', help='测试模式，只分析少量数据')
    # 并发参数（默认高并发，无限流）
    parser.add_argument('--sa-concurrency', type=int, default=8, help='情感API并发数，默认8')
    parser.add_argument('--sa-batch-size', type=int, default=200, help='情感API批大小，默认200')
    parser.add_argument('--sa-throttle-ms', type=int, default=0, help='情感API节流毫秒，默认0=不限制')
    
    args = parser.parse_args()
    
    # 测试模式设置
    if args.test:
        if not args.limit:
            args.limit = 10
        print("[TEST] 测试模式：只分析少量数据")
    
    # 显示配置信息
    print("=== 优化版情感分析工具 ===")
    print(f"分析器类型: {args.type}")
    if args.video_id:
        print(f"视频ID: {args.video_id}")
    if args.limit:
        print(f"限制数量: {args.limit}")
    if args.use_cleaned_data:
        print("使用清洗数据")
    print("=" * 30)
    
    # 检查阿里云API配置
    if args.type == 'aliyun':
        access_key_id = os.getenv('NLP_AK_ENV')
        access_key_secret = os.getenv('NLP_SK_ENV')
        if not access_key_id or not access_key_secret:
            print("[ERR] 阿里云API密钥未配置")
            print("请设置环境变量：")
            print("  - NLP_AK_ENV: 阿里云AccessKey ID")
            print("  - NLP_SK_ENV: 阿里云AccessKey Secret")
            print("  - NLP_REGION_ENV: 阿里云区域ID (可选，默认为cn-hangzhou)")
            return
        print("[OK] 阿里云API环境变量已配置")
    
    # 创建分析器
    try:
        # 映射参数类型
        analyzer_type = "dictionary" if args.type == "local" else "aliyun"
        analyzer = SentimentAnalyzer(
            analyzer_type,
            video_id=args.video_id,
            sa_concurrency=max(1, args.sa_concurrency),
            sa_batch_size=max(1, args.sa_batch_size),
            sa_throttle_ms=max(0, args.sa_throttle_ms),
        )
        print("[OK] 情感分析器初始化成功")
    except Exception as e:
        print(f"[ERR] 初始化失败: {e}")
        return
    
    # 加载数据
    if args.use_cleaned_data:
        # 从清洗数据文件加载
        try:
            if args.cleaned_data_path:
                cleaned_data_path = args.cleaned_data_path
            else:
                auto_path = resolve_latest_cleaned_data(args.video_id)
                cleaned_data_path = auto_path or os.path.join(PROJECT_ROOT, 'data', 'processed', 'douyin_comments_processed.json')
            
            if not os.path.exists(cleaned_data_path):
                print(f"[ERR] 清洗数据文件不存在: {cleaned_data_path}")
                return
            
            with open(cleaned_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            print(f"[OK] 成功加载清洗数据: {len(df)} 条记录")
            
            # 限制数据量
            if args.limit and len(df) > args.limit:
                df = df.head(args.limit)
                print(f"[OK] 限制数据量: {len(df)} 条记录")
            
        except Exception as e:
            print(f"[ERR] 加载清洗数据失败: {e}")
            return
    else:
        # 从数据库加载
        try:
            conn = get_db_conn()
            print("[OK] 数据库连接成功")
        except Exception as e:
            print(f"[ERR] 数据库连接失败: {e}")
            return
        
        try:
            # 分析评论
            df = analyzer.analyze_comments(conn, args.video_id, args.limit)
            
            if df.empty:
                print("[ERR] 没有找到评论数据")
                return
        except Exception as e:
            print(f"[ERR] 从数据库加载数据失败: {e}")
            return
    
    # 执行分析
    try:
        # 对数据进行情感分析
        print("=== 开始情感分析 ===")
        results = []
        for idx, row in df.iterrows():
            result = analyzer.analyze_text(row['content'])
            results.append(result)
        
        # 将结果添加到DataFrame
        df['sentiment'] = [r['sentiment'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        print("[OK] 情感分析完成")
        
        # 保存结果
        analyzer.save_results(df)
        
        # 生成报告
        analyzer.generate_report(df)
        
        # 创建可视化
        analyzer.create_visualizations(df)
        
        print("\n[OK] 情感分析完成!")
        
    except Exception as e:
        print(f"[ERR] 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if not args.use_cleaned_data and 'conn' in locals():
            conn.close()
            print("[OK] 数据库连接已关闭")

if __name__ == "__main__":
    main()
