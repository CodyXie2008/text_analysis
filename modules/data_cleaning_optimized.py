#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版数据清洗模块
统一的数据准备与清洗流程，支持视频ID和数据源选择
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime
from typing import Dict, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import jieba
import logging
import matplotlib.pyplot as plt
import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from text_analysis.core.base_analyzer import BaseAnalyzer, create_parser, parse_common_args
from dotenv import load_dotenv, find_dotenv
try:
    from text_analysis.modules.aliyun_tokenizer_analyzer import AliyunTokenizerClient
except Exception:
    AliyunTokenizerClient = None  # 允许在未安装/未创建文件时回退到本地分词

# 抑制jieba调试日志
try:
    jieba.setLogLevel(logging.WARNING)
except Exception:
    logging.getLogger('jieba').setLevel(logging.WARNING)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DataCleaningAnalyzer(BaseAnalyzer):
    """数据清洗分析器"""
    
    def __init__(self, video_id: str = None, segment_mode: str = 'local'):
        super().__init__("cleaning", video_id)
        self.segment_mode = segment_mode  # 'local' | 'api'
        self.stop_words = self._load_stop_words()
        
        # 简化分词模式逻辑，默认使用本地分词
        if self.segment_mode == 'api':
            self._init_aliyun_tokenizer()
            if getattr(self, 'aliyun_client', None) is None:
                print('❌ API模式需要配置阿里云 AK/SK，自动回退到本地分词')
                self.segment_mode = 'local'
            else:
                print('✅ 分词模式: API（使用阿里云分词）')
        else:
            print('✅ 分词模式: 本地（使用 jieba 分词）')

    def _init_aliyun_tokenizer(self):
        """初始化阿里云分词客户端（从 .env / 环境变量读取）。若未配置则置空回退到本地 jieba。"""
        # 依次尝试加载根目录与 text_analysis 目录下的 .env
        # 1) 当前工作目录
        load_dotenv(override=False)
        # 2) 项目根目录 .env
        root_env = os.path.join(project_root, '.env')
        if os.path.exists(root_env):
            load_dotenv(dotenv_path=root_env, override=True)
        # 3) text_analysis 目录 .env
        ta_env = os.path.join(project_root, 'text_analysis', '.env')
        if os.path.exists(ta_env):
            load_dotenv(dotenv_path=ta_env, override=True)

        self.aliyun_client = None
        if AliyunTokenizerClient is None:
            return
        ak = os.getenv('NLP_AK_ENV')
        sk = os.getenv('NLP_SK_ENV')
        endpoint = os.getenv('NLP_REGION_ENV', 'cn-hangzhou')
        # 构建完整的endpoint URL
        endpoint = f'https://alinlp.{endpoint}.aliyuncs.com'
        self.aliyun_tokenizer_id = os.getenv('ALIYUN_TOKENIZER_ID', 'GENERAL_CHN')
        self.aliyun_out_type = os.getenv('ALIYUN_OUT_TYPE', '1')
        if ak and sk:
            try:
                self.aliyun_client = AliyunTokenizerClient(ak, sk, endpoint)
                print('✅ 已启用阿里云分词 API (env 已加载)')
            except Exception as e:
                print(f"❌ 初始化阿里云分词失败，回退本地分词: {e}")
        
    def _load_stop_words(self):
        """加载停用词"""
        stop_words = set()
        try:
            # 仅从 modules 目录加载停用词
            module_dir = os.path.dirname(os.path.abspath(__file__))
            stop_words_file = os.path.join(module_dir, 'hit_stopwords.txt')
            if os.path.exists(stop_words_file):
                with open(stop_words_file, 'r', encoding='utf-8') as f:
                    stop_words = set(line.strip() for line in f if line.strip())
            else:
                # 使用基础停用词
                stop_words = {
                    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
                    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
                    '自己', '这', '那', '他', '她', '它', '们', '们', '个', '只', '条', '张', '片'
                }
        except Exception as e:
            print(f"加载停用词失败: {e}")
            stop_words = set()
        return stop_words
    
    def _load_from_database(self, limit: Optional[int] = None) -> pd.DataFrame:
        """从数据库加载数据"""
        print("=== 从数据库加载评论数据 ===")
        
        if self.video_id:
            # 加载指定视频的评论
            sql = """
            SELECT 
                comment_id, aweme_id, parent_comment_id, content, create_time,
                like_count, sub_comment_count, user_id, nickname, user_signature
            FROM douyin_aweme_comment
            WHERE content IS NOT NULL AND LENGTH(content) > 0 AND aweme_id = %s
            ORDER BY create_time DESC
            """
            params = [self.video_id]
            print(f"加载视频 {self.video_id} 的评论...")
        else:
            # 加载所有评论
            sql = """
            SELECT 
                comment_id, aweme_id, parent_comment_id, content, create_time,
                like_count, sub_comment_count, user_id, nickname, user_signature
            FROM douyin_aweme_comment
            WHERE content IS NOT NULL AND LENGTH(content) > 0
            ORDER BY create_time DESC
            """
            params = []
            print("加载所有评论...")
        
        if limit:
            sql += f" LIMIT {limit}"
            print(f"限制加载数量: {limit}")
        
        try:
            df = pd.read_sql_query(sql, self.conn, params=params)
            
            # 数据类型转换
            df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0).astype(int)
            df['sub_comment_count'] = pd.to_numeric(df['sub_comment_count'], errors='coerce').fillna(0).astype(int)
            df['create_time'] = pd.to_numeric(df['create_time'], errors='coerce')
            
            print(f"✅ 成功加载 {len(df)} 条评论")
            return df
        except Exception as e:
            print(f"❌ 数据库加载失败: {e}")
            return pd.DataFrame()
    
    def is_spam_comment(self, content: str, user_signature: str = None) -> bool:
        """判断是否为垃圾评论"""
        if not content:
            return True
        
        # 内容长度检查
        if len(content.strip()) < 5:
            return True
        
        # 特殊字符比例检查
        special_char_ratio = len(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', content)) / len(content)
        if special_char_ratio > 0.6:
            return True
        
        # 广告关键词检查
        ad_keywords = [
            '加微信', '加qq', '加群', '私信', '联系我', '合作', '推广', '广告',
            '赚钱', '兼职', '代理', '加盟', '招商', '投资', '理财', '贷款',
            '免费领取', '限时优惠', '点击链接', '扫码关注', '关注公众号'
        ]
        content_lower = content.lower()
        for keyword in ad_keywords:
            if keyword in content_lower:
                return True
        
        # 重复字符检查
        if len(set(content)) < len(content) * 0.2:
            return True
        
        # 用户签名检查
        if user_signature:
            if len(user_signature) > 100:
                return True
            if re.search(r'[0-9]{8,}', user_signature):
                return True
        
        return False
    
    def clean_text(self, text: str) -> str:
        """文本清洗"""
        if not text:
            return ""
        
        # 1. 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 3. 去除方括号内容（如[666][比心]等）
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # 4. 去除多余空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 5. 保留中文、英文、数字、中文标点，去除其他符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\u3000-\u303f\uff00-\uffef]', '', text)
        
        return text
    
    def segment_text(self, text: str) -> List[str]:
        """文本分词"""
        if not text:
            return []
        
        # 简化分词逻辑，默认使用本地分词
        if self.segment_mode == 'api' and getattr(self, 'aliyun_client', None) is not None:
            try:
                text_cut = text[:1024]
                resp = self.aliyun_client.get_ws_ch_general(text_cut, self.aliyun_tokenizer_id, self.aliyun_out_type)
                data_field = resp.get('Data')
                payload = json.loads(data_field) if isinstance(data_field, str) else data_field
                result = payload.get('result', []) if isinstance(payload, dict) else []
                words = [item.get('word') for item in result if isinstance(item, dict) and item.get('word')]
                try:
                    req_id = resp.get('RequestId')
                    sample_words = words[:3]
                    print(f"[AliyunSeg OK] RequestId={req_id} words={sample_words}")
                except Exception:
                    pass
            except Exception:
                # API调用失败时回退到本地分词
                words = jieba.lcut(text)
        else:
            # 使用本地 jieba 分词
            words = jieba.lcut(text)
        
        # 过滤停用词和短词
        filtered_words = [word for word in words if word and word.strip() and len(word.strip()) > 1 and word not in self.stop_words]
        
        return filtered_words
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """执行数据清洗分析"""
        print("=== 开始数据清洗分析 ===")
        
        # 记录原始数据统计
        original_stats = {
            'total_comments': int(len(df)),
            'unique_videos': int(df['aweme_id'].nunique()),
            'unique_users': int(df['user_id'].nunique()),
            'avg_content_length': float(df['content'].str.len().mean()),
            'avg_likes': float(df['like_count'].mean()),
            'avg_sub_comments': float(df['sub_comment_count'].mean())
        }
        try:
            # 父子评论统计（基于 parent_comment_id 是否为 0/空）
            parent_mask = df['parent_comment_id'].isna() | (df['parent_comment_id'] == '0') | (df['parent_comment_id'] == 0)
            child_mask = ~(parent_mask)
            original_stats['parent_comments'] = int(parent_mask.sum())
            original_stats['child_comments'] = int(child_mask.sum())
        except Exception:
            pass
        
        # 1. 数据质量检查
        print("1. 数据质量检查...")
        quality_stats = self._check_data_quality(df)
        
        # 2. 垃圾评论过滤
        print("2. 垃圾评论过滤...")
        df_filtered = self._filter_spam_comments(df)
        spam_stats = {
            'spam_count': int(len(df) - len(df_filtered)),
            'spam_ratio': float((len(df) - len(df_filtered)) / len(df) * 100)
        }
        
        # 3. 文本清洗
        print("3. 文本清洗...")
        df_cleaned = self._clean_and_process_data(df_filtered)
        cleaning_stats = {
            'cleaned_count': int(len(df_cleaned)),
            'cleaning_ratio': float(len(df_cleaned) / len(df_filtered) * 100)
        }
        
        # 4. 分词处理
        print("4. 分词处理...")
        df_processed = self._add_word_segments(df_cleaned)
        
        # 5. 保存清洗后的数据
        print("5. 保存清洗数据...")
        cleaned_data_path = self.path_manager.get_cleaned_data_path()
        self._save_cleaned_data(df_processed, cleaned_data_path)
        
        # 汇总统计
        final_stats = {
            'total_comments': int(len(df_processed)),
            'unique_videos': int(df_processed['aweme_id'].nunique()),
            'unique_users': int(df_processed['user_id'].nunique()),
            'avg_content_length': float(df_processed['content'].str.len().mean()),
            'avg_likes': float(df_processed['like_count'].mean()),
            'avg_sub_comments': float(df_processed['sub_comment_count'].mean()),
            'avg_word_count': float(df_processed['word_count'].mean())
        }
        try:
            parent_mask_p = df_processed['parent_comment_id'].isna() | (df_processed['parent_comment_id'] == '0') | (df_processed['parent_comment_id'] == 0)
            child_mask_p = ~(parent_mask_p)
            final_stats['parent_comments'] = int(parent_mask_p.sum())
            final_stats['child_comments'] = int(child_mask_p.sum())
        except Exception:
            pass
        
        return {
            'original_stats': original_stats,
            'quality_stats': quality_stats,
            'spam_stats': spam_stats,
            'cleaning_stats': cleaning_stats,
            'final_stats': final_stats,
            'cleaned_data_path': cleaned_data_path
        }
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict:
        """检查数据质量"""
        stats = {
            'null_content': int(df['content'].isnull().sum()),
            'empty_content': int((df['content'] == '').sum()),
            'short_content': int((df['content'].str.len() < 5).sum()),
            'duplicate_content': int(df['content'].duplicated().sum()),
            'missing_user_id': int(df['user_id'].isnull().sum()),
            'missing_aweme_id': int(df['aweme_id'].isnull().sum())
        }
        
        print(f"   - 空内容: {stats['null_content']}")
        print(f"   - 短内容: {stats['short_content']}")
        print(f"   - 重复内容: {stats['duplicate_content']}")
        
        return stats
    
    def _filter_spam_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤垃圾评论"""
        # 应用垃圾评论检测
        df['is_spam'] = df.apply(
            lambda row: self.is_spam_comment(row['content'], row.get('user_signature')), 
            axis=1
        )
        
        # 过滤垃圾评论
        df_filtered = df[~df['is_spam']].copy()
        df_filtered = df_filtered.drop('is_spam', axis=1)
        
        print(f"   - 过滤垃圾评论: {len(df) - len(df_filtered)} 条")
        
        return df_filtered
    
    def _clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗和处理数据"""
        df_cleaned = df.copy()
        
        # 文本清洗
        df_cleaned['content_cleaned'] = df_cleaned['content'].apply(self.clean_text)
        
        # 过滤清洗后为空的内容
        df_cleaned = df_cleaned[df_cleaned['content_cleaned'].str.len() > 0]
        
        # 更新content字段
        df_cleaned['content'] = df_cleaned['content_cleaned']
        df_cleaned = df_cleaned.drop('content_cleaned', axis=1)
        
        print(f"   - 清洗后保留: {len(df_cleaned)} 条")
        
        return df_cleaned
    
    def _add_word_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加分词结果"""
        df_processed = df.copy()
        
        # 分词
        df_processed['words'] = df_processed['content'].apply(self.segment_text)
        df_processed['word_count'] = df_processed['words'].apply(len)
        
        # 过滤词数过少的评论
        df_processed = df_processed[df_processed['word_count'] >= 2]
        
        print(f"   - 分词后保留: {len(df_processed)} 条")
        
        return df_processed
    
    def _save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        """保存清洗后的数据"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 转换为JSON格式保存
            data = df.to_dict('records')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 清洗数据已保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 保存清洗数据失败: {e}")
    
    def _create_charts(self, df: pd.DataFrame, results: Dict, output_path: str):
        """创建数据清洗可视化图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('数据清洗分析结果', fontsize=16, fontweight='bold')
            
            # 1. 数据质量分布
            ax1 = axes[0, 0]
            quality_stats = results.get('quality_stats', {})
            labels = ['空内容', '短内容', '重复内容', '缺失用户ID', '缺失视频ID']
            values = [
                quality_stats.get('null_content', 0), 
                quality_stats.get('empty_content', 0), 
                quality_stats.get('short_content', 0), 
                quality_stats.get('missing_user_id', 0), 
                quality_stats.get('missing_aweme_id', 0)
            ]
            
            # 过滤掉值为0的项
            non_zero_indices = [i for i, v in enumerate(values) if v > 0]
            if non_zero_indices:
                filtered_labels = [labels[i] for i in non_zero_indices]
                filtered_values = [values[i] for i in non_zero_indices]
                colors = ['red', 'orange', 'yellow', 'lightblue', 'lightgreen']
                filtered_colors = [colors[i] for i in non_zero_indices]
                
                ax1.bar(filtered_labels, filtered_values, color=filtered_colors)
                ax1.set_title('数据质量问题分布')
                ax1.tick_params(axis='x', rotation=45)
            else:
                ax1.text(0.5, 0.5, '无数据质量问题', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.axis('off')
        
            # 2. 清洗流程统计
            ax2 = axes[0, 1]
            original_stats = results.get('original_stats', {})
            spam_stats = results.get('spam_stats', {})
            final_stats = results.get('final_stats', {})
            
            original_count = original_stats.get('total_comments', 0)
            filtered_count = original_count - spam_stats.get('spam_count', 0)
            cleaned_count = final_stats.get('total_comments', 0)
            
            stages = ['原始数据', '过滤垃圾', '文本清洗', '最终数据']
            counts = [original_count, filtered_count, cleaned_count, cleaned_count]
            
            ax2.plot(stages, counts, marker='o', linewidth=2, markersize=8)
            ax2.set_title('数据清洗流程统计')
            ax2.set_ylabel('评论数量')
            ax2.grid(True, alpha=0.3)
        
            # 3. 内容长度分布
            ax3 = axes[1, 0]
            if 'content' in df.columns and len(df) > 0:
                content_lengths = df['content'].str.len()
                if not content_lengths.isna().all():
                    ax3.hist(content_lengths.dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax3.set_xlabel('内容长度')
                    ax3.set_ylabel('频次')
                    ax3.set_title('内容长度分布')
                else:
                    ax3.text(0.5, 0.5, '无内容长度数据', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
                    ax3.set_xlim(0, 1)
                    ax3.set_ylim(0, 1)
                    ax3.axis('off')
            else:
                ax3.text(0.5, 0.5, '无内容数据', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
                ax3.axis('off')
            ax3.grid(True, alpha=0.3)
            
            # 4. 词数分布
            ax4 = axes[1, 1]
            if 'word_count' in df.columns and len(df) > 0:
                word_counts = df['word_count']
                if not word_counts.isna().all():
                    ax4.hist(word_counts.dropna(), bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                    ax4.set_xlabel('词数')
                    ax4.set_ylabel('频次')
                    ax4.set_title('词数分布')
                else:
                    ax4.text(0.5, 0.5, '无词数数据', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
                    ax4.set_xlim(0, 1)
                    ax4.set_ylim(0, 1)
                    ax4.axis('off')
            else:
                ax4.text(0.5, 0.5, '无词数数据', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.axis('off')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ 创建可视化图表失败: {e}")
            plt.close('all')  # 关闭所有图表

def main():
    """主函数"""
    parser = create_parser("cleaning", "数据清洗分析工具")
    # 添加分词模式参数：local（默认，本地分词）、api（阿里云分词）
    parser.add_argument('--segment-mode', type=str, choices=['local', 'api'], default='local', help='分词模式：local(默认)/api')
    args = parser.parse_args()
    
    # 解析通用参数
    args = parse_common_args(parser, args)
    
    # 创建分析器
    analyzer = DataCleaningAnalyzer(args.video_id, segment_mode=args.segment_mode)
    
    # 运行分析
    analyzer.run_analysis(
        limit=args.limit,
        use_cleaned_data=args.use_cleaned_data,
        cleaned_data_path=args.cleaned_data_path,
        save_results=not args.no_save,
        generate_report=not args.no_report,
        create_visualizations=not args.no_viz
    )

if __name__ == "__main__":
    main()
