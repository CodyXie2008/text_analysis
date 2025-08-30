#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版点赞分析模块
从众心理点赞互动分析，支持视频ID和数据源选择
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from text_analysis.core.base_analyzer import BaseAnalyzer, create_parser, parse_common_args

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class LikeAnalysisAnalyzer(BaseAnalyzer):
    """点赞分析器"""
    
    def __init__(self, video_id: str = None):
        super().__init__("like", video_id)
        self.like_thresholds = {
            'low': 0,           # 低点赞：0
            'medium': 5,        # 中等点赞：5-19
            'high': 20,         # 高点赞：20-99
            'very_high': 100,   # 很高点赞：100-499
            'extreme': 500      # 极高点赞：500+
        }
        
        self.follow_speed_thresholds = {
            'immediate': 5,      # 立即跟随：5分钟内
            'quick': 30,         # 快速跟随：30分钟内
            'medium': 120,       # 中等跟随：2小时内
            'slow': 1440,        # 缓慢跟随：24小时内
            'delayed': 10080     # 延迟跟随：1周内
        }
    
    def _load_from_database(self, limit: Optional[int] = None) -> pd.DataFrame:
        """从数据库加载数据"""
        print("=== 从数据库加载评论数据 ===")
        
        if self.video_id:
            # 加载指定视频的评论
            sql = """
            SELECT 
                comment_id, aweme_id, parent_comment_id, content, create_time,
                like_count, sub_comment_count, user_id, nickname
            FROM douyin_aweme_comment
            WHERE content IS NOT NULL AND LENGTH(content) > 5 AND aweme_id = %s
            ORDER BY create_time ASC
            """
            params = [self.video_id]
            print(f"加载视频 {self.video_id} 的评论...")
        else:
            # 加载所有评论
            sql = """
            SELECT 
                comment_id, aweme_id, parent_comment_id, content, create_time,
                like_count, sub_comment_count, user_id, nickname
            FROM douyin_aweme_comment
            WHERE content IS NOT NULL AND LENGTH(content) > 5
            ORDER BY create_time ASC
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
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """执行点赞分析"""
        print("=== 开始点赞分析 ===")
        
        # 1. 分析点赞分布
        print("1. 分析点赞分布...")
        like_stats = self._analyze_like_distribution(df)
        
        # 2. 分析父评论点赞分布
        print("2. 分析父评论点赞分布...")
        like_ranges = self._analyze_parent_like_ranges(df)
        
        # 3. 识别意见领袖
        print("3. 识别意见领袖...")
        opinion_leaders = self._identify_opinion_leaders(df)
        
        # 4. 分析社会认同信号
        print("4. 分析社会认同信号...")
        approval_signals = self._analyze_social_approval_signals(df)
        
        # 5. 分析跟随速度
        print("5. 分析跟随速度...")
        follow_speed = self._analyze_follow_speed(df)
        
        return {
            'like_stats': like_stats,
            'like_ranges': like_ranges,
            'opinion_leaders': opinion_leaders,
            'approval_signals': approval_signals,
            'follow_speed': follow_speed,
            'total_comments': len(df),
            'parent_comments': len(df[df['parent_comment_id'].isna() | (df['parent_comment_id'] == '0') | (df['parent_comment_id'] == 0)]),
            'child_comments': len(df[df['parent_comment_id'].notna() & (df['parent_comment_id'] != '0') & (df['parent_comment_id'] != 0)])
        }
    
    def _analyze_like_distribution(self, df: pd.DataFrame) -> Dict:
        """分析点赞分布"""
        # 确保like_count是数值类型
        like_counts = pd.to_numeric(df['like_count'], errors='coerce').fillna(0).astype(int).values
        
        stats = {
            'mean': float(np.mean(like_counts)),
            'median': float(np.median(like_counts)),
            'std': float(np.std(like_counts)),
            'min': int(np.min(like_counts)),
            'max': int(np.max(like_counts)),
            'q25': float(np.percentile(like_counts, 25)),
            'q75': float(np.percentile(like_counts, 75))
        }
        
        # 点赞范围分布
        ranges = {
            '0点赞': len(like_counts[like_counts == 0]),
            '1-4点赞': len(like_counts[(like_counts >= 1) & (like_counts <= 4)]),
            '5-19点赞': len(like_counts[(like_counts >= 5) & (like_counts <= 19)]),
            '20-99点赞': len(like_counts[(like_counts >= 20) & (like_counts <= 99)]),
            '100-499点赞': len(like_counts[(like_counts >= 100) & (like_counts <= 499)]),
            '500+点赞': len(like_counts[like_counts >= 500])
        }
        
        return {
            'statistics': stats,
            'ranges': ranges
        }
    
    def _analyze_parent_like_ranges(self, df: pd.DataFrame) -> Dict:
        """分析父评论点赞范围"""
        # 分离父评论和子评论
        parent_comments = df[df['parent_comment_id'].isna() | (df['parent_comment_id'] == '0') | (df['parent_comment_id'] == 0)].copy()
        child_comments = df[df['parent_comment_id'].notna() & (df['parent_comment_id'] != '0') & (df['parent_comment_id'] != 0)].copy()
        
        if len(parent_comments) == 0 or len(child_comments) == 0:
            return {}
        
        # 创建父评论点赞映射
        parent_likes = parent_comments.set_index('comment_id')['like_count'].to_dict()
        
        # 为子评论添加父评论点赞数
        child_comments['parent_like_count'] = child_comments['parent_comment_id'].map(parent_likes)
        
        # 确保parent_like_count是数值类型
        child_comments['parent_like_count'] = pd.to_numeric(child_comments['parent_like_count'], errors='coerce').fillna(0).astype(int)
        
        # 过滤有效的子评论
        valid_child_comments = child_comments[child_comments['parent_like_count'].notna()]
        
        if len(valid_child_comments) == 0:
            return {}
        
        # 分析父评论点赞范围对子评论的影响
        like_ranges = {}
        for range_name, threshold in self.like_thresholds.items():
            if range_name == 'low':
                mask = valid_child_comments['parent_like_count'] == threshold
            elif range_name == 'extreme':
                mask = valid_child_comments['parent_like_count'] >= threshold
            else:
                next_threshold = list(self.like_thresholds.values())[list(self.like_thresholds.keys()).index(range_name) + 1]
                mask = (valid_child_comments['parent_like_count'] >= threshold) & (valid_child_comments['parent_like_count'] < next_threshold)
            
            range_comments = valid_child_comments[mask]
            if len(range_comments) > 0:
                # 确保like_count是数值类型
                child_likes = pd.to_numeric(range_comments['like_count'], errors='coerce').fillna(0).astype(int)
                parent_likes = range_comments['parent_like_count']
                
                like_ranges[range_name] = {
                    'count': len(range_comments),
                    'avg_child_likes': float(child_likes.mean()),
                    'avg_parent_likes': float(parent_likes.mean()),
                    'correlation': float(child_likes.corr(parent_likes)) if len(child_likes) > 1 else 0.0
                }
        
        return like_ranges
    
    def _identify_opinion_leaders(self, df: pd.DataFrame, like_threshold: int = 20, follow_speed_threshold: int = 30) -> Dict:
        """识别意见领袖"""
        # 分离父评论和子评论
        parent_comments = df[df['parent_comment_id'].isna() | (df['parent_comment_id'] == '0') | (df['parent_comment_id'] == 0)].copy()
        child_comments = df[df['parent_comment_id'].notna() & (df['parent_comment_id'] != '0') & (df['parent_comment_id'] != 0)].copy()
        
        if len(parent_comments) == 0 or len(child_comments) == 0:
            return {}
        
        # 转换时间格式
        parent_comments['create_time'] = pd.to_datetime(parent_comments['create_time'], unit='s')
        child_comments['create_time'] = pd.to_datetime(child_comments['create_time'], unit='s')
        
        # 创建父评论时间映射
        parent_times = parent_comments.set_index('comment_id')['create_time'].to_dict()
        
        # 计算跟随速度
        def get_follow_speed(row):
            parent_id = str(row['parent_comment_id'])
            if parent_id not in parent_times:
                return None
            
            parent_time = parent_times[parent_id]
            child_time = row['create_time']
            
            if pd.isna(parent_time) or pd.isna(child_time):
                return None
            
            time_diff = (child_time - parent_time).total_seconds() / 60
            return time_diff if time_diff >= 0 else None
        
        child_comments['follow_speed'] = child_comments.apply(get_follow_speed, axis=1)
        
        # 识别意见领袖
        opinion_leaders = []
        for _, parent in parent_comments.iterrows():
            parent_id = parent['comment_id']
            parent_likes = parent['like_count']
            parent_time = parent['create_time']
            
            # 找到该父评论的所有子评论
            children = child_comments[child_comments['parent_comment_id'] == parent_id]
            
            if len(children) == 0:
                continue
            
            # 计算跟随速度统计
            valid_follow_speeds = children[children['follow_speed'].notna()]['follow_speed']
            if len(valid_follow_speeds) == 0:
                continue
            
            avg_follow_speed = valid_follow_speeds.mean()
            quick_followers = len(valid_follow_speeds[valid_follow_speeds <= follow_speed_threshold])
            
            # 确保parent_likes是数值类型
            parent_likes_numeric = pd.to_numeric(parent_likes, errors='coerce')
            if pd.isna(parent_likes_numeric):
                continue
                
            # 判断是否为意见领袖
            if (parent_likes_numeric >= like_threshold and 
                len(children) >= 3 and 
                quick_followers >= len(children) * 0.3):
                
                opinion_leaders.append({
                    'comment_id': parent_id,
                    'content': parent['content'][:50] + '...' if len(parent['content']) > 50 else parent['content'],
                    'like_count': int(parent_likes),
                    'child_count': len(children),
                    'avg_follow_speed': float(avg_follow_speed),
                    'quick_followers': int(quick_followers),
                    'quick_follower_ratio': float(quick_followers / len(children))
                })
        
        # 按点赞数排序
        opinion_leaders.sort(key=lambda x: x['like_count'], reverse=True)
        
        return {
            'total_leaders': len(opinion_leaders),
            'leaders': opinion_leaders[:10]  # 返回前10个意见领袖
        }
    
    def _analyze_social_approval_signals(self, df: pd.DataFrame) -> Dict:
        """分析社会认同信号"""
        # 分离父评论和子评论
        parent_comments = df[df['parent_comment_id'].isna() | (df['parent_comment_id'] == '0') | (df['parent_comment_id'] == 0)].copy()
        child_comments = df[df['parent_comment_id'].notna() & (df['parent_comment_id'] != '0') & (df['parent_comment_id'] != 0)].copy()
        
        if len(parent_comments) == 0 or len(child_comments) == 0:
            return {}
        
        # 创建父评论点赞映射
        parent_likes = parent_comments.set_index('comment_id')['like_count'].to_dict()
        
        # 为子评论添加父评论点赞数
        child_comments['parent_like_count'] = child_comments['parent_comment_id'].map(parent_likes)
        
        # 确保数据类型正确
        child_comments['parent_like_count'] = pd.to_numeric(child_comments['parent_like_count'], errors='coerce').fillna(0).astype(int)
        child_comments['like_count'] = pd.to_numeric(child_comments['like_count'], errors='coerce').fillna(0).astype(int)
        
        valid_child_comments = child_comments[child_comments['parent_like_count'].notna()]
        
        if len(valid_child_comments) == 0:
            return {}
        
        # 分析社会认同信号
        high_like_mask = valid_child_comments['parent_like_count'] >= 50
        low_like_mask = valid_child_comments['parent_like_count'] < 10
        
        signals = {
            'high_like_parents': {
                'count': len(valid_child_comments[high_like_mask]),
                'avg_child_likes': float(valid_child_comments[high_like_mask]['like_count'].mean()) if len(valid_child_comments[high_like_mask]) > 0 else 0.0
            },
            'low_like_parents': {
                'count': len(valid_child_comments[low_like_mask]),
                'avg_child_likes': float(valid_child_comments[low_like_mask]['like_count'].mean()) if len(valid_child_comments[low_like_mask]) > 0 else 0.0
            }
        }
        
        # 计算点赞相关性
        correlation = valid_child_comments['like_count'].corr(valid_child_comments['parent_like_count'])
        signals['correlation'] = float(correlation) if not pd.isna(correlation) else 0.0
        
        return signals
    
    def _analyze_follow_speed(self, df: pd.DataFrame) -> Dict:
        """分析跟随速度"""
        # 分离父评论和子评论
        parent_comments = df[df['parent_comment_id'].isna() | (df['parent_comment_id'] == '0') | (df['parent_comment_id'] == 0)].copy()
        child_comments = df[df['parent_comment_id'].notna() & (df['parent_comment_id'] != '0') & (df['parent_comment_id'] != 0)].copy()
        
        if len(parent_comments) == 0 or len(child_comments) == 0:
            return {}
        
        # 转换时间格式
        parent_comments['create_time'] = pd.to_datetime(parent_comments['create_time'], unit='s')
        child_comments['create_time'] = pd.to_datetime(child_comments['create_time'], unit='s')
        
        # 创建父评论时间映射
        parent_times = parent_comments.set_index('comment_id')['create_time'].to_dict()
        
        # 计算跟随速度
        def get_follow_speed(row):
            parent_id = str(row['parent_comment_id'])
            if parent_id not in parent_times:
                return None
            
            parent_time = parent_times[parent_id]
            child_time = row['create_time']
            
            if pd.isna(parent_time) or pd.isna(child_time):
                return None
            
            time_diff = (child_time - parent_time).total_seconds() / 60
            return time_diff if time_diff >= 0 else None
        
        child_comments['follow_speed'] = child_comments.apply(get_follow_speed, axis=1)
        valid_child_comments = child_comments[child_comments['follow_speed'].notna()]
        
        if len(valid_child_comments) == 0:
            return {}
        
        follow_speeds = valid_child_comments['follow_speed'].values
        
        # 分析跟随速度分布
        speed_stats = {
            'mean': float(np.mean(follow_speeds)),
            'median': float(np.median(follow_speeds)),
            'std': float(np.std(follow_speeds)),
            'min': float(np.min(follow_speeds)),
            'max': float(np.max(follow_speeds))
        }
        
        # 跟随速度范围分布
        speed_ranges = {}
        for range_name, threshold in self.follow_speed_thresholds.items():
            if range_name == 'immediate':
                mask = follow_speeds <= threshold
            elif range_name == 'delayed':
                mask = follow_speeds > threshold
            else:
                next_threshold = list(self.follow_speed_thresholds.values())[list(self.follow_speed_thresholds.keys()).index(range_name) + 1]
                mask = (follow_speeds > threshold) & (follow_speeds <= next_threshold)
            
            speed_ranges[range_name] = {
                'count': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(follow_speeds) * 100)
            }
        
        return {
            'statistics': speed_stats,
            'ranges': speed_ranges
        }
    
    def _create_charts(self, df: pd.DataFrame, results: Dict, output_path: str):
        """创建点赞分析可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('从众心理点赞分析结果', fontsize=16, fontweight='bold')
        
        # 1. 点赞分布直方图
        ax1 = axes[0, 0]
        like_counts = df['like_count'].values
        ax1.hist(like_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('点赞数')
        ax1.set_ylabel('频次')
        ax1.set_title('点赞数分布')
        ax1.grid(True, alpha=0.3)
        
        # 2. 点赞范围分布饼图
        ax2 = axes[0, 1]
        if 'like_stats' in results and 'ranges' in results['like_stats']:
            like_ranges = results['like_stats']['ranges']
            labels = list(like_ranges.keys())
            sizes = list(like_ranges.values())
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'yellow', 'orange', 'red']
            
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title('点赞范围分布')
        
        # 3. 父评论点赞vs子评论点赞散点图
        ax3 = axes[1, 0]
        # 分离父评论和子评论
        parent_comments = df[df['parent_comment_id'].isna() | (df['parent_comment_id'] == '0') | (df['parent_comment_id'] == 0)]
        child_comments = df[df['parent_comment_id'].notna() & (df['parent_comment_id'] != '0') & (df['parent_comment_id'] != 0)]
        
        if len(parent_comments) > 0 and len(child_comments) > 0:
            # 创建父评论点赞映射
            parent_likes = parent_comments.set_index('comment_id')['like_count'].to_dict()
            child_comments['parent_like_count'] = child_comments['parent_comment_id'].map(parent_likes)
            valid_child_comments = child_comments[child_comments['parent_like_count'].notna()]
            
            if len(valid_child_comments) > 0:
                ax3.scatter(valid_child_comments['parent_like_count'], valid_child_comments['like_count'], 
                           alpha=0.6, color='lightgreen')
                ax3.set_xlabel('父评论点赞数')
                ax3.set_ylabel('子评论点赞数')
                ax3.set_title('父评论点赞 vs 子评论点赞')
                ax3.grid(True, alpha=0.3)
        
        # 4. 意见领袖统计
        ax4 = axes[1, 1]
        if 'opinion_leaders' in results and results['opinion_leaders']:
            leaders = results['opinion_leaders']['leaders'][:5]  # 显示前5个
            if leaders:
                leader_names = [f"领袖{i+1}" for i in range(len(leaders))]
                leader_likes = [leader['like_count'] for leader in leaders]
                
                ax4.bar(leader_names, leader_likes, color='lightcoral')
                ax4.set_xlabel('意见领袖')
                ax4.set_ylabel('点赞数')
                ax4.set_title('意见领袖点赞数')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    parser = create_parser("like", "从众心理点赞分析工具")
    args = parser.parse_args()
    
    # 解析通用参数
    args = parse_common_args(parser, args)
    
    # 创建分析器
    analyzer = LikeAnalysisAnalyzer(args.video_id)
    
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
