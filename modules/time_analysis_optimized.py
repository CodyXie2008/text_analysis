#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版时间分析模块
从众心理时间分析，支持视频ID和数据源选择
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

class TimeAnalysisAnalyzer(BaseAnalyzer):
    """时间分析器"""
    
    def __init__(self, video_id: str = None):
        super().__init__("time", video_id)
        self.time_windows = {
            'immediate': (0, 5),      # 0-5分钟：立即从众
            'quick': (5, 30),         # 5-30分钟：快速从众
            'normal': (30, 120),      # 30分钟-2小时：正常从众
            'slow': (120, 1440),      # 2小时-24小时：慢速从众
            'delayed': (1440, 10080)  # 1-7天：延迟从众
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
            print(f"✅ 成功加载 {len(df)} 条评论")
            return df
        except Exception as e:
            print(f"❌ 数据库加载失败: {e}")
            return pd.DataFrame()
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """执行时间分析"""
        print("=== 开始时间分析 ===")
        
        # 1. 计算时间差
        print("1. 计算时间差...")
        df_with_time_diff = self._calculate_time_differences(df)
        
        # 2. 分析时间分布
        print("2. 分析时间分布...")
        time_distribution = self._analyze_time_distribution(df_with_time_diff)
        
        # 3. 计算窗口分布
        print("3. 计算窗口分布...")
        window_distribution = self._calculate_window_distribution(df_with_time_diff)
        
        # 4. 检测从众窗口
        print("4. 检测从众窗口...")
        conformity_windows = self._detect_conformity_windows(df_with_time_diff)
        
        # 5. 分析密集时段
        print("5. 分析密集时段...")
        dense_periods = self._find_dense_periods(df_with_time_diff)
        
        # 保存处理后的DataFrame用于可视化
        self.df_processed = df_with_time_diff
        
        return {
            'time_distribution': time_distribution,
            'window_distribution': window_distribution,
            'conformity_windows': conformity_windows,
            'dense_periods': dense_periods,
            'total_comments': len(df),
            'valid_child_comments': len(df_with_time_diff[df_with_time_diff['time_diff'].notna()])
        }
    
    def _calculate_time_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算子评论与父评论的时间差"""
        df_with_diff = df.copy()
        
        # 转换时间格式
        df_with_diff['create_time'] = pd.to_datetime(df_with_diff['create_time'], unit='s')
        
        # 创建父评论时间映射
        parent_times = df_with_diff.set_index('comment_id')['create_time'].to_dict()
        
        def get_time_diff(row):
            """计算时间差（分钟）"""
            if pd.isna(row['parent_comment_id']) or row['parent_comment_id'] == '0' or row['parent_comment_id'] == 0:
                return None
            
            parent_id = str(row['parent_comment_id'])
            if parent_id not in parent_times:
                return None
            
            parent_time = parent_times[parent_id]
            child_time = row['create_time']
            
            if pd.isna(parent_time) or pd.isna(child_time):
                return None
            
            time_diff = (child_time - parent_time).total_seconds() / 60  # 转换为分钟
            return time_diff if time_diff >= 0 else None
        
        # 计算时间差
        df_with_diff['time_diff'] = df_with_diff.apply(get_time_diff, axis=1)
        
        # 过滤有效的子评论
        valid_child_comments = df_with_diff[df_with_diff['time_diff'].notna()]
        print(f"   - 有效子评论: {len(valid_child_comments)} 条")
        
        return df_with_diff
    
    def _analyze_time_distribution(self, df: pd.DataFrame) -> Dict:
        """分析时间分布"""
        valid_comments = df[df['time_diff'].notna()]
        
        if len(valid_comments) == 0:
            return {}
        
        time_diffs = valid_comments['time_diff'].values
        
        stats = {
            'mean': float(np.mean(time_diffs)),
            'median': float(np.median(time_diffs)),
            'std': float(np.std(time_diffs)),
            'min': float(np.min(time_diffs)),
            'max': float(np.max(time_diffs)),
            'q25': float(np.percentile(time_diffs, 25)),
            'q75': float(np.percentile(time_diffs, 75))
        }
        
        # 时间分布统计
        distribution = {
            '0-5分钟': len(time_diffs[time_diffs <= 5]),
            '5-30分钟': len(time_diffs[(time_diffs > 5) & (time_diffs <= 30)]),
            '30分钟-2小时': len(time_diffs[(time_diffs > 30) & (time_diffs <= 120)]),
            '2小时-24小时': len(time_diffs[(time_diffs > 120) & (time_diffs <= 1440)]),
            '1-7天': len(time_diffs[(time_diffs > 1440) & (time_diffs <= 10080)]),
            '7天以上': len(time_diffs[time_diffs > 10080])
        }
        
        return {
            'statistics': stats,
            'distribution': distribution
        }
    
    def _calculate_window_distribution(self, df: pd.DataFrame) -> Dict:
        """计算窗口分布"""
        valid_comments = df[df['time_diff'].notna()]
        
        if len(valid_comments) == 0:
            return {}
        
        window_counts = defaultdict(int)
        
        for _, row in valid_comments.iterrows():
            time_diff = row['time_diff']
            
            if time_diff <= 5:
                window_counts['immediate'] += 1
            elif time_diff <= 30:
                window_counts['quick'] += 1
            elif time_diff <= 120:
                window_counts['normal'] += 1
            elif time_diff <= 1440:
                window_counts['slow'] += 1
            elif time_diff <= 10080:
                window_counts['delayed'] += 1
        
        # 计算百分比
        total = len(valid_comments)
        window_percentages = {k: (v / total * 100) for k, v in window_counts.items()}
        
        return {
            'counts': dict(window_counts),
            'percentages': window_percentages
        }
    
    def _detect_conformity_windows(self, df: pd.DataFrame, threshold_percentage: float = 10) -> Dict:
        """检测从众窗口"""
        valid_comments = df[df['time_diff'].notna()]
        
        if len(valid_comments) == 0:
            return {}
        
        # 按分钟统计评论数量
        minute_counts = defaultdict(int)
        for _, row in valid_comments.iterrows():
            minute = int(row['time_diff'])
            minute_counts[minute] += 1
        
        # 计算阈值
        total_comments = len(valid_comments)
        threshold = total_comments * threshold_percentage / 100
        
        # 找到密集时段
        dense_minutes = [minute for minute, count in minute_counts.items() if count >= threshold]
        
        # 合并连续的密集分钟
        conformity_windows = []
        if dense_minutes:
            dense_minutes.sort()
            start = dense_minutes[0]
            end = dense_minutes[0]
            
            for minute in dense_minutes[1:]:
                if minute == end + 1:
                    end = minute
                else:
                    conformity_windows.append((start, end))
                    start = minute
                    end = minute
            
            conformity_windows.append((start, end))
        
        return {
            'threshold_percentage': threshold_percentage,
            'threshold_count': threshold,
            'conformity_windows': conformity_windows,
            'dense_minutes': dense_minutes
        }
    
    def _find_dense_periods(self, df: pd.DataFrame) -> Dict:
        """查找密集时段"""
        valid_comments = df[df['time_diff'].notna()]
        
        if len(valid_comments) == 0:
            return {}
        
        # 按小时统计
        hour_counts = defaultdict(int)
        for _, row in valid_comments.iterrows():
            hour = int(row['time_diff'] / 60)
            hour_counts[hour] += 1
        
        # 找到最密集的时段
        if hour_counts:
            max_hour = max(hour_counts, key=hour_counts.get)
            max_count = hour_counts[max_hour]
            
            # 计算前10个最密集的时段
            top_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'peak_hour': max_hour,
                'peak_count': max_count,
                'top_hours': top_hours
            }
        
        return {}
    
    def _create_charts(self, df: pd.DataFrame, results: Dict, output_path: str):
        """创建时间分析可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('从众心理时间分析结果', fontsize=16, fontweight='bold')
        
        # 使用处理后的DataFrame
        if hasattr(self, 'df_processed'):
            df = self.df_processed
        
        valid_comments = df[df['time_diff'].notna()]
        
        if len(valid_comments) == 0:
            # 如果没有有效数据，显示提示
            for ax in axes.flat:
                ax.text(0.5, 0.5, '没有有效的子评论数据', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
        else:
            # 1. 时间差分布直方图
            ax1 = axes[0, 0]
            if 'time_diff' in valid_comments.columns:
                time_diffs = valid_comments['time_diff'].values
                ax1.hist(time_diffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_xlabel('时间差（分钟）')
                ax1.set_ylabel('频次')
                ax1.set_title('子评论时间差分布')
            else:
                ax1.text(0.5, 0.5, '无时间差数据', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.axis('off')
            ax1.grid(True, alpha=0.3)
            
            # 2. 窗口分布饼图
            ax2 = axes[0, 1]
            if 'window_distribution' in results and results['window_distribution']:
                window_data = results['window_distribution']
                labels = list(window_data['counts'].keys())
                sizes = list(window_data['counts'].values())
                if sum(sizes) > 0:
                    colors = ['lightcoral', 'lightblue', 'lightgreen', 'yellow', 'orange']
                    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                    ax2.set_title('从众时间窗口分布')
                else:
                    ax2.text(0.5, 0.5, '无窗口分布数据', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                    ax2.set_xlim(0, 1)
                    ax2.set_ylim(0, 1)
                    ax2.axis('off')
            
            # 3. 时间差箱线图
            ax3 = axes[1, 0]
            if 'time_diff' in valid_comments.columns:
                time_diffs = valid_comments['time_diff'].values
                ax3.boxplot(time_diffs, vert=True)
                ax3.set_ylabel('时间差（分钟）')
                ax3.set_title('时间差分布箱线图')
            else:
                ax3.text(0.5, 0.5, '无时间差数据', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
                ax3.axis('off')
            ax3.grid(True, alpha=0.3)
            
            # 4. 密集时段分析
            ax4 = axes[1, 1]
            if 'dense_periods' in results and results['dense_periods']:
                dense_data = results['dense_periods']
                if 'top_hours' in dense_data:
                    hours, counts = zip(*dense_data['top_hours'][:5])
                    ax4.bar(range(len(hours)), counts, color='lightcoral')
                    ax4.set_xlabel('小时')
                    ax4.set_ylabel('评论数量')
                    ax4.set_title('最密集时段（前5小时）')
                    ax4.set_xticks(range(len(hours)))
                    ax4.set_xticklabels([f'{h}小时' for h in hours])
                    ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    parser = create_parser("time", "从众心理时间分析工具")
    args = parser.parse_args()
    
    # 解析通用参数
    args = parse_common_args(parser, args)
    
    # 创建分析器
    analyzer = TimeAnalysisAnalyzer(args.video_id)
    
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
