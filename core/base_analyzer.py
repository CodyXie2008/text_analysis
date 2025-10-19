#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分析模块基类
提供统一的分析接口和通用功能
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
from text_analysis.core.db_config import get_db_conn
from text_analysis.core.data_paths import AnalysisPathManager, PROJECT_ROOT, resolve_latest_cleaned_data

# 使用PROJECT_ROOT作为项目根目录
project_root = PROJECT_ROOT

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseAnalyzer:
    """分析模块基类"""
    
    def __init__(self, module_name: str, video_id: str = None):
        """
        初始化分析器
        
        Args:
            module_name: 模块名称
            video_id: 视频ID
        """
        self.module_name = module_name
        self.video_id = video_id
        self.path_manager = AnalysisPathManager(module_name, video_id)
        self.conn = None
        
    def connect_database(self):
        """连接数据库"""
        try:
            self.conn = get_db_conn()
            print("✅ 数据库连接成功")
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False
    
    def close_database(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("✅ 数据库连接已关闭")
    
    def load_data(self, limit: Optional[int] = None, use_cleaned_data: bool = False, cleaned_data_path: str = None) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            limit: 限制数据量
            use_cleaned_data: 是否使用清洗后的数据
            cleaned_data_path: 清洗数据文件路径
            
        Returns:
            DataFrame: 加载的数据
        """
        if use_cleaned_data:
            return self._load_from_cleaned_data(cleaned_data_path)
        else:
            return self._load_from_database(limit)
    
    def _load_from_database(self, limit: Optional[int] = None) -> pd.DataFrame:
        """从数据库加载数据"""
        raise NotImplementedError("子类必须实现此方法")
    
    def _load_from_cleaned_data(self, cleaned_data_path: str = None) -> pd.DataFrame:
        """从清洗数据文件加载数据"""
        try:
            if cleaned_data_path is None:
                # 优先解析最新的清洗数据（兼容新老命名）
                auto_path = resolve_latest_cleaned_data(self.video_id)
                cleaned_data_path = auto_path or os.path.join(PROJECT_ROOT, 'data', 'processed', 'douyin_comments_processed.json')
            
            if not os.path.exists(cleaned_data_path):
                print(f"❌ 清洗数据文件不存在: {cleaned_data_path}")
                return pd.DataFrame()
            
            with open(cleaned_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            # 如指定 video_id，则按 aweme_id 过滤
            if self.video_id and 'aweme_id' in df.columns:
                df = df[df['aweme_id'] == self.video_id]
            print(f"✅ 成功加载清洗数据: {len(df)} 条记录")
            return df
            
        except Exception as e:
            print(f"❌ 加载清洗数据失败: {e}")
            return pd.DataFrame()
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        执行分析
        
        Args:
            df: 输入数据
            
        Returns:
            Dict: 分析结果
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def save_results(self, df: pd.DataFrame, results: Dict):
        """保存分析结果"""
        try:
            # 保存CSV结果
            csv_path = self.path_manager.get_results_paths()['csv']
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 结果已保存到: {csv_path}")
            
            # 保存JSON结果
            json_path = self.path_manager.get_results_paths()['json']
            results_data = {
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'module_name': self.module_name,
                'video_id': self.video_id,
                'total_records': len(df),
                'results': results
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存到: {json_path}")
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def generate_report(self, df: pd.DataFrame, results: Dict):
        """生成分析报告"""
        try:
            report_path = self.path_manager.get_report_path()
            
            report = {
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'module_name': self.module_name,
                'video_id': self.video_id,
                'summary': {
                    'total_records': len(df),
                    'analysis_duration': results.get('duration', 0)
                },
                'detailed_results': results
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 分析报告已保存到: {report_path}")
            
            # 打印报告摘要
            self._print_report_summary(report)
            
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
    
    def create_visualizations(self, df: pd.DataFrame, results: Dict):
        """创建可视化图表"""
        try:
            viz_path = self.path_manager.get_visualization_path()
            self._create_charts(df, results, viz_path)
            print(f"✅ 可视化图表已保存到: {viz_path}")
        except Exception as e:
            print(f"❌ 创建可视化失败: {e}")
    
    def _create_charts(self, df: pd.DataFrame, results: Dict, output_path: str):
        """创建图表（子类可重写）"""
        # 默认实现：创建简单的统计图表
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'{self.module_name.title()} 分析结果', fontsize=16, fontweight='bold')
        
        # 这里可以添加默认的图表内容
        axes.text(0.5, 0.5, f'分析完成\n总记录数: {len(df)}', 
                 ha='center', va='center', transform=axes.transAxes, fontsize=14)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_report_summary(self, report: Dict):
        """打印报告摘要"""
        print("\n=== 分析报告摘要 ===")
        print(f"分析时间: {report['analysis_time']}")
        print(f"模块名称: {report['module_name']}")
        if report['video_id']:
            print(f"视频ID: {report['video_id']}")
        print(f"总记录数: {report['summary']['total_records']:,}")
    
    def run_analysis(self, limit: Optional[int] = None, use_cleaned_data: bool = False, 
                    cleaned_data_path: str = None, save_results: bool = True, 
                    generate_report: bool = True, create_visualizations: bool = True, 
                    strict_mode: bool = False):
        """
        运行完整的分析流程
        
        Args:
            limit: 限制数据量
            use_cleaned_data: 是否使用清洗后的数据
            cleaned_data_path: 清洗数据文件路径
            save_results: 是否保存结果
            generate_report: 是否生成报告
            create_visualizations: 是否创建可视化
            strict_mode: 是否使用严格模式（主要用于数据清洗模块）
        """
        print(f"=== {self.module_name.title()} 分析开始 ===")
        
        # 连接数据库
        if not use_cleaned_data and not self.connect_database():
            return
        
        try:
            # 加载数据
            df = self.load_data(limit, use_cleaned_data, cleaned_data_path)
            if df.empty:
                print("❌ 没有找到数据")
                return
            
            # 执行分析
            start_time = datetime.now()
            # 传递strict_mode参数给analyze方法
            results = self.analyze(df, strict_mode=strict_mode)
            end_time = datetime.now()
            results['duration'] = (end_time - start_time).total_seconds()
            
            # 清洗模块额外打印摘要
            if self.module_name == 'cleaning':
                try:
                    orig = results.get('original_stats', {})
                    fin = results.get('final_stats', {})
                    # 美化终端摘要输出
                    print("\n┌────────── 清洗摘要 ──────────")
                    print(f"│ 原始总评论: {orig.get('total_comments', 0):,}    父:{orig.get('parent_comments','?')}  子:{orig.get('child_comments','?')}")
                    print(f"│ 清洗后总评论: {fin.get('total_comments', 0):,}  父:{fin.get('parent_comments','?')}  子:{fin.get('child_comments','?')}")
                    print("└────────────────────────────")
                except Exception:
                    pass

            # 保存结果
            if save_results:
                self.save_results(df, results)
            
            # 生成报告
            if generate_report:
                self.generate_report(df, results)
            
            # 创建可视化
            if create_visualizations:
                self.create_visualizations(df, results)
            
            print(f"\n✅ {self.module_name.title()} 分析完成!")
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 关闭数据库连接
            if not use_cleaned_data:
                self.close_database()

def create_parser(module_name: str, description: str) -> argparse.ArgumentParser:
    """创建统一的命令行参数解析器"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--video-id', type=str, help='视频ID，如果不指定则分析所有数据')
    parser.add_argument('--limit', type=int, help='限制分析数量')
    parser.add_argument('--use-cleaned-data', action='store_true', help='使用清洗后的数据文件')
    parser.add_argument('--cleaned-data-path', type=str, help='清洗数据文件路径')
    parser.add_argument('--test', action='store_true', help='测试模式，只分析少量数据')
    parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    parser.add_argument('--no-viz', action='store_true', help='不创建可视化图表')
    
    return parser

def parse_common_args(parser: argparse.ArgumentParser, args):
    """解析通用参数"""
    # 测试模式设置
    if args.test:
        if not args.limit:
            args.limit = 10
        print("🧪 测试模式：只分析少量数据")
    
    # 显示配置信息
    print(f"=== {parser.description} ===")
    if args.video_id:
        print(f"视频ID: {args.video_id}")
    if args.limit:
        print(f"限制数量: {args.limit}")
    if args.use_cleaned_data:
        print("使用清洗数据")
    print("=" * 30)
    
    return args
