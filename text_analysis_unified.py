#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分析统一入口
提供所有分析模块的统一接口，支持情感分析、时间分析、点赞分析、数据清洗
"""

import os
import sys
import argparse
from typing import Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="文本分析统一工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
推荐分析流程:
  1. 首先运行数据清洗
     python text_analysis_unified.py cleaning --video-id 123456
  
  2. 时间分析（使用清洗数据）
     python text_analysis_unified.py time --use-cleaned-data --video-id 123456
  
  3. 点赞分析（使用清洗数据）
     python text_analysis_unified.py like --use-cleaned-data --video-id 123456
  
  4. 情感分析（使用清洗数据）
     python text_analysis_unified.py sentiment --use-cleaned-data --type local --video-id 123456
  
  5. 相似度分析（使用清洗数据）
     python text_analysis_unified.py similarity --use-cleaned-data --video-id 123456

测试模式示例:
  python text_analysis_unified.py cleaning --test
  python text_analysis_unified.py time --use-cleaned-data --test
  python text_analysis_unified.py like --use-cleaned-data --test
  python text_analysis_unified.py sentiment --use-cleaned-data --type local --test
  python text_analysis_unified.py similarity --use-cleaned-data --test
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='module', help='选择分析模块')
    
    # 情感分析子命令
    sentiment_parser = subparsers.add_parser('sentiment', help='情感分析')
    sentiment_parser.add_argument('--use-cleaned-data', action='store_true', 
                                 help='使用清洗后的数据文件（推荐）')
    sentiment_parser.add_argument('--type', choices=['local', 'aliyun'], 
                                 default='aliyun', help='分析器类型：local(本地词典) 或 aliyun(阿里云API)，默认aliyun')
    sentiment_parser.add_argument('--video-id', type=str, help='视频ID，如果不指定则分析所有评论')
    sentiment_parser.add_argument('--limit', type=int, help='限制分析数量')
    sentiment_parser.add_argument('--cleaned-data-path', type=str, help='清洗数据文件路径')
    sentiment_parser.add_argument('--test', action='store_true', help='测试模式，只分析少量数据')
    sentiment_parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    sentiment_parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    sentiment_parser.add_argument('--no-viz', action='store_true', help='不创建可视化图表')
    # 情感分析并发参数（透传到模块）
    sentiment_parser.add_argument('--sa-concurrency', type=int, default=8, help='情感API并发数，默认8')
    sentiment_parser.add_argument('--sa-batch-size', type=int, default=200, help='情感API批大小，默认200')
    sentiment_parser.add_argument('--sa-throttle-ms', type=int, default=0, help='情感API节流毫秒，默认0=不限制')
    
    # 时间分析子命令
    time_parser = subparsers.add_parser('time', help='从众心理时间分析')
    time_parser.add_argument('--use-cleaned-data', action='store_true', 
                            help='使用清洗后的数据文件（推荐）')
    time_parser.add_argument('--video-id', type=str, help='视频ID，如果不指定则分析所有数据')
    time_parser.add_argument('--limit', type=int, help='限制分析数量')
    time_parser.add_argument('--cleaned-data-path', type=str, help='清洗数据文件路径')
    time_parser.add_argument('--test', action='store_true', help='测试模式，只分析少量数据')
    time_parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    time_parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    time_parser.add_argument('--no-viz', action='store_true', help='不创建可视化图表')
    
    # 点赞分析子命令
    like_parser = subparsers.add_parser('like', help='从众心理点赞分析')
    like_parser.add_argument('--use-cleaned-data', action='store_true', 
                            help='使用清洗后的数据文件（推荐）')
    like_parser.add_argument('--video-id', type=str, help='视频ID，如果不指定则分析所有数据')
    like_parser.add_argument('--limit', type=int, help='限制分析数量')
    like_parser.add_argument('--cleaned-data-path', type=str, help='清洗数据文件路径')
    like_parser.add_argument('--test', action='store_true', help='测试模式，只分析少量数据')
    like_parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    like_parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    like_parser.add_argument('--no-viz', action='store_true', help='不创建可视化图表')
    
    # 数据清洗子命令
    cleaning_parser = subparsers.add_parser('cleaning', help='数据清洗')
    cleaning_parser.add_argument('--video-id', type=str, help='视频ID，如果不指定则清洗所有数据')
    cleaning_parser.add_argument('--limit', type=int, help='限制清洗数量')
    cleaning_parser.add_argument('--test', action='store_true', help='测试模式，只清洗少量数据')
    cleaning_parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    cleaning_parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    cleaning_parser.add_argument('--no-viz', action='store_true', help='不创建可视化图表')
    
    # 相似度分析子命令
    similarity_parser = subparsers.add_parser('similarity', help='文本相似度分析')
    similarity_parser.add_argument('--use-cleaned-data', action='store_true', 
                                  help='使用清洗后的数据文件（推荐）')
    similarity_parser.add_argument('--video-id', type=str, help='视频ID，如果不指定则分析所有数据')
    similarity_parser.add_argument('--limit', type=int, help='限制分析数量')
    similarity_parser.add_argument('--cleaned-data-path', type=str, help='清洗数据文件路径')
    similarity_parser.add_argument('--similarity-threshold', type=float, default=0.7, 
                                  help='相似度阈值（默认0.7）')
    similarity_parser.add_argument('--time-diff-threshold', type=int, default=3600, 
                                  help='时间差阈值(秒)（默认3600）')
    similarity_parser.add_argument('--test', action='store_true', help='测试模式，只分析少量数据')
    similarity_parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    similarity_parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    similarity_parser.add_argument('--no-viz', action='store_true', help='不创建可视化图表')
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.module:
        parser.print_help()
        return
    
    # 测试模式设置
    if args.test and not args.limit:
        args.limit = 10
        print("🧪 测试模式：只分析少量数据")
    
    # 显示配置信息
    print(f"=== 文本分析统一工具 - {args.module.upper()} 模块 ===")
    if hasattr(args, 'video_id') and args.video_id:
        print(f"视频ID: {args.video_id}")
    if hasattr(args, 'type') and args.type:
        print(f"分析器类型: {args.type}")
    if args.limit:
        print(f"限制数量: {args.limit}")
    if hasattr(args, 'use_cleaned_data') and args.use_cleaned_data:
        print("使用清洗数据")
    print("=" * 50)
    
    # 批量模式：未指定 --video-id 时，按视频ID批处理
    if not getattr(args, 'video_id', None):
        from text_analysis.utils import enumerate_aweme_ids
        id_list = enumerate_aweme_ids(use_cleaned_data=getattr(args, 'use_cleaned_data', False), cleaned_data_path=getattr(args, 'cleaned_data_path', None))
        if id_list:
            print(f"🔁 未指定视频ID，按 {len(id_list)} 个视频ID 批量处理...")
            for vid in id_list:
                args.video_id = str(vid)
                dispatch_module(args)
            return
        else:
            print("⚠️ 未发现可处理的视频ID，回退到全量执行")
            args.video_id = None
    dispatch_module(args)
def dispatch_module(args):
    """按模块分发执行"""
    if args.module == 'sentiment':
        run_sentiment_analysis(args)
    elif args.module == 'time':
        run_time_analysis(args)
    elif args.module == 'like':
        run_like_analysis(args)
    elif args.module == 'cleaning':
        run_cleaning_analysis(args)
    elif args.module == 'similarity':
        run_similarity_analysis(args)
    else:
        print(f"❌ 未知的分析模块: {args.module}")

def run_sentiment_analysis(args):
    """运行情感分析"""
    try:
        from modules.sentiment_analyzer_optimized import main as sentiment_main
        import sys
        
        # 构建命令行参数
        sys.argv = ['sentiment_analyzer_optimized.py']
        if args.type:
            sys.argv.extend(['--type', args.type])
        if args.video_id:
            sys.argv.extend(['--video-id', args.video_id])
        if args.limit:
            sys.argv.extend(['--limit', str(args.limit)])
        if args.use_cleaned_data:
            sys.argv.append('--use-cleaned-data')
        if args.cleaned_data_path:
            sys.argv.extend(['--cleaned-data-path', args.cleaned_data_path])
        if args.test:
            sys.argv.append('--test')
        # sentiment 模块当前不支持 --no-report/--no-viz 原生参数，跳过透传
        # 不透传 --no-save，sentiment 模块当前未定义该参数
        # 并发性能参数透传
        if hasattr(args, 'sa_concurrency'):
            sys.argv.extend(['--sa-concurrency', str(args.sa_concurrency)])
        if hasattr(args, 'sa_batch_size'):
            sys.argv.extend(['--sa-batch-size', str(args.sa_batch_size)])
        if hasattr(args, 'sa_throttle_ms'):
            sys.argv.extend(['--sa-throttle-ms', str(args.sa_throttle_ms)])
        
        sentiment_main()
        
    except ImportError as e:
        print(f"❌ 导入情感分析模块失败: {e}")
    except Exception as e:
        print(f"[ERR] 情感分析执行失败: {e}")

def run_time_analysis(args):
    """运行时间分析"""
    try:
        from modules.time_analysis_optimized import main as time_main
        import sys
        
        # 构建命令行参数
        sys.argv = ['time_analysis_optimized.py']
        if args.video_id:
            sys.argv.extend(['--video-id', args.video_id])
        if args.limit:
            sys.argv.extend(['--limit', str(args.limit)])
        if args.use_cleaned_data:
            sys.argv.append('--use-cleaned-data')
        if args.cleaned_data_path:
            sys.argv.extend(['--cleaned-data-path', args.cleaned_data_path])
        if args.test:
            sys.argv.append('--test')
        if args.no_save:
            sys.argv.append('--no-save')
        if args.no_report:
            sys.argv.append('--no-report')
        if args.no_viz:
            sys.argv.append('--no-viz')
        
        time_main()
        
    except ImportError as e:
        print(f"❌ 导入时间分析模块失败: {e}")
    except Exception as e:
        print(f"❌ 时间分析执行失败: {e}")

def run_like_analysis(args):
    """运行点赞分析"""
    try:
        from modules.like_analysis_optimized import main as like_main
        import sys
        
        # 构建命令行参数
        sys.argv = ['like_analysis_optimized.py']
        if args.video_id:
            sys.argv.extend(['--video-id', args.video_id])
        if args.limit:
            sys.argv.extend(['--limit', str(args.limit)])
        if args.use_cleaned_data:
            sys.argv.append('--use-cleaned-data')
        if args.cleaned_data_path:
            sys.argv.extend(['--cleaned-data-path', args.cleaned_data_path])
        if args.test:
            sys.argv.append('--test')
        if args.no_save:
            sys.argv.append('--no-save')
        if args.no_report:
            sys.argv.append('--no-report')
        if args.no_viz:
            sys.argv.append('--no-viz')
        
        like_main()
        
    except ImportError as e:
        print(f"❌ 导入点赞分析模块失败: {e}")
    except Exception as e:
        print(f"❌ 点赞分析执行失败: {e}")

def run_cleaning_analysis(args):
    """运行数据清洗"""
    try:
        from modules.data_cleaning_optimized import main as cleaning_main
        import sys
        
        # 构建命令行参数
        sys.argv = ['data_cleaning_optimized.py']
        if args.video_id:
            sys.argv.extend(['--video-id', args.video_id])
        if args.limit:
            sys.argv.extend(['--limit', str(args.limit)])
        if args.test:
            sys.argv.append('--test')
        if args.no_save:
            sys.argv.append('--no-save')
        if args.no_report:
            sys.argv.append('--no-report')
        if args.no_viz:
            sys.argv.append('--no-viz')
        
        cleaning_main()
        
    except ImportError as e:
        print(f"❌ 导入数据清洗模块失败: {e}")
    except Exception as e:
        print(f"❌ 数据清洗执行失败: {e}")

def run_similarity_analysis(args):
    """运行相似度分析"""
    try:
        from modules.similarity_analysis_optimized import main as similarity_main
        import sys
        
        # 构建命令行参数
        sys.argv = ['similarity_analysis_optimized.py']
        if args.video_id:
            sys.argv.extend(['--video-id', args.video_id])
        if args.limit:
            sys.argv.extend(['--limit', str(args.limit)])
        if args.use_cleaned_data:
            sys.argv.append('--use-cleaned-data')
        if args.cleaned_data_path:
            sys.argv.extend(['--cleaned-data-path', args.cleaned_data_path])
        if args.similarity_threshold:
            sys.argv.extend(['--similarity-threshold', str(args.similarity_threshold)])
        if args.time_diff_threshold:
            sys.argv.extend(['--time-diff-threshold', str(args.time_diff_threshold)])
        if args.test:
            sys.argv.append('--test')
        if args.no_save:
            sys.argv.append('--no-save')
        if args.no_report:
            sys.argv.append('--no-report')
        if args.no_viz:
            sys.argv.append('--no-viz')
        
        similarity_main()
        
    except ImportError as e:
        print(f"❌ 导入相似度分析模块失败: {e}")
    except Exception as e:
        print(f"❌ 相似度分析执行失败: {e}")

if __name__ == "__main__":
    main()
