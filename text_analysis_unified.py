#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分析统一入口
提供所有分析模块的统一接口，支持数据清洗、从众心理分析（时间、情感、相似度、点赞）
"""

import os
import sys
import argparse
from typing import Optional
from datetime import datetime
import numpy as np

def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'to_pydatetime'):  # pandas Timestamp
        return obj.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
    elif hasattr(obj, 'strftime'):  # datetime objects
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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
  
  2. 从众心理分析（使用清洗数据）
     python text_analysis_unified.py conformity --use-cleaned-data --video-id 123456
  
  3. 单独模块分析
     python text_analysis_unified.py time --use-cleaned-data --video-id 123456
     python text_analysis_unified.py sentiment --use-cleaned-data --video-id 123456
     python text_analysis_unified.py similarity --use-cleaned-data --video-id 123456
     python text_analysis_unified.py like --use-cleaned-data --video-id 123456

测试模式示例:
  python text_analysis_unified.py cleaning --test
  python text_analysis_unified.py conformity --use-cleaned-data --test
  python text_analysis_unified.py time --use-cleaned-data --test
  python text_analysis_unified.py sentiment --use-cleaned-data --test
  python text_analysis_unified.py similarity --use-cleaned-data --test
  python text_analysis_unified.py like --use-cleaned-data --test
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='module', help='选择分析模块')
    
    # 从众心理综合分析子命令
    conformity_parser = subparsers.add_parser('conformity', help='从众心理综合分析（时间+情感+相似度+点赞）')
    conformity_parser.add_argument('--use-cleaned-data', action='store_true', 
                                 help='使用清洗后的数据文件（推荐）')
    conformity_parser.add_argument('--video-id', type=str, help='视频ID，如果不指定则分析所有数据')
    conformity_parser.add_argument('--limit', type=int, help='限制分析数量')
    conformity_parser.add_argument('--cleaned-data-path', type=str, help='清洗数据文件路径')
    conformity_parser.add_argument('--test', action='store_true', help='测试模式，只分析少量数据')
    conformity_parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    conformity_parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    conformity_parser.add_argument('--no-viz', action='store_true', help='不创建可视化图表')
    
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
    if args.module == 'conformity':
        run_conformity_analysis(args)
    elif args.module == 'sentiment':
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

def run_conformity_analysis(args):
    """运行从众心理综合分析"""
    try:
        import json
        import pandas as pd
        from datetime import datetime
        from modules.conformity_time_analyzer import ConformityTimeAnalyzer
        from modules.sentiment_conformity_analyzer import SentimentConformityAnalyzer
        from modules.similarity_conformity_analyzer import SimilarityConformityAnalyzer
        from modules.like_conformity_analyzer import LikeConformityAnalyzer
        
        print("🚀 开始从众心理综合分析...")
        
        # 加载数据
        if args.use_cleaned_data:
            if args.cleaned_data_path:
                data_path = args.cleaned_data_path
            else:
                # 查找清洗后的数据文件
                import glob
                cleaned_files = glob.glob("modules/data/cleaned/*_cleaned.json")
                if not cleaned_files:
                    print("❌ 未找到清洗后的数据文件")
                    return
                data_path = cleaned_files[0]
                print(f"📁 使用清洗数据: {data_path}")
        else:
            print("❌ 从众心理分析需要清洗后的数据")
            return
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if args.limit:
            data = data[:args.limit]
            print(f"📊 限制分析数量: {args.limit}")
        
        # 初始化分析器
        time_analyzer = ConformityTimeAnalyzer()
        sentiment_analyzer = SentimentConformityAnalyzer()
        similarity_analyzer = SimilarityConformityAnalyzer()
        like_analyzer = LikeConformityAnalyzer()
        
        # 运行各项分析
        results = {}
        
        print("⏰ 运行时间从众心理分析...")
        time_result = time_analyzer.analyze(data)
        results['time_conformity'] = time_result
        
        print("😊 运行情感从众心理分析...")
        sentiment_result = sentiment_analyzer.analyze(data)
        results['sentiment_conformity'] = sentiment_result
        
        print("📝 运行相似度从众心理分析...")
        similarity_result = similarity_analyzer.analyze(data)
        results['similarity_conformity'] = similarity_result
        
        print("👍 运行点赞从众心理分析...")
        like_result = like_analyzer.analyze(data)
        results['like_conformity'] = like_result
        
        # 生成综合结果
        print("📊 生成综合分析结果...")
        comprehensive_result = generate_comprehensive_result(results, data)
        
        # 保存结果
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"modules/data/comprehensive_conformity_analysis_{timestamp}.json"
            
            # 转换NumPy类型为Python原生类型
            converted_result = convert_numpy_types(comprehensive_result)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_result, f, ensure_ascii=False, indent=2)
            
            print(f"💾 综合分析结果已保存: {output_file}")
        
        # 生成报告
        if not args.no_report:
            generate_comprehensive_report(comprehensive_result)
        
        print("✅ 从众心理综合分析完成！")
        
    except Exception as e:
        print(f"❌ 从众心理综合分析执行失败: {e}")
        import traceback
        traceback.print_exc()

def run_sentiment_analysis(args):
    """运行情感从众心理分析"""
    try:
        from modules.sentiment_conformity_analyzer import SentimentConformityAnalyzer
        import json
        
        print("😊 开始情感从众心理分析...")
        
        # 加载数据
        if args.use_cleaned_data:
            if args.cleaned_data_path:
                data_path = args.cleaned_data_path
            else:
                import glob
                cleaned_files = glob.glob("modules/data/cleaned/*_cleaned.json")
                if not cleaned_files:
                    print("❌ 未找到清洗后的数据文件")
                    return
                data_path = cleaned_files[0]
        else:
            print("❌ 情感从众心理分析需要清洗后的数据")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if args.limit:
            data = data[:args.limit]
        
        # 运行分析
        analyzer = SentimentConformityAnalyzer()
        result = analyzer.analyze(data)
        
        # 保存结果
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"modules/data/sentiment_conformity_analysis_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"💾 情感从众心理分析结果已保存: {output_file}")
        
        print("✅ 情感从众心理分析完成！")
        
    except Exception as e:
        print(f"❌ 情感从众心理分析执行失败: {e}")
        import traceback
        traceback.print_exc()

def run_time_analysis(args):
    """运行时间从众心理分析"""
    try:
        from modules.conformity_time_analyzer import ConformityTimeAnalyzer
        import json
        
        print("⏰ 开始时间从众心理分析...")
        
        # 加载数据
        if args.use_cleaned_data:
            if args.cleaned_data_path:
                data_path = args.cleaned_data_path
            else:
                import glob
                cleaned_files = glob.glob("modules/data/cleaned/*_cleaned.json")
                if not cleaned_files:
                    print("❌ 未找到清洗后的数据文件")
                    return
                data_path = cleaned_files[0]
        else:
            print("❌ 时间从众心理分析需要清洗后的数据")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if args.limit:
            data = data[:args.limit]
        
        # 运行分析
        analyzer = ConformityTimeAnalyzer()
        result = analyzer.analyze(data)
        
        # 保存结果
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"modules/data/time_conformity_analysis_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"💾 时间从众心理分析结果已保存: {output_file}")
        
        print("✅ 时间从众心理分析完成！")
        
    except Exception as e:
        print(f"❌ 时间从众心理分析执行失败: {e}")
        import traceback
        traceback.print_exc()

def run_like_analysis(args):
    """运行点赞从众心理分析"""
    try:
        from modules.like_conformity_analyzer import LikeConformityAnalyzer
        import json
        
        print("👍 开始点赞从众心理分析...")
        
        # 加载数据
        if args.use_cleaned_data:
            if args.cleaned_data_path:
                data_path = args.cleaned_data_path
            else:
                import glob
                cleaned_files = glob.glob("modules/data/cleaned/*_cleaned.json")
                if not cleaned_files:
                    print("❌ 未找到清洗后的数据文件")
                    return
                data_path = cleaned_files[0]
        else:
            print("❌ 点赞从众心理分析需要清洗后的数据")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if args.limit:
            data = data[:args.limit]
        
        # 运行分析
        analyzer = LikeConformityAnalyzer()
        result = analyzer.analyze(data)
        
        # 保存结果
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"modules/data/like_conformity_analysis_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"💾 点赞从众心理分析结果已保存: {output_file}")
        
        print("✅ 点赞从众心理分析完成！")
        
    except Exception as e:
        print(f"❌ 点赞从众心理分析执行失败: {e}")
        import traceback
        traceback.print_exc()

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
    """运行相似度从众心理分析"""
    try:
        from modules.similarity_conformity_analyzer import SimilarityConformityAnalyzer
        import json
        
        print("📝 开始相似度从众心理分析...")
        
        # 加载数据
        if args.use_cleaned_data:
            if args.cleaned_data_path:
                data_path = args.cleaned_data_path
            else:
                import glob
                cleaned_files = glob.glob("modules/data/cleaned/*_cleaned.json")
                if not cleaned_files:
                    print("❌ 未找到清洗后的数据文件")
                    return
                data_path = cleaned_files[0]
        else:
            print("❌ 相似度从众心理分析需要清洗后的数据")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if args.limit:
            data = data[:args.limit]
        
        # 运行分析
        analyzer = SimilarityConformityAnalyzer()
        result = analyzer.analyze(data)
        
        # 保存结果
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"modules/data/similarity_conformity_analysis_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"💾 相似度从众心理分析结果已保存: {output_file}")
        
        print("✅ 相似度从众心理分析完成！")
        
    except Exception as e:
        print(f"❌ 相似度从众心理分析执行失败: {e}")
        import traceback
        traceback.print_exc()

def generate_comprehensive_result(results, data):
    """生成综合分析结果"""
    from datetime import datetime
    import numpy as np
    
    # 提取各项分析结果
    time_result = results.get('time_conformity', {})
    sentiment_result = results.get('sentiment_conformity', {})
    similarity_result = results.get('similarity_conformity', {})
    like_result = results.get('like_conformity', {})
    
    # 提取父评论信息
    parent_comment_id = None
    parent_content = None
    if data:
        parent_comment = next((item for item in data if item.get('parent_comment_id') is None), None)
        if parent_comment:
            parent_comment_id = parent_comment.get('comment_id')
            parent_content = parent_comment.get('content', '')
    
    # 计算综合从众心理分数
    conformity_scores = []
    score_details = {}
    
    # 时间从众心理分数
    if 'parent_environment_time_analysis' in time_result:
        time_score = time_result['parent_environment_time_analysis'].get('parent_conformity_score', 0)
        conformity_scores.append(time_score)
        score_details['time_conformity_score'] = time_score
    
    # 情感从众心理分数
    if 'parent_environment_sentiment_analysis' in sentiment_result:
        sentiment_score = sentiment_result['parent_environment_sentiment_analysis'].get('parent_sentiment_conformity_score', 0)
        conformity_scores.append(sentiment_score)
        score_details['sentiment_conformity_score'] = sentiment_score
    
    # 相似度从众心理分数
    if 'parent_environment_similarity_analysis' in similarity_result:
        similarity_score = similarity_result['parent_environment_similarity_analysis'].get('parent_similarity_conformity_score', 0)
        conformity_scores.append(similarity_score)
        score_details['similarity_conformity_score'] = similarity_score
    
    # 点赞从众心理分数
    if 'parent_environment_like_analysis' in like_result:
        like_score = like_result['parent_environment_like_analysis'].get('parent_like_conformity_score', 0)
        conformity_scores.append(like_score)
        score_details['like_conformity_score'] = like_score
    
    # 计算综合分数
    overall_conformity_score = float(np.mean(conformity_scores)) if conformity_scores else 0.0
    
    # 确保所有分数都是Python原生类型
    score_details = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                    for k, v in score_details.items()}
    
    # 生成综合分析结果
    comprehensive_result = {
        "analysis_metadata": {
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analyzer_version": "v1.0.0",
            "analysis_type": "comprehensive_conformity_analysis",
            "total_comments": int(len(data)),
            "parent_comment_id": str(parent_comment_id)
        },
        "parent_comment_info": {
            "comment_id": str(parent_comment_id),
            "content": str(parent_content)
        },
        "comprehensive_conformity_analysis": {
            "overall_conformity_score": float(overall_conformity_score),
            "conformity_level": get_conformity_level(overall_conformity_score),
            "score_breakdown": score_details,
            "analysis_summary": {
                "time_conformity": {
                    "score": score_details.get('time_conformity_score', 0),
                    "description": "时间维度从众心理分析"
                },
                "sentiment_conformity": {
                    "score": score_details.get('sentiment_conformity_score', 0),
                    "description": "情感维度从众心理分析"
                },
                "similarity_conformity": {
                    "score": score_details.get('similarity_conformity_score', 0),
                    "description": "文本相似度从众心理分析"
                },
                "like_conformity": {
                    "score": score_details.get('like_conformity_score', 0),
                    "description": "点赞行为从众心理分析"
                }
            }
        },
        "detailed_results": {
            "time_conformity_analysis": time_result,
            "sentiment_conformity_analysis": sentiment_result,
            "similarity_conformity_analysis": similarity_result,
            "like_conformity_analysis": like_result
        }
    }
    
    return comprehensive_result

def get_conformity_level(score):
    """根据分数确定从众心理等级"""
    if score >= 0.8:
        return "极高从众心理"
    elif score >= 0.6:
        return "高从众心理"
    elif score >= 0.4:
        return "中等从众心理"
    elif score >= 0.2:
        return "低从众心理"
    else:
        return "极低从众心理"

def generate_comprehensive_report(result):
    """生成综合分析报告"""
    print("\n" + "="*60)
    print("📊 从众心理综合分析报告")
    print("="*60)
    
    # 基本信息
    metadata = result.get('analysis_metadata', {})
    parent_info = result.get('parent_comment_info', {})
    comprehensive = result.get('comprehensive_conformity_analysis', {})
    
    print(f"📅 分析时间: {metadata.get('analysis_time', 'N/A')}")
    print(f"📝 父评论ID: {parent_info.get('comment_id', 'N/A')}")
    print(f"💬 父评论内容: {parent_info.get('content', 'N/A')[:100]}...")
    print(f"📊 总评论数: {metadata.get('total_comments', 0)}")
    
    print("\n" + "-"*40)
    print("🎯 综合从众心理分析结果")
    print("-"*40)
    
    overall_score = comprehensive.get('overall_conformity_score', 0)
    conformity_level = comprehensive.get('conformity_level', 'N/A')
    
    print(f"🏆 综合从众心理分数: {overall_score:.4f}")
    print(f"📈 从众心理等级: {conformity_level}")
    
    # 各维度分数
    score_breakdown = comprehensive.get('score_breakdown', {})
    print("\n📊 各维度从众心理分数:")
    print(f"  ⏰ 时间从众心理: {score_breakdown.get('time_conformity_score', 0):.4f}")
    print(f"  😊 情感从众心理: {score_breakdown.get('sentiment_conformity_score', 0):.4f}")
    print(f"  📝 相似度从众心理: {score_breakdown.get('similarity_conformity_score', 0):.4f}")
    print(f"  👍 点赞从众心理: {score_breakdown.get('like_conformity_score', 0):.4f}")
    
    # 分析建议
    print("\n💡 分析建议:")
    if overall_score >= 0.7:
        print("  🔥 该父评论环境表现出强烈的从众心理特征")
        print("  📈 建议关注内容传播模式和用户行为特征")
    elif overall_score >= 0.4:
        print("  ⚖️ 该父评论环境表现出中等程度的从众心理")
        print("  🔍 建议进一步分析具体维度的从众心理表现")
    else:
        print("  🎯 该父评论环境从众心理特征不明显")
        print("  📊 建议关注其他社交行为特征")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
