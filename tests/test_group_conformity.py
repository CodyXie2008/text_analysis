#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试群体从众心理指数分析模块
"""

import os
import sys
import json
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from modules.group_conformity_analyzer import GroupConformityAnalyzer
from core.data_paths import DATA_DIRS

def test_group_conformity_analyzer():
    """
    测试群体从众心理指数分析器的基本功能
    """
    print("=== 测试群体从众心理指数分析器 ===")
    
    try:
        # 创建分析器实例
        analyzer = GroupConformityAnalyzer()
        print("✅ 分析器初始化成功")
        
        # 测试加载预处理数据
        print("\n测试加载预处理数据...")
        processed_dir = DATA_DIRS['processed']
        print(f"预处理数据目录: {processed_dir}")
        
        # 加载少量数据进行测试
        data = analyzer.load_processed_data()
        if len(data) == 0:
            print("❌ 未加载到预处理数据")
            return False
        
        print(f"✅ 成功加载 {len(data)} 条评论集")
        
        # 显示前2条数据的结构
        print("\n前2条评论集的基本结构:")
        for i, thread in enumerate(data[:2]):
            comment_id = thread.get('comment_id', 'N/A')
            content = thread.get('content', '').strip()[:50] + '...' if len(thread.get('content', '')) > 50 else thread.get('content', '')
            sub_count = len(thread.get('children', []))
            like_count = thread.get('like_count', 0)
            print(f"\n评论集 {i+1}:")
            print(f"  comment_id: {comment_id}")
            print(f"  content: {content}")
            print(f"  子评论数: {sub_count}")
            print(f"  点赞数: {like_count}")
        
        # 测试单个评论集的从众心理指数计算
        print("\n测试单个评论集的从众心理指数计算...")
        sample_thread = data[0]
        start_time = time.time()
        result = analyzer.calculate_group_conformity(sample_thread)
        end_time = time.time()
        
        print(f"✅ 单个评论集分析完成，耗时: {end_time - start_time:.2f}秒")
        print("分析结果:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 测试小规模批量分析（限制为2个评论集）
        print("\n测试小规模批量分析...")
        small_data = data[:2]
        start_time = time.time()
        results = analyzer.analyze(small_data)
        end_time = time.time()
        
        print(f"✅ 批量分析完成，处理了 {len(results)} 个评论集，耗时: {end_time - start_time:.2f}秒")
        print(f"平均从众心理指数: {analyzer.stats['average_group_conformity']:.4f}")
        
        # 测试保存结果功能
        print("\n测试保存结果功能...")
        analyzer.save_results(results)
        print("✅ 结果保存成功")
        
        print("\n=== 测试完成 ===")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_herfindahl_index():
    """
    测试Herfindahl指数计算功能
    """
    print("\n=== 测试Herfindahl指数计算 ===")
    
    try:
        analyzer = GroupConformityAnalyzer()
        
        # 测试用例1: 示例数据（从建模文档中）
        likes1 = [100, 50, 30, 20]  # 父评论100赞，子评论50,30,20赞
        expected1 = 0.345  # 0.5² + 0.25² + 0.15² + 0.10² = 0.345
        result1 = analyzer.calculate_herfindahl_index(likes1)
        print(f"测试用例1 (建模文档示例):")
        print(f"  输入: {likes1}")
        print(f"  预期: {expected1}")
        print(f"  实际: {result1:.6f}")
        print(f"  结果: {'✓ 正确' if abs(result1 - expected1) < 0.001 else '✗ 错误'}")
        
        # 测试用例2: 极端集中（所有赞都在一条评论）
        likes2 = [100, 0, 0, 0]
        expected2 = 1.0  # 1.0² + 0 + 0 + 0 = 1.0
        result2 = analyzer.calculate_herfindahl_index(likes2)
        print(f"\n测试用例2 (极端集中):")
        print(f"  输入: {likes2}")
        print(f"  预期: {expected2}")
        print(f"  实际: {result2:.6f}")
        print(f"  结果: {'✓ 正确' if abs(result2 - expected2) < 0.001 else '✗ 错误'}")
        
        # 测试用例3: 完全分散（所有评论点赞相同）
        likes3 = [25, 25, 25, 25]
        expected3 = 0.25  # 4*(0.25²) = 0.25
        result3 = analyzer.calculate_herfindahl_index(likes3)
        print(f"\n测试用例3 (完全分散):")
        print(f"  输入: {likes3}")
        print(f"  预期: {expected3}")
        print(f"  实际: {result3:.6f}")
        print(f"  结果: {'✓ 正确' if abs(result3 - expected3) < 0.001 else '✗ 错误'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Herfindahl指数测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试群体从众心理指数分析器...")
    
    # 运行Herfindahl指数测试
    herfindahl_result = test_herfindahl_index()
    
    # 运行主要功能测试
    main_result = test_group_conformity_analyzer()
    
    # 输出总体测试结果
    print("\n=== 总体测试结果 ===")
    print(f"Herfindahl指数测试: {'✓ 通过' if herfindahl_result else '✗ 失败'}")
    print(f"主要功能测试: {'✓ 通过' if main_result else '✗ 失败'}")
    
    if herfindahl_result and main_result:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查问题。")
        sys.exit(1)