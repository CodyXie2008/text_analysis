#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单独计算指定评论的群体从众指数
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, List

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from modules.group_conformity_analyzer import GroupConformityAnalyzer

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 目标评论ID
target_comment_id = "7306505996271829810"

# 目标文件路径
target_file_path = "e:/Creation/mycode/MediaCrawler/text_analysis/data/processed/7306437681045654834_20251019_224320.json"

def load_target_comment() -> Dict:
    """
    从文件中加载目标评论数据
    
    Returns:
        Dict: 目标评论数据
    """
    logger.info(f"从文件 {target_file_path} 加载目标评论 {target_comment_id}...")
    
    try:
        with open(target_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if 'root' in data and isinstance(data['root'], list):
                for comment in data['root']:
                    if comment.get('comment_id') == target_comment_id:
                        logger.info(f"找到目标评论，包含 {len(comment.get('children', []))} 个子评论")
                        return comment
            
        logger.error(f"未找到目标评论 {target_comment_id}")
        return None
    except Exception as e:
        logger.error(f"加载文件失败: {e}")
        return None

def main():
    """主函数"""
    try:
        # 加载目标评论数据
        target_comment = load_target_comment()
        if not target_comment:
            sys.exit(1)
        
        # 创建分析器
        analyzer = GroupConformityAnalyzer()
        
        # 计算群体从众指数
        logger.info("开始计算群体从众指数...")
        result = analyzer.calculate_group_conformity(target_comment)
        
        # 输出结果
        logger.info("\n群体从众指数计算结果:")
        logger.info(f"评论ID: {result['comment_id']}")
        logger.info(f"群体从众指数: {result['group_conformity']:.4f}")
        logger.info(f"语义趋同度: {result['semantic_similarity']:.4f}")
        logger.info(f"情绪一致性: {result['emotion_alignment']:.4f}")
        logger.info(f"点赞集中度: {result['like_concentration']:.4f}")
        logger.info(f"评论总数: {result['total_comments']}")
        logger.info(f"父评论点赞数: {result['parent_likes']}")
        logger.info(f"子评论数: {result['sub_comments_count']}")
        
        # 保存结果到文件
        output_file = f"single_comment_{target_comment_id}_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"\n结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"计算过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()