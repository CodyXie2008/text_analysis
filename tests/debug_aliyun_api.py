#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试阿里云API错误
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

from core.aliyun_api_manager import AliyunAPIManager, get_aliyun_api_manager

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 测试文本
test_texts = [
    "大家的愿望我已经发送",  # 父评论
    "谢谢",  # 典型子评论
    "谢谢兄弟"
]

def debug_api_errors():
    """
    调试阿里云API错误
    """
    try:
        # 获取API管理器
        logger.info("初始化阿里云API管理器...")
        api_manager = get_aliyun_api_manager()
        if not api_manager:
            logger.error("无法初始化阿里云API管理器")
            return
        
        logger.info("API管理器初始化成功")
        
        # 测试情感分析API
        logger.info("\n测试情感分析API:")
        for i, text in enumerate(test_texts):
            logger.info(f"\n测试文本 {i+1}: {text}")
            try:
                # 记录完整的调用过程
                logger.debug("准备调用analyze_sentiment方法")
                result = api_manager.analyze_sentiment(text)
                logger.debug(f"API返回结果: {result}")
                logger.info(f"情感分析成功: 情感={result.get('sentiment')}, 得分={result.get('score')}, 置信度={result.get('confidence')}")
            except Exception as e:
                logger.error(f"情感分析失败: {str(e)}")
                logger.error(f"错误类型: {type(e).__name__}")
                logger.error("详细错误栈:")
                logger.error(traceback.format_exc())
        
        # 测试文本向量API
        logger.info("\n测试文本向量API:")
        for i, text in enumerate(test_texts):
            logger.info(f"\n测试文本 {i+1}: {text}")
            try:
                logger.debug("准备调用get_text_vector方法")
                vector = api_manager.get_text_vector(text)
                logger.debug(f"向量长度: {len(vector) if vector else 0}")
                logger.info(f"文本向量获取成功: 向量维度={len(vector) if vector else 0}")
            except Exception as e:
                logger.error(f"文本向量获取失败: {str(e)}")
                logger.error(f"错误类型: {type(e).__name__}")
                logger.error("详细错误栈:")
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"整体执行失败: {str(e)}")
        logger.error("详细错误栈:")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    debug_api_errors()