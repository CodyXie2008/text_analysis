#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿里云API统一管理器
整合分词、情感分析、文本向量等功能的统一调用接口

环境变量配置：
- ALIYUN_ACCESS_KEY_ID: 阿里云访问密钥ID
- ALIYUN_ACCESS_KEY_SECRET: 阿里云访问密钥Secret
- ALIYUN_NLP_ENDPOINT: NLP服务端点（可选，默认cn-hangzhou）
- ALIYUN_TOKENIZER_ID: 分词器ID（可选，默认GENERAL_CHN）
- ALIYUN_OUT_TYPE: 分词输出类型（可选，默认1）

支持的API：
1. 中文分词（基础版）- GetWsChGeneral
2. 情感分析（基础版）- GetSaChGeneral  
3. 文本向量（基础版）- GetWeChGeneral
"""

import os
import json
import hmac
import base64
import hashlib
import random
import string
import urllib.parse
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

import requests

# 配置日志
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()
# 尝试加载core目录下的.env文件
import os
core_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(core_env_path):
    load_dotenv(core_env_path)


def _percent_encode(value: str) -> str:
    """URL编码"""
    return urllib.parse.quote(value, safe='~')


def _utc_time() -> str:
    """获取UTC时间"""
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


class AliyunAPIManager:
    """阿里云API统一管理器"""
    
    def __init__(self, 
                 access_key_id: str = None,
                 access_key_secret: str = None,
                 endpoint: str = None,
                 region: str = None):
        """
        初始化阿里云API管理器
        
        Args:
            access_key_id: 访问密钥ID，默认从环境变量读取
            access_key_secret: 访问密钥Secret，默认从环境变量读取
            endpoint: 服务端点，默认从环境变量读取
            region: 地域，默认从环境变量读取
        """
        # 从环境变量读取配置
        self.access_key_id = access_key_id or os.getenv('NLP_AK_ENV') or os.getenv('ALIYUN_ACCESS_KEY_ID')
        self.access_key_secret = access_key_secret or os.getenv('NLP_SK_ENV') or os.getenv('ALIYUN_ACCESS_KEY_SECRET')
        
        # 设置端点
        if endpoint:
            self.endpoint = endpoint
        elif region:
            self.endpoint = f'https://alinlp.{region}.aliyuncs.com'
        else:
            region = os.getenv('ALIYUN_NLP_ENDPOINT') or os.getenv('NLP_REGION_ENV', 'cn-hangzhou')
            self.endpoint = f'https://alinlp.{region}.aliyuncs.com'
        
        # 默认配置
        self.api_version = '2020-06-29'
        self.service_code = 'alinlp'
        
        # 验证配置
        if not self.access_key_id or not self.access_key_secret:
            raise ValueError("阿里云API配置缺失，请设置 ALIYUN_ACCESS_KEY_ID 和 ALIYUN_ACCESS_KEY_SECRET")
        
        logger.info(f"✅ 阿里云API管理器初始化成功，端点: {self.endpoint}")
    
    def _sign(self, method: str, params: Dict[str, str]) -> str:
        """生成API签名"""
        sorted_params = sorted(params.items())
        canonicalized = '&'.join(f"{_percent_encode(k)}={_percent_encode(v)}" for k, v in sorted_params)
        string_to_sign = f"{method}&%2F&{_percent_encode(canonicalized)}"
        key = f"{self.access_key_secret}&".encode('utf-8')
        h = hmac.new(key, string_to_sign.encode('utf-8'), hashlib.sha1)
        signature = base64.b64encode(h.digest()).decode('utf-8')
        return signature
    
    def _make_request(self, action: str, params: Dict[str, str]) -> Dict:
        """发送API请求"""
        # 构建请求参数
        request_params = {
            'Format': 'JSON',
            'Version': self.api_version,
            'AccessKeyId': self.access_key_id,
            'SignatureMethod': 'HMAC-SHA1',
            'Timestamp': _utc_time(),
            'SignatureVersion': '1.0',
            'SignatureNonce': ''.join(random.choices(string.ascii_letters + string.digits, k=16)),
            'Action': action,
            'ServiceCode': self.service_code,
            **params
        }
        
        # 生成签名
        request_params['Signature'] = self._sign('GET', request_params)
        
        # 构建URL
        url = f"{self.endpoint}/?{urllib.parse.urlencode(request_params)}"
        
        try:
            # 发送请求
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            # 记录请求日志
            request_id = result.get('RequestId', 'Unknown')
            logger.info(f"[AliyunAPI] {action} RequestId={request_id}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[AliyunAPI] {action} 请求失败: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"[AliyunAPI] {action} 响应解析失败: {e}")
            raise
    
    def segment_text(self, text: str, tokenizer_id: str = 'GENERAL_CHN', out_type: str = '1') -> List[str]:
        """
        中文分词
        
        Args:
            text: 待分词文本
            tokenizer_id: 分词器ID，默认GENERAL_CHN
            out_type: 输出类型，默认1
            
        Returns:
            分词结果列表
        """
        if not text or not text.strip():
            return []
        
        # 截断文本（API限制）
        text = text[:1024]
        
        params = {
            'Text': text,
            'TokenizerId': tokenizer_id,
            'OutType': out_type,
        }
        
        try:
            result = self._make_request('GetWsChGeneral', params)
            
            # 解析结果
            data_field = result.get('Data')
            if not data_field:
                return []
            
            payload = json.loads(data_field) if isinstance(data_field, str) else data_field
            result_list = payload.get('result', []) if isinstance(payload, dict) else []
            
            # 提取词汇
            words = [item.get('word') for item in result_list if isinstance(item, dict) and item.get('word')]
            
            # 记录样本词汇
            if words:
                sample_words = words[:3]
                logger.info(f"[AliyunSeg] 分词成功，样本词汇: {sample_words}")
            
            return words
            
        except Exception as e:
            logger.error(f"[AliyunSeg] 分词失败: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        情感分析
        
        Args:
            text: 待分析文本
            
        Returns:
            情感分析结果字典
        """
        if not text or not text.strip():
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        # 截断文本（API限制）
        text = text[:1024]
        
        params = {
            'Text': text,
        }
        
        try:
            result = self._make_request('GetSaChGeneral', params)
            
            # 解析结果
            data_field = result.get('Data')
            if not data_field:
                return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
            
            payload = json.loads(data_field) if isinstance(data_field, str) else data_field
            sentiment_result = payload.get('result', {}) if isinstance(payload, dict) else {}
            
            # 提取情感信息
            sentiment = sentiment_result.get('sentiment', 'neutral')
            score = float(sentiment_result.get('score', 0.0))
            confidence = float(sentiment_result.get('confidence', 0.0))
            
            logger.info(f"[AliyunSA] 情感分析成功: {sentiment} (score={score:.3f}, confidence={confidence:.3f})")
            
            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"[AliyunSA] 情感分析失败: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
    
    def get_text_vector(self, text: str, size: str = '100', vector_type: str = 'word', operation: str = 'average') -> List[float]:
        """
        获取文本向量
        
        Args:
            text: 待向量化文本
            size: 向量维度，默认100
            vector_type: 向量类型，默认word
            operation: 操作类型，默认average
            
        Returns:
            文本向量列表
        """
        if not text or not text.strip():
            return []
        
        # 截断文本（API限制）
        text = text[:1024]
        
        params = {
            'Text': text,
            'Size': size,
            'Type': vector_type,
            'Operation': operation,
        }
        
        try:
            result = self._make_request('GetWeChGeneral', params)
            
            # 解析结果
            data_field = result.get('Data')
            if not data_field:
                return []
            
            payload = json.loads(data_field) if isinstance(data_field, str) else data_field
            vector_result = payload.get('result', {}) if isinstance(payload, dict) else {}
            
            # 提取向量
            if isinstance(vector_result, dict) and 'vec' in vector_result:
                vector = vector_result['vec']
            elif isinstance(vector_result, list) and vector_result:
                vectors = [item.get('vec', []) for item in vector_result if isinstance(item, dict) and 'vec' in item]
                if vectors:
                    import numpy as np
                    vector = np.mean(vectors, axis=0).tolist()
                else:
                    vector = []
            else:
                vector = []
            
            if vector:
                logger.info(f"[AliyunVec] 文本向量化成功，维度: {len(vector)}")
            
            return vector if isinstance(vector, list) else []
            
        except Exception as e:
            logger.error(f"[AliyunVec] 文本向量化失败: {e}")
            return []
    
    def batch_segment_texts(self, texts: List[str], tokenizer_id: str = 'GENERAL_CHN', out_type: str = '1') -> List[List[str]]:
        """
        批量分词
        
        Args:
            texts: 文本列表
            tokenizer_id: 分词器ID
            out_type: 输出类型
            
        Returns:
            分词结果列表的列表
        """
        results = []
        for i, text in enumerate(texts):
            try:
                words = self.segment_text(text, tokenizer_id, out_type)
                results.append(words)
                # 添加延迟避免API限制
                if i < len(texts) - 1:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"[AliyunSeg] 批量分词第{i+1}个文本失败: {e}")
                results.append([])
        return results
    
    def batch_analyze_sentiments(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        批量情感分析
        
        Args:
            texts: 文本列表
            
        Returns:
            情感分析结果列表
        """
        results = []
        for i, text in enumerate(texts):
            try:
                sentiment = self.analyze_sentiment(text)
                results.append(sentiment)
                # 添加延迟避免API限制
                if i < len(texts) - 1:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"[AliyunSA] 批量情感分析第{i+1}个文本失败: {e}")
                results.append({'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0})
        return results


# 全局API管理器实例
_global_api_manager = None


def get_aliyun_api_manager() -> Optional[AliyunAPIManager]:
    """
    获取全局阿里云API管理器实例
    
    Returns:
        API管理器实例，如果配置无效则返回None
    """
    global _global_api_manager
    
    if _global_api_manager is None:
        try:
            _global_api_manager = AliyunAPIManager()
        except ValueError as e:
            logger.warning(f"阿里云API配置无效: {e}")
            return None
    
    return _global_api_manager


def is_aliyun_api_available() -> bool:
    """
    检查阿里云API是否可用
    
    Returns:
        是否可用
    """
    return get_aliyun_api_manager() is not None
