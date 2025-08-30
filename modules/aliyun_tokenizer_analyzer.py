#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿里云中文分词客户端
- 提供 AliyunTokenizerClient 类供其他模块使用
- 实现阿里云 NLP 分词 API（GetWsChGeneral）调用

环境变量
- ALIYUN_ACCESS_KEY_ID
- ALIYUN_ACCESS_KEY_SECRET

参考文档：中文分词（基础版）
https://help.aliyun.com/document_detail/181284.html
"""

import os
import json
import hmac
import base64
import hashlib
import random
import string
import urllib.parse
from datetime import datetime
from typing import Dict

import requests


def _percent_encode(value: str) -> str:
    return urllib.parse.quote(value, safe='~')


def _utc_time() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


class AliyunTokenizerClient:
    """阿里云 RPC 签名客户端（HMAC-SHA1）"""

    def __init__(self,
                 access_key_id: str,
                 access_key_secret: str,
                 endpoint: str = 'https://alinlp.cn-hangzhou.aliyuncs.com',
                 api_version: str = '2020-06-29'):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = endpoint.rstrip('/')
        self.api_version = api_version

    def _sign(self, method: str, params: Dict[str, str]) -> str:
        sorted_params = sorted(params.items())
        canonicalized = '&'.join(f"{_percent_encode(k)}={_percent_encode(v)}" for k, v in sorted_params)
        string_to_sign = f"{method}&%2F&{_percent_encode(canonicalized)}"
        key = f"{self.access_key_secret}&".encode('utf-8')
        h = hmac.new(key, string_to_sign.encode('utf-8'), hashlib.sha1)
        signature = base64.b64encode(h.digest()).decode('utf-8')
        return signature

    def get_ws_ch_general(self, text: str, tokenizer_id: str = 'GENERAL_CHN', out_type: str = '1') -> Dict:
        """调用 GetWsChGeneral 获取分词结果"""
        params = {
            'Format': 'JSON',
            'Version': self.api_version,
            'AccessKeyId': self.access_key_id,
            'SignatureMethod': 'HMAC-SHA1',
            'Timestamp': _utc_time(),
            'SignatureVersion': '1.0',
            'SignatureNonce': ''.join(random.choices(string.ascii_letters + string.digits, k=16)),
            'Action': 'GetWsChGeneral',
            'ServiceCode': 'alinlp',
            'Text': text,
            'TokenizerId': tokenizer_id,
            'OutType': out_type,
        }
        params['Signature'] = self._sign('GET', params)
        url = f"{self.endpoint}/?{urllib.parse.urlencode(params)}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()


