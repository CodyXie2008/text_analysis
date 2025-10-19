#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
根据数据预处理文档实现标准化清洗与结构化处理
"""

import os
import sys
import re
import json
import yaml
import argparse
from datetime import datetime
from typing import Dict, List, Union, Optional, Set
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import jieba

from core.base_analyzer import BaseAnalyzer, create_parser, parse_common_args
from core.db_config import get_db_conn
from core.data_paths import AnalysisPathManager, PROJECT_ROOT, ensure_directories

class DataPreprocessingAnalyzer(BaseAnalyzer):
    """数据预处理分析器"""
    
    def __init__(self, video_id: str = None):
        super().__init__("preprocessing", video_id)
        self.config = self._load_config()
        self.stop_words = self._load_stop_words()
        self._load_user_dict()
        # 确保所有目录存在
        ensure_directories()
        # 使用默认输出目录
        self.processed_dir = PROJECT_ROOT / 'data' / 'processed'
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config = {
            'text': {
                'min_length': 2,
                'stopwords_path': 'hit_stopwords.txt',
                'jieba_dict': 'userdict.txt'
            },
            'filter': {
                'stopwords': ['关注我', '点击', '链接', '广告', '代购', '加微信', '加qq']
            },
            'output': {
                'json_path': './processed_json'
            }
        }
        
        # 尝试加载配置文件
        config_path = PROJECT_ROOT / 'config.yaml'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        # 更新配置
                        for section, values in loaded_config.items():
                            if section in config and isinstance(values, dict):
                                config[section].update(values)
                        print(f"✅ 成功加载配置文件: {config_path}")
            except Exception as e:
                print(f"⚠️  加载配置文件失败: {e}，使用默认配置")
        else:
            print("⚠️  配置文件不存在，使用默认配置")
        
        return config
    
    def _load_user_dict(self):
        """加载用户词典"""
        try:
            dict_path = self.config['text']['jieba_dict']
            if not os.path.isabs(dict_path):
                # 尝试在modules目录下查找
                module_dir = os.path.dirname(os.path.abspath(__file__))
                dict_path = os.path.join(module_dir, dict_path)
            
            if os.path.exists(dict_path):
                jieba.load_userdict(dict_path)
                print(f"✅ 成功加载用户词典: {dict_path}")
            else:
                print("⚠️  用户词典文件不存在，使用默认词典")
        except Exception as e:
            print(f"⚠️  加载用户词典失败: {e}")
    
    def _load_stop_words(self):
        """加载停用词"""
        stop_words = set()
        try:
            # 从配置中获取停用词路径
            stopwords_path = self.config['text']['stopwords_path']
            if not os.path.isabs(stopwords_path):
                # 尝试在modules目录下查找
                module_dir = os.path.dirname(os.path.abspath(__file__))
                stop_words_file = os.path.join(module_dir, stopwords_path)
            else:
                stop_words_file = stopwords_path
            
            if os.path.exists(stop_words_file):
                with open(stop_words_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 跳过注释行和空行
                        if line.strip().startswith('#') or not line.strip():
                            continue
                        word = line.strip()
                        # 只保留有效的中文、英文停用词
                        if word and (re.search(r'[\u4e00-\u9fa5]', word) or re.search(r'[a-zA-Z]', word)):
                            stop_words.add(word)
                print(f"✅ 成功加载 {len(stop_words)} 个停用词")
            else:
                # 使用基础停用词
                stop_words = {
                    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
                    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
                    '自己', '这', '那', '他', '她', '它', '们', '个', '只', '条', '张', '片'
                }
                print(f"⚠️ 使用默认停用词集")
                
        except Exception as e:
            print(f"加载停用词失败: {e}")
            # 确保即使加载失败也有基本停用词可用
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一'}
        
        # 添加配置文件中的过滤词
        if 'stopwords' in self.config.get('filter', {}):
            filter_stopwords = self.config['filter']['stopwords']
            stop_words.update(filter_stopwords)
            print(f"✅ 从配置中添加 {len(filter_stopwords)} 个过滤词")
            
        return stop_words
    
    def _load_from_database(self, limit: Optional[int] = None) -> pd.DataFrame:
        """从数据库加载数据"""
        print("=== 从数据库加载评论数据 ===")
        
        # 使用pymysql直接查询，不通过pandas
        sql = "SELECT * FROM douyin_aweme_comment WHERE content IS NOT NULL"
        
        params = []
        
        if self.video_id:
            sql += " AND aweme_id = %s"
            params = [self.video_id]
            print(f"加载视频 {self.video_id} 的评论...")
        else:
            print("加载所有评论...")
        
        # 先添加ORDER BY，再添加LIMIT
        sql += " ORDER BY create_time DESC"
        
        if limit:
            sql += f" LIMIT {limit}"
            print(f"限制加载数量: {limit}")
        
        try:
            # 直接使用conn.cursor()获取游标
            with self.conn.cursor() as cursor:
                print(f"执行SQL: {sql}")
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                print(f"直接从数据库获取到 {len(rows)} 条记录")
                
                # 打印前2条原始数据用于调试
                if rows:
                    print("前2条原始数据:")
                    for i, row in enumerate(rows[:2]):
                        print(f"  记录{i+1}: {row}")
                        # 检查content字段是否为实际内容
                        if isinstance(row, dict):
                            content = row.get('content', '无内容')
                            print(f"    content字段: {content}")
                        else:
                            print(f"    记录类型: {type(row)}")
                
                # 将获取的数据转换为DataFrame
                if rows:
                    # 确保rows中的元素都是字典
                    if isinstance(rows[0], dict):
                        df = pd.DataFrame(rows)
                    else:
                        # 如果是元组，需要获取列名
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            df = pd.DataFrame(rows, columns=columns)
                        else:
                            # 无法获取列名，使用默认列名
                            df = pd.DataFrame(rows)
                    
                    # 确保必要的列存在
                    required_columns = ['comment_id', 'parent_comment_id', 'content', 'aweme_id', 'create_time']
                    
                    # 处理可能的列名差异
                    column_mapping = {
                        'parent_comment_id': ['parent_comment_id', 'parent_id'],
                        'user_id': ['user_id', 'userid'],
                        'nickname': ['nickname', 'user_nickname'],
                        'user_signature': ['user_signature', 'signature'],
                        'ip_location': ['ip_location', 'location']
                    }
                    
                    # 重命名列
                    for target_col, source_cols in column_mapping.items():
                        for source_col in source_cols:
                            if source_col in df.columns and target_col not in df.columns:
                                df = df.rename(columns={source_col: target_col})
                                print(f"⚠️  将列 '{source_col}' 重命名为 '{target_col}'")
                                break
                    
                    # 确保所有必要列都存在
                    for col in required_columns:
                        if col not in df.columns:
                            print(f"⚠️  缺少列 '{col}'，创建空列")
                            df[col] = ''
                    
                    # 数据类型转换
                    numeric_columns = ['like_count', 'sub_comment_count']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    
                    print(f"✅ 成功加载并转换 {len(df)} 条评论到DataFrame")
                    # 打印前2条数据用于调试
                    if len(df) > 0:
                        print("前2条数据样本:")
                        for i in range(min(2, len(df))):
                            content = str(df.iloc[i]['content']) if pd.notna(df.iloc[i]['content']) else '无内容'
                            print(f"  记录{i+1}: 评论ID={df.iloc[i]['comment_id']}, 内容预览={content[:30]}...")
                    return df
                else:
                    # 返回空的DataFrame
                    print("⚠️  没有获取到数据，返回空DataFrame")
                    return pd.DataFrame(columns=['comment_id', 'parent_comment_id', 'content', 'aweme_id', 'create_time'])
        except Exception as e:
            print(f"❌ 数据库加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def clean_text(self, text: str) -> str:
        """文本清洗"""
        if not text:
            return ""
        
        # 1. 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 3. 移除@用户（格式：@用户名 或 @用户名空格）
        text = re.sub(r'@[^\s]+', '', text)
        
        # 4. 去除方括号内容（如[泪奔][666][比心]等表情）
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # 5. 去除多余空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. 保留中文、英文、数字、中文标点，去除其他符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\u3000-\u303f\uff00-\uffef]', '', text)
        
        return text
    
    def segment_text(self, text: str) -> List[str]:
        """文本分词（优化版）"""
        if not text:
            return []
        
        # 使用jieba分词，启用HMM模型提高未登录词识别率
        words = jieba.lcut(text, HMM=True)
        
        # 过滤停用词和低质量词汇
        filtered_words = []
        # 定义常见的无意义单字符（除了数字和重要的量词）
        meaningless_chars = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 临时存储连续数字和英文
        temp_num_eng = ""
        
        for word in words:
            word = word.strip()
            if not word:
                continue
            
            # 处理连续的数字和英文
            if re.match(r'^[0-9a-zA-Z]+$', word):
                temp_num_eng += word
                continue
            else:
                # 如果临时存储不为空，先处理它
                if temp_num_eng:
                    if len(temp_num_eng) > 1:  # 数字和英文组合保留
                        filtered_words.append(temp_num_eng)
                    temp_num_eng = ""
            
            # 过滤条件：
            # 1. 不是停用词
            # 2. 不是无意义单字符
            # 3. 长度大于1 或者 是重要的量词/数字
            if (word not in self.stop_words and 
                word not in meaningless_chars and 
                (len(word) > 1 or (len(word) == 1 and word.isdigit()))):
                filtered_words.append(word)
        
        # 处理最后剩余的数字和英文
        if temp_num_eng and len(temp_num_eng) > 1:
            filtered_words.append(temp_num_eng)
        
        # 分词后处理：合并重复的词（如"哈哈哈哈" -> "哈哈"）
        if len(filtered_words) > 0:
            processed_words = [filtered_words[0]]
            for word in filtered_words[1:]:
                # 检查是否与前一个词重复且都是简单词
                if word != processed_words[-1] or len(word) > 2:
                    processed_words.append(word)
            return processed_words
        
        return filtered_words
    
    def build_comment_tree(self, df: pd.DataFrame) -> Dict:
        """构建评论树结构，只保留建模所需的必要字段，优化结构便于电脑调用和人工观察
        
        优化点：
        1. sub_comment_count仅在父评论中显示
        2. 将sub_comment_count放在children前面，使结构更清晰
        3. 优化字段顺序，使核心信息在前面展示
        """
        print("=== 构建评论树 ===")
        
        # 为每个评论创建字典，只保留建模所需的必要字段
        comment_dict = {}
        for idx, row in df.iterrows():
            # 安全地处理每一行数据
            try:
                row_dict = row.to_dict()
                # 确保comment_id存在且为字符串
                comment_id = str(row_dict.get('comment_id', f'comment_{idx}'))
                parent_id = str(row_dict.get('parent_comment_id', ''))
                
                # 优化字段顺序，将核心信息放在前面
                # 优先使用清洗后的content_cleaned字段，如果不存在则使用原始content
                content_value = row_dict.get('content_cleaned', row_dict.get('content', ''))
                essential_comment = {
                    'comment_id': comment_id,
                    'parent_comment_id': parent_id,
                    'content': content_value,
                    'datetime': row_dict.get('datetime', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'like_count': int(row_dict.get('like_count', 0)) if pd.notna(row_dict.get('like_count')) else 0,
                    'user_id': str(row_dict.get('user_id', '')),
                    'nickname': row_dict.get('nickname', ''),
                    'ip_location': row_dict.get('ip_location', ''),
                    'tokens': row_dict.get('tokens', []),
                    'children': []
                    # sub_comment_count暂时不在这里添加，只在需要时添加到父评论
                }
                
                comment_dict[comment_id] = essential_comment
            except Exception as e:
                print(f"   - 处理行 {idx} 时出错: {e}")
                continue
        
        # 构建评论树
        root_comments = []
        parent_found = set()
        
        # 第一次遍历：处理ID格式，确保一致性
        for comment_id in list(comment_dict.keys()):
            comment = comment_dict[comment_id]
            # 确保所有ID都是字符串
            comment_id_str = str(comment.get('comment_id', comment_id))
            parent_id_str = comment.get('parent_comment_id')
            
            # 转换parent_id为字符串进行比较
            if pd.isna(parent_id_str) or parent_id_str is None:
                parent_id_str = ''
            parent_id_str = str(parent_id_str)
            
            # 更新comment中的ID值
            comment['comment_id'] = comment_id_str
            comment['parent_comment_id'] = parent_id_str
        
        # 找出所有父评论ID，这些评论需要添加sub_comment_count
        potential_parent_ids = set()
        for comment_id in comment_dict:
            parent_id = comment_dict[comment_id].get('parent_comment_id')
            if parent_id and parent_id != '0' and parent_id in comment_dict:
                potential_parent_ids.add(parent_id)
        
        # 为所有潜在的父评论添加sub_comment_count字段，初始化为0
        for parent_id in potential_parent_ids:
            # 创建新的有序字典，确保sub_comment_count在children前面
            original_comment = comment_dict[parent_id]
            ordered_comment = {}
            
            # 按照优化的顺序添加字段
            for key in ['comment_id', 'parent_comment_id', 'content', 'datetime', 
                       'like_count', 'user_id', 'nickname', 'ip_location', 'tokens']:
                if key in original_comment:
                    ordered_comment[key] = original_comment[key]
            
            # 添加sub_comment_count，放在children前面
            ordered_comment['sub_comment_count'] = 0
            ordered_comment['children'] = original_comment.get('children', [])
            
            comment_dict[parent_id] = ordered_comment
        
        # 第二次遍历：构建树结构并更新子评论计数
        for comment_id in list(comment_dict.keys()):
            comment = comment_dict[comment_id]
            parent_id = comment.get('parent_comment_id')
            
            # 更宽松的根评论识别：空、0、'0'、'null'、'None'等都视为根评论
            if not parent_id or parent_id.lower() in ['0', 'null', 'none']:
                root_comments.append(comment)
            else:
                # 查找父评论并添加为子评论
                if parent_id in comment_dict:
                    comment_dict[parent_id]['children'].append(comment)
                    # 如果父评论有sub_comment_count字段，则增加计数
                    if 'sub_comment_count' in comment_dict[parent_id]:
                        comment_dict[parent_id]['sub_comment_count'] += 1
                    parent_found.add(comment_id)
        
        # 处理孤立的子评论（父评论不存在），将它们提升为根评论
        for comment_id in list(comment_dict.keys()):
            if comment_id not in parent_found and comment_id not in [c['comment_id'] for c in root_comments]:
                root_comments.append(comment_dict[comment_id])
        
        # 构建最终的树结构
        tree = {'root': root_comments}
        
        parent_count = len(root_comments)
        total_comments = len(comment_dict)
        child_count = total_comments - parent_count
        
        print(f"   - 根评论数量: {parent_count}")
        print(f"   - 子评论数量: {child_count}")
        print(f"   - 评论树构建完成")
        
        return tree
    
    def analyze(self, df: pd.DataFrame, strict_mode: bool = False) -> Dict:
        """执行数据预处理分析"""
        min_length = self.config['text']['min_length']  # 从配置获取最小长度限制
        print(f"=== 开始数据预处理 ===")
        print(f"   - 原始数据总数: {len(df)}")
        print(f"   - 最小长度限制: {min_length}")
        
        # 记录原始数据统计
        original_stats = {
            'total_comments': int(len(df)),
            'unique_videos': int(df['aweme_id'].nunique()),
            'unique_users': int(df['user_id'].nunique()) if 'user_id' in df.columns else 0
        }
        
        # 1. 删除无效内容
        print("1. 过滤无效内容...")
        df_filtered = df[df['content'].str.strip().str.len() >= min_length].copy()
        removed_short = len(df) - len(df_filtered)
        print(f"   - 过滤短内容: {removed_short} 条")
        
        # 2. 去除垃圾信息（广告关键词和模式）
        print("2. 去除垃圾信息...")
        # 从配置获取禁止词，默认提供更全面的广告关键词列表
        default_ban_words = [
            # 常见广告词汇
            '关注我', '点击', '链接', '广告', '代购', '加微信', '加qq',
            '联系方式', '扫一扫', '二维码', '优惠', '折扣', '促销', '返利',
            '赚钱', '兼职', '刷单', '推广', '代理', '加盟', '免费领取',
            '薇\S*信', 'V\S*X', 'v\S*x', 'Q\S*Q', 'q\S*q',
            # 电话号码模式
            '1[3-9]\d{9}', 
            # 网址特征
            'www\.', '.com', '.cn', '.net',
            # 特殊符号组合（常见于广告）
            '💖', '👉', '👇', '🔥', '福利', '红包', '现金',
            # 重复字符（垃圾内容特征）
            '666666', '233333', '哈哈哈哈哈{5,}',
            # 平台引流词汇
            '抖音号', '快手号', '小红书', '微博', '直播间', '粉丝群'
        ]
        ban_words = self.config['filter'].get('stopwords', default_ban_words)
        
        # 创建垃圾内容检测函数
        def is_spam_content(content):
            if not content:
                return False
            
            # 1. 关键词匹配
            for ban_word in ban_words:
                if re.search(ban_word, content, re.IGNORECASE):
                    return True
            
            # 2. 检测电话号码模式
            if re.search(r'1[3-9]\d{9}', content):
                return True
            
            # 3. 检测网址模式
            if re.search(r'(https?://|www\.|\.com|\.cn|\.net)', content, re.IGNORECASE):
                return True
            
            # 4. 检测重复字符过多（垃圾内容特征）
            if re.search(r'(.)\1{5,}', content):
                return True
            
            # 5. 检测emoji表情过多（垃圾内容特征）
            emoji_pattern = re.compile(r'[\u1F600-\u1F6FF\u2600-\u26FF\u2700-\u27BF]')
            emoji_count = len(emoji_pattern.findall(content))
            if emoji_count > 3:
                return True
            
            return False
        
        # 应用垃圾内容过滤
        df_filtered = df_filtered[~df_filtered['content'].apply(is_spam_content)].copy()
        removed_spam = len(df) - removed_short - len(df_filtered)
        print(f"   - 过滤广告内容: {removed_spam} 条")
        
        # 3. 文本清洗
        print("3. 文本清洗...")
        df_filtered['content_cleaned'] = df_filtered['content'].apply(self.clean_text)
        df_filtered = df_filtered[df_filtered['content_cleaned'].str.len() >= min_length].copy()
        print(f"   - 清洗后保留: {len(df_filtered)} 条")
        
        # 3. 文本清洗
        print("3. 文本清洗...")
        df_filtered['content_cleaned'] = df_filtered['content'].apply(self.clean_text)
        df_filtered = df_filtered[df_filtered['content_cleaned'].str.len() >= min_length].copy()
        print(f"   - 清洗后保留: {len(df_filtered)} 条")
        
        # 4. 分词处理
        print("4. 文本分词...")
        df_filtered['tokens'] = df_filtered['content_cleaned'].apply(self.segment_text)
        # 过滤分词后为空的内容
        df_filtered = df_filtered[df_filtered['tokens'].apply(len) > 0].copy()
        print(f"   - 分词后保留: {len(df_filtered)} 条")
        
        # 5. 时间格式化
        print("5. 时间格式化...")
        if 'create_time' in df_filtered.columns:
            try:
                # 更健壮的时间处理方式
                df_filtered['datetime'] = df_filtered['create_time']
                
                # 尝试不同的时间格式处理
                # 1. 检查是否为数字类型（可能是时间戳）
                if pd.api.types.is_numeric_dtype(df_filtered['create_time']):
                    numeric_times = pd.to_numeric(df_filtered['create_time'], errors='coerce')
                    # 尝试秒级时间戳
                    df_filtered['datetime'] = pd.to_datetime(numeric_times, unit='s', errors='ignore')
                    # 对于仍未转换成功的，尝试毫秒级
                    mask = pd.to_datetime(df_filtered['datetime'], errors='coerce').isna()
                    if mask.any():
                        df_filtered.loc[mask, 'datetime'] = pd.to_datetime(numeric_times[mask], unit='ms', errors='ignore')
                else:
                    # 字符串格式时间
                    df_filtered['datetime'] = pd.to_datetime(df_filtered['create_time'], errors='ignore')
                
                # 对于仍无法转换的时间，使用当前时间作为默认值
                mask = pd.to_datetime(df_filtered['datetime'], errors='coerce').isna()
                if mask.any():
                    print(f"   - 使用默认时间替代无效时间: {mask.sum()} 条")
                    df_filtered.loc[mask, 'datetime'] = datetime.now()
                
                # 确保所有时间都转换为字符串格式
                df_filtered['datetime'] = df_filtered['datetime'].astype(str)
                # 对于已成功转换为datetime对象的，格式化输出
                mask = df_filtered['datetime'].str.contains(r'\d{4}-\d{2}-\d{2}')
                if mask.any():
                    df_filtered.loc[mask, 'datetime'] = pd.to_datetime(df_filtered.loc[mask, 'datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"   - 时间格式化完成，保留所有记录")
            except Exception as e:
                print(f"   - 时间格式化出错: {e}")
                # 出错时，为所有记录设置默认时间
                df_filtered['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("   - 使用默认时间")
        
        # 6. 按视频ID分组并构建评论树
        print("6. 按视频分组并构建评论树...")
        
        # 确保有视频ID列
        if 'aweme_id' in df_filtered.columns:
            video_id_col = 'aweme_id'
        elif 'video_id' in df_filtered.columns:
            video_id_col = 'video_id'
        else:
            # 如果没有视频ID列，创建一个默认值
            df_filtered['video_id'] = 'default_video'
            video_id_col = 'video_id'
        
        video_groups = df_filtered.groupby(video_id_col)
        total_videos = len(video_groups)
        print(f"   - 视频总数: {total_videos}")
        
        # 7. 为每个视频保存JSON文件
        saved_videos = 0
        for video_id, group in video_groups:
            try:
                # 确保视频ID非空
                if pd.isna(video_id) or video_id == '':
                    video_id = 'unknown_video'
                
                # 转换为字符串类型
                video_id = str(video_id)
                
                # 使用build_comment_tree方法构建评论树
                tree = self.build_comment_tree(group)
                root_comments = tree['root']
                
                # 准备保存的评论树数据，包含更完整的元数据
                comment_tree = {
                    'video_id': video_id,
                    'total_comments': len(group),
                    'root_comments': len(root_comments),
                    'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'metadata': {
                        'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'data_source': 'douyin_aweme_comment'
                    },
                    'root': root_comments
                }
                
                # 生成带时间戳的文件名：aweme_id_YYYYMMDD_HHMMSS.json
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = os.path.join(self.processed_dir, f"{video_id}_{timestamp}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(comment_tree, f, ensure_ascii=False, indent=2)
                saved_videos += 1
                
            except Exception as e:
                print(f"   - 处理视频 {video_id} 出错: {e}")
        
        print(f"   - 成功保存 {saved_videos} 个视频的评论树到 {self.processed_dir}")
        
        # 最终统计
        final_stats = {
            'total_comments': int(len(df_filtered)),
            'unique_videos': int(df_filtered[video_id_col].nunique()),
            'saved_videos': saved_videos,
            'removed_short': removed_short,
            'removed_spam': removed_spam,
            'output_dir': str(self.processed_dir)
        }
        
        return {
            'original_stats': original_stats,
            'final_stats': final_stats,
            'output_dir': str(self.processed_dir)
        }
    
    def save_results(self, df: pd.DataFrame, results: Dict):
        """保存处理结果（已移除preprocessing_stats.json生成）"""
        # 根据需求移除了生成preprocessing_stats.json文件的功能
        # 预处理结果已经通过各个视频的JSON文件保存，不需要额外的统计文件
        pass

def main():
    """主函数"""
    parser = create_parser("preprocessing", "数据预处理工具")
    # 添加预处理特定参数
    parser.add_argument('--min-length', type=int, help='最小内容长度（覆盖配置文件）')
    parser.add_argument('--output-dir', type=str, help='自定义输出目录')
    parser.add_argument('--stopwords-file', type=str, help='自定义停用词文件路径')
    
    args = parser.parse_args()
    
    # 解析通用参数
    args = parse_common_args(parser, args)
    
    # 创建分析器
    analyzer = DataPreprocessingAnalyzer(args.video_id)
    
    # 如果命令行参数指定了最小长度，覆盖配置
    if args.min_length is not None:
        analyzer.config['text']['min_length'] = args.min_length
        print(f"⚠️  使用命令行指定的最小长度: {args.min_length}")
    
    # 如果命令行参数指定了停用词文件，重新加载
    if args.stopwords_file:
        analyzer.config['text']['stopwords_path'] = args.stopwords_file
        analyzer.stop_words = analyzer._load_stop_words()
    
    # 运行预处理
    # 禁用报告和可视化，只生成处理后的数据JSON
    analyzer.run_analysis(
        limit=args.limit,
        create_visualizations=False,
        generate_report=False
    )

if __name__ == "__main__":
    main()