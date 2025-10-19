# 数据预处理模块技术文档

## 1. 概述

`data_preprocessing.py` 是 MediaCrawler 项目中的核心数据预处理模块，负责对视频评论数据进行清洗、过滤、分词和结构化处理，为后续的文本分析任务提供高质量的输入数据。该模块实现了完整的数据流水线，包括：数据加载、无效内容过滤、垃圾信息识别、文本清洗、分词处理、评论树构建和结果序列化存储。

## 2. 核心类与方法

### 2.1 DataPreprocessingAnalyzer 类

该类继承自 `BaseAnalyzer`，是整个预处理流程的核心实现。

```python
class DataPreprocessingAnalyzer(BaseAnalyzer):
    def __init__(self, video_id: str = None):
        # 初始化配置和资源
        # video_id参数用于筛选特定视频的评论
```

#### 2.1.1 初始化方法

```python
def __init__(self, video_id: str = None):
    super().__init__("preprocessing", video_id)
    # 加载配置
    self.config = self._load_config()
    # 初始化数据路径
    self.processed_dir = os.path.join(self.path_manager.processed_dir, "preprocessing")
    ensure_directories([self.processed_dir])
    # 加载用户词典和停用词
    self._load_user_dict()
    self.stop_words = self._load_stop_words()
```

**功能**: 初始化分析器，加载配置、用户词典和停用词，确保输出目录存在。

### 2.2 配置与资源加载

#### 2.2.1 配置加载 (`_load_config`)

```python
def _load_config(self) -> Dict:
    # 基础配置
    config = {
        'text': {
            'min_length': 5,  # 最小文本长度
            'user_dict_path': os.path.join(PROJECT_ROOT, 'resources', 'user_dict.txt'),
            'stopwords_path': os.path.join(PROJECT_ROOT, 'resources', 'stopwords.txt')
        },
        'filter': {
            'enabled': True,
            'strict_mode': False
        }
    }
    # 加载自定义配置（如果存在）
    # ...
    return config
```

**功能**: 加载并返回预处理配置，包括文本长度要求、词典路径和过滤设置。

#### 2.2.2 用户词典加载 (`_load_user_dict`)

```python
def _load_user_dict(self):
    user_dict_path = self.config['text'].get('user_dict_path')
    if os.path.exists(user_dict_path):
        try:
            jieba.load_userdict(user_dict_path)
            print(f"✅ 加载用户词典: {user_dict_path}")
        except Exception as e:
            print(f"❌ 加载用户词典失败: {e}")
    else:
        print(f"⚠️ 用户词典不存在: {user_dict_path}")
```

**功能**: 加载自定义用户词典到jieba分词器，提升专业词汇和领域词的分词准确性。

#### 2.2.3 停用词加载 (`_load_stop_words`)

```python
def _load_stop_words(self) -> Set[str]:
    stopwords_path = self.config['text'].get('stopwords_path')
    stop_words = set()
    if os.path.exists(stopwords_path):
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
            print(f"✅ 加载停用词: {stopwords_path}, 共 {len(stop_words)} 个")
        except Exception as e:
            print(f"❌ 加载停用词失败: {e}")
    else:
        print(f"⚠️ 停用词文件不存在: {stopwords_path}")
    
    # 添加额外的停用词
    additional_stop_words = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
        '会', '着', '没有', '看', '好', '自己', '这', '。', ',', '，', '！', '!', '？', '?', '；', ';', '：', ':', 
        '(', ')', '[', ']', '{', '}', '<', '>', '《', '》', '、', '/', '\', '|', '“', '”', '"', "'", "‘", "’", 
        '...', '…', '*', '**', '***', '****', '*****', '******', '（', '）', '　', '\t', '\n', '\r'
    }
    stop_words.update(additional_stop_words)
    return stop_words
```

**功能**: 加载停用词词典，并合并额外的通用停用词，用于后续分词后的过滤。

### 2.3 数据加载

#### 2.3.1 数据库加载 (`_load_from_database`)

```python
def _load_from_database(self, limit: Optional[int] = None) -> pd.DataFrame:
    # 从数据库获取评论数据
    conn = get_db_conn()
    try:
        # 构建SQL查询
        query = """SELECT 
            comment_id, parent_comment_id, content, create_time, 
            like_count, user_id, nickname, ip_location, aweme_id
        FROM douyin_aweme_comment"""
        
        # 如果指定了video_id，添加筛选条件
        if self.video_id:
            query += f" WHERE aweme_id = '{self.video_id}'"
        
        # 添加限制
        if limit:
            query += f" LIMIT {limit}"
        
        # 执行查询
        df = pd.read_sql(query, conn)
        print(f"✅ 从数据库加载 {len(df)} 条评论数据")
        return df
    except Exception as e:
        print(f"❌ 从数据库加载失败: {e}")
        # 返回空DataFrame
        return pd.DataFrame()
```

**功能**: 从数据库加载评论数据，支持按视频ID筛选和限制返回记录数。

### 2.4 文本处理算法

#### 2.4.1 文本清洗 (`clean_text`)

**算法**: 
1. 去除多余空白字符
2. 移除@提及和话题标签
3. 清理特殊字符和HTML标签
4. 统一空格和标点符号
5. 标准化中文字符（全角转半角等）

```python
def clean_text(self, text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    # 转换为字符串并去除首尾空白
    text = str(text).strip()
    
    # 移除@提及
    text = re.sub(r'@\w+', '', text)
    # 移除话题标签
    text = re.sub(r'#.*?#', '', text)
    # 移除URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 移除多余标点
    text = re.sub(r'([.!?,;:])\1+', r'\1', text)
    # 去除首尾空白
    text = text.strip()
    
    return text
```

#### 2.4.2 分词处理 (`segment_text`)

**算法**: 
1. 使用jieba分词器进行中文分词，启用HMM模型提高生僻词识别率
2. 过滤停用词和单字符（除了有意义的数字和英文）
3. 合并连续的数字和英文字符

```python
def segment_text(self, text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    
    # 使用jieba分词，启用HMM模型提高生僻词识别
    words = jieba.cut(text, cut_all=False, HMM=True)
    
    # 过滤停用词和单字符（除了有意义的数字和英文）
    result = []
    for word in words:
        # 跳过停用词
        if word in self.stop_words:
            continue
        
        # 处理单字符：过滤无意义的单字符，但保留数字和英文
        if len(word) == 1 and not (word.isdigit() or word.isalpha()):
            continue
        
        result.append(word)
    
    # 合并连续的数字和英文字符
    merged = []
    temp_num = []
    temp_alpha = []
    
    for word in result:
        if word.isdigit():
            temp_num.append(word)
            # 如果有积累的英文字符，先添加它们
            if temp_alpha:
                merged.append(''.join(temp_alpha))
                temp_alpha = []
        elif word.isalpha():
            temp_alpha.append(word)
            # 如果有积累的数字，先添加它们
            if temp_num:
                merged.append(''.join(temp_num))
                temp_num = []
        else:
            # 添加积累的数字和英文
            if temp_num:
                merged.append(''.join(temp_num))
                temp_num = []
            if temp_alpha:
                merged.append(''.join(temp_alpha))
                temp_alpha = []
            merged.append(word)
    
    # 添加最后积累的数字和英文
    if temp_num:
        merged.append(''.join(temp_num))
    if temp_alpha:
        merged.append(''.join(temp_alpha))
    
    return merged
```

#### 2.4.3 垃圾信息过滤

**算法**: 在`analyze`方法中实现，采用多模式检测机制：
1. 关键词匹配：使用预定义广告关键词列表
2. 电话号码模式检测：识别中国大陆手机号
3. 网址模式检测：识别URL链接特征
4. 重复字符检测：识别垃圾内容常见的重复字符模式
5. Emoji数量检测：识别表情符号过多的情况

```python
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
```

### 2.5 评论树构建 (`build_comment_tree`)

**算法**: 
1. 遍历所有评论，构建评论ID到评论对象的映射
2. 规范化ID格式，确保一致性
3. 识别潜在的父评论，添加子评论计数字段
4. 构建树结构：
   - 将无父ID或父ID不存在的评论作为根评论
   - 将子评论添加到对应父评论的children列表中
5. 处理孤立的子评论（父评论不存在），提升为根评论

```python
def build_comment_tree(self, df: pd.DataFrame) -> Dict:
    # 构建评论ID到评论对象的映射
    comment_dict = {}
    
    # 遍历DataFrame中的每一行
    for idx, row in df.iterrows():
        # 处理每一行数据，构建评论对象
        # ...
    
    # 构建评论树
    root_comments = []
    parent_found = set()
    
    # 第一次遍历：处理ID格式，确保一致性
    # ...
    
    # 找出所有父评论ID，这些评论需要添加sub_comment_count
    # ...
    
    # 为所有潜在的父评论添加sub_comment_count字段
    # ...
    
    # 第二次遍历：构建树结构并更新子评论计数
    # ...
    
    # 处理孤立的子评论
    # ...
    
    # 构建最终的树结构
    tree = {'root': root_comments}
    
    return tree
```

### 2.6 主分析流程 (`analyze`)

**算法**: 完整的预处理流水线，按顺序执行以下步骤：
1. 记录原始数据统计信息
2. 过滤短内容（基于最小长度限制）
3. 过滤垃圾信息（使用多模式检测）
4. 文本清洗
5. 分词处理
6. 时间格式化
7. 按视频ID分组
8. 构建评论树并保存为JSON文件

```python
def analyze(self, df: pd.DataFrame, strict_mode: bool = False) -> Dict:
    # 从配置获取最小长度限制
    min_length = self.config['text']['min_length']
    
    # 1. 删除无效内容（短内容）
    # ...
    
    # 2. 去除垃圾信息
    # ...
    
    # 3. 文本清洗
    # ...
    
    # 4. 分词处理
    # ...
    
    # 5. 时间格式化
    # ...
    
    # 6. 按视频ID分组并构建评论树
    # ...
    
    # 7. 为每个视频保存JSON文件
    # ...
    
    return {
        'original_stats': original_stats,
        'final_stats': final_stats,
        'output_dir': str(self.processed_dir)
    }
```

## 3. 工作流程

### 3.1 完整数据流程

1. **初始化与配置**
   - 加载配置文件
   - 初始化输出目录
   - 加载用户词典和停用词

2. **数据加载**
   - 从数据库读取评论数据
   - 支持按视频ID筛选和限制返回记录数

3. **数据过滤与清洗**
   - 过滤过短内容
   - 过滤垃圾信息（广告、电话号码、网址等）
   - 文本清洗（去除@提及、话题标签、URL等）

4. **文本处理**
   - 分词处理（jieba分词，启用HMM模型）
   - 过滤停用词
   - 合并连续数字和英文

5. **结构化处理**
   - 时间格式化
   - 构建评论树结构

6. **结果输出**
   - 按视频ID分组保存为JSON文件
   - 文件名格式：`{video_id}_{timestamp}.json`

### 3.2 工作流程图

```
数据加载 → 短内容过滤 → 垃圾信息过滤 → 文本清洗 → 分词处理 → 时间格式化 → 构建评论树 → 保存JSON
```

## 4. JSON输出结构

### 4.1 视频评论树结构

```json
{
  "video_id": "视频ID字符串",
  "total_comments": 评论总数,
  "root_comments": 根评论数量,
  "create_time": "处理时间",
  "metadata": {
    "processing_time": "处理时间",
    "data_source": "douyin_aweme_comment"
  },
  "root": [
    // 根评论对象数组
  ]
}
```

### 4.2 评论对象结构

```json
{
  "comment_id": "评论ID",
  "parent_comment_id": "父评论ID", // 根评论为""或其他空值表示
  "content": "清洗后的内容",
  "datetime": "2025-10-19 21:51:14",
  "like_count": 点赞数,
  "user_id": "用户ID",
  "nickname": "用户昵称",
  "ip_location": "IP位置",
  "tokens": ["分词1", "分词2", ...],
  "sub_comment_count": 子评论数, // 仅父评论有此字段
  "children": [
    // 子评论对象数组（递归结构）
  ]
}
```

## 5. 配置参数

### 5.1 默认配置

```python
config = {
    'text': {
        'min_length': 5,  # 最小文本长度
        'user_dict_path': 'resources/user_dict.txt',
        'stopwords_path': 'resources/stopwords.txt'
    },
    'filter': {
        'enabled': True,
        'strict_mode': False
    }
}
```

### 5.2 命令行参数

- `--min-length`: 最小内容长度（覆盖配置文件）
- `--output-dir`: 自定义输出目录
- `--stopwords-file`: 自定义停用词文件路径
- `--limit`: 限制处理的评论数量

## 6. 算法特点与优化

### 6.1 分词优化

1. **HMM模型启用**：提高生僻词和未登录词的识别率
2. **单字符处理**：智能过滤无意义单字符，保留有意义的数字和英文
3. **连续数字英文合并**：提高语义完整性

### 6.2 垃圾信息过滤增强

1. **多模式检测**：关键词匹配 + 模式识别
2. **扩展广告关键词**：包含常见广告词汇、联系方式格式、引流词汇
3. **重复字符检测**：识别垃圾内容特征
4. **Emoji数量检测**：防止表情滥用

### 6.3 评论树构建优化

1. **ID格式规范化**：确保ID类型一致性，提高树构建准确性
2. **孤立评论处理**：智能处理父评论不存在的情况
3. **子评论计数**：自动计算并更新父评论的子评论数量

## 7. 性能与效率

- **批处理机制**：按视频ID分组处理，减少内存占用
- **错误处理**：全面的异常捕获和错误恢复机制
- **资源管理**：合理的文件操作和数据库连接管理

## 8. 输入输出示例

### 8.1 输入示例（原始评论数据）

```python
# 从数据库读取的评论数据示例
comment_data = {
    'comment_id': '123456789',
    'parent_comment_id': '',
    'content': '@用户123 这部电影真的很好看！#推荐# https://example.com',
    'create_time': '1633046400',
    'like_count': 100,
    'user_id': 'user123',
    'nickname': '测试用户',
    'ip_location': '北京市',
    'aweme_id': '1234567890123456789'
}
```

### 8.2 输出示例（处理后的JSON结构）

```json
{
  "video_id": "1234567890123456789",
  "total_comments": 1,
  "root_comments": 1,
  "create_time": "2025-10-19 21:51:14",
  "metadata": {
    "processing_time": "2025-10-19 21:51:14",
    "data_source": "douyin_aweme_comment"
  },
  "root": [
    {
      "comment_id": "123456789",
      "parent_comment_id": "",
      "content": "这部电影真的很好看！",
      "datetime": "2025-10-19 21:51:14",
      "like_count": 100,
      "user_id": "user123",
      "nickname": "测试用户",
      "ip_location": "北京市",
      "tokens": ["这部", "电影", "真的", "很好", "看"],
      "children": []
    }
  ]
}
```

## 9. 总结

`data_preprocessing.py` 模块实现了一套完整的中文评论数据预处理流水线，通过多步数据清洗、智能分词和结构化处理，将原始评论数据转换为适合后续文本分析的高质量数据格式。该模块在处理效率、准确性和鲁棒性方面进行了多项优化，特别是在分词处理和垃圾信息过滤方面采用了先进的算法和技术，确保了数据预处理的质量和效率。