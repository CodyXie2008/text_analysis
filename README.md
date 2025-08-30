# 文本分析模块

## 📋 项目概述

文本分析模块是一个功能完整的抖音评论分析工具，支持数据清洗、情感分析、相似度分析、时间分析和点赞分析等功能。该模块采用统一的API调用规范，支持本地处理和阿里云API两种模式。

## 🏗️ 项目结构

```
text_analysis/
├── core/                           # 核心功能模块
│   ├── __init__.py                 # 核心模块初始化
│   ├── base_analyzer.py            # 基础分析器
│   ├── data_paths.py               # 路径管理
│   ├── aliyun_api_manager.py      # 阿里云API统一管理器
│   ├── env.example                 # 环境变量示例
│   └── .env                        # API配置文件
├── modules/                        # 分析功能模块
│   ├── __init__.py                 # 模块初始化
│   ├── data_cleaning_optimized.py  # 数据清洗模块
│   ├── sentiment_analyzer_optimized.py  # 情感分析模块
│   ├── similarity_analysis_optimized.py # 相似度分析模块
│   ├── time_analysis_optimized.py  # 时间分析模块
│   ├── like_analysis_optimized.py  # 点赞分析模块
│   └── hit_stopwords.txt           # 停用词文件
├── algorithms/                     # 统计算法模块
│   ├── __init__.py                 # 算法模块初始化
│   ├── normalization.py            # 归一化算法
│   ├── test_normalization.py       # 归一化测试脚本
│   └── README_normalization.md     # 归一化使用文档
├── docs/                           # 文档目录
├── text_analysis_unified.py        # 统一入口点
├── utils.py                        # 工具函数
└── README.md                       # 项目说明文档
```

## 🚀 核心功能

### 1. 数据清洗模块 (`data_cleaning_optimized.py`)
- **功能**: 清洗和预处理抖音评论数据
- **分词模式**: 支持本地分词(jieba)和阿里云API分词
- **清洗规则**: 垃圾评论过滤、文本清洗、停用词过滤
- **输出格式**: JSON、CSV、可视化图表

### 2. 情感分析模块 (`sentiment_analyzer_optimized.py`)
- **功能**: 分析评论情感倾向
- **分析模式**: 本地词典分析和阿里云API分析
- **情感分类**: 正面、负面、中性
- **并发处理**: 支持批量并发分析

### 3. 相似度分析模块 (`similarity_analysis_optimized.py`)
- **功能**: 分析评论间的相似度，识别模仿性评论
- **技术**: 基于阿里云文本向量API
- **应用**: 从众心理分析、内容重复检测
- **输出**: 相似度矩阵、模仿性评论列表

### 4. 时间分析模块 (`time_analysis_optimized.py`)
- **功能**: 分析评论时间分布和集中度
- **指标**: 时间集中度、评论时间间隔
- **可视化**: 时间分布图表

### 5. 点赞分析模块 (`like_analysis_optimized.py`)
- **功能**: 分析点赞互动模式
- **指标**: 点赞分布、互动热度
- **应用**: 内容质量评估

## 🔧 统一API管理器

### 阿里云API统一管理器 (`core/aliyun_api_manager.py`)

**支持的API**:
- 中文分词（基础版）- `GetWsChGeneral`
- 情感分析（基础版）- `GetSaChGeneral`
- 文本向量（基础版）- `GetWeChGeneral`

**环境变量配置**:
```bash
# 阿里云API配置
NLP_AK_ENV=your_access_key_id_here
NLP_SK_ENV=your_access_key_secret_here
NLP_REGION_ENV=cn-hangzhou
```

**主要功能**:
- 统一的API调用接口
- 自动环境变量加载
- 错误处理和重试机制
- 请求日志记录
- 批量处理支持

## 📖 使用指南

### 1. 环境配置

```bash
# 复制环境变量示例文件
cp text_analysis/core/.env.example text_analysis/core/.env

# 编辑配置文件，填入阿里云API密钥
NLP_AK_ENV=your_access_key_id_here
NLP_SK_ENV=your_access_key_secret_here
NLP_REGION_ENV=cn-hangzhou
```

### 2. 数据清洗

```bash
# 本地分词模式（默认）
python text_analysis/modules/data_cleaning_optimized.py --video-id 7306437681045654834 --limit 100

# API分词模式
python text_analysis/modules/data_cleaning_optimized.py --video-id 7306437681045654834 --limit 100 --segment-mode api
```

### 3. 情感分析

```bash
# 本地词典分析
python text_analysis/modules/sentiment_analyzer_optimized.py --video-id 7306437681045654834 --limit 100 --type local

# 阿里云API分析
python text_analysis/modules/sentiment_analyzer_optimized.py --video-id 7306437681045654834 --limit 100 --type aliyun
```

### 4. 相似度分析

```bash
python text_analysis/modules/similarity_analysis_optimized.py --video-id 7306437681045654834 --limit 100
```

### 5. 统一入口

```bash
# 使用统一入口点
python text_analysis/text_analysis_unified.py cleaning --video-id 7306437681045654834 --limit 100
python text_analysis/text_analysis_unified.py sentiment --video-id 7306437681045654834 --limit 100 --type aliyun
python text_analysis/text_analysis_unified.py similarity --video-id 7306437681045654834 --limit 100
```

## 🔍 API调用测试

### 测试API可用性
```python
from text_analysis.core.aliyun_api_manager import is_aliyun_api_available
print('API可用性:', is_aliyun_api_available())
```

### 测试分词功能
```python
from text_analysis.core.aliyun_api_manager import get_aliyun_api_manager
manager = get_aliyun_api_manager()
words = manager.segment_text("这是一个测试文本")
print('分词结果:', words)
```

### 测试情感分析
```python
sentiment = manager.analyze_sentiment("这个视频很棒！")
print('情感分析:', sentiment)
```

### 测试文本向量
```python
vector = manager.get_text_vector("这是一个测试文本")
print('向量维度:', len(vector))
```

## 📊 输出结果

所有分析模块都会生成以下输出：

1. **数据文件**: JSON、CSV格式的分析结果
2. **分析报告**: 详细的统计报告
3. **可视化图表**: 图表和可视化结果
4. **日志记录**: 详细的执行日志

输出目录结构：
```
data/
├── processed/      # 清洗后的数据
├── results/        # 分析结果
├── reports/        # 分析报告
└── visualizations/ # 可视化图表
```

## 🛠️ 技术特点

### 代码优化
- ✅ **统一API调用**: 所有模块使用统一的API管理器
- ✅ **代码去冗余**: 删除了重复的比较和可视化工具
- ✅ **模块化设计**: 清晰的模块分离和职责划分
- ✅ **错误处理**: 完善的异常处理和回退机制

### 性能优化
- ✅ **并发处理**: 支持批量并发API调用
- ✅ **缓存机制**: 避免重复API调用
- ✅ **限流控制**: 防止API调用频率过高
- ✅ **资源管理**: 自动释放数据库连接

### 用户体验
- ✅ **灵活配置**: 支持多种环境变量配置方式
- ✅ **详细日志**: 完整的执行日志和错误信息
- ✅ **进度显示**: 实时显示处理进度
- ✅ **结果可视化**: 丰富的图表和可视化输出

## 🔧 开发指南

### 添加新的分析模块

1. 继承 `BaseAnalyzer` 类
2. 实现 `analyze()` 方法
3. 在 `modules/__init__.py` 中注册模块
4. 更新统一入口点

### 扩展API功能

1. 在 `AliyunAPIManager` 中添加新的API方法
2. 更新环境变量配置
3. 添加相应的错误处理
4. 更新文档说明

## 📝 更新日志

### v2.0.0 (2025-08-30)
- ✅ 重构API调用架构，统一使用 `AliyunAPIManager`
- ✅ 优化代码结构，删除冗余文件
- ✅ 改进环境变量配置管理
- ✅ 增强错误处理和日志记录
- ✅ 完善文档和使用指南

### v1.0.0
- 初始版本发布
- 基础分析功能实现

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## �� 许可证

本项目采用MIT许可证。
