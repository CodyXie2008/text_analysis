# 文本分析模块

## 📋 项目概述

文本分析模块是一个功能完整的抖音评论从众心理分析工具，支持数据清洗、时间从众心理分析、情感从众心理分析、相似度从众心理分析和点赞从众心理分析等功能。该模块采用统一的API调用规范，支持本地处理和阿里云API两种模式。

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
│   ├── conformity_time_analyzer.py # 时间从众心理分析模块
│   ├── sentiment_conformity_analyzer.py # 情感从众心理分析模块
│   ├── similarity_conformity_analyzer.py # 相似度从众心理分析模块
│   ├── like_conformity_analyzer.py # 点赞从众心理分析模块
│   └── hit_stopwords.txt           # 停用词文件
├── algorithms/                     # 统计算法模块
│   ├── __init__.py                 # 算法模块初始化
│   ├── normalization.py            # 归一化算法
│   ├── test_normalization.py       # 归一化测试脚本
│   └── README_normalization.md     # 归一化使用文档
├── docs/                           # 详细算法文档
│   ├── README.md                   # 算法文档索引
│   ├── 数据清洗算法详解.md          # 数据清洗算法详解
│   ├── 从众心理时间分析算法详解.md   # 时间从众心理分析算法详解
│   ├── 情感从众心理分析算法详解.md   # 情感从众心理分析算法详解
│   ├── 相似度从众心理分析算法详解.md # 相似度从众心理分析算法详解
│   ├── 点赞从众心理分析算法详解.md   # 点赞从众心理分析算法详解
│   ├── 文档结构说明.md              # 文档结构说明
│   └── 集成工作流总结报告.md         # 集成工作流总结报告
├── text_analysis_unified.py        # 统一入口点
├── utils.py                        # 工具函数
└── README.md                       # 项目说明文档
```

## 🚀 核心功能

### 1. 数据清洗模块 (`data_cleaning_optimized.py`)
- **功能**: 清洗和预处理抖音评论数据，集成时间标准化
- **特点**: 支持本地/API双模式分词，自动父子评论识别
- **输出**: 标准化JSON数据，支持后续分析模块直接使用

### 2. 从众心理分析模块
- **时间从众心理分析** (`conformity_time_analyzer.py`): 分析评论时间从众心理特征
- **情感从众心理分析** (`sentiment_conformity_analyzer.py`): 分析情感从众心理特征
- **相似度从众心理分析** (`similarity_conformity_analyzer.py`): 分析文本相似度从众心理特征
- **点赞从众心理分析** (`like_conformity_analyzer.py`): 分析点赞从众心理特征

### 3. 综合分析模块 (`text_analysis_unified.py`)
- **功能**: 统一入口点，支持所有分析模块的集成调用
- **特点**: 一键运行完整的从众心理综合分析
- **输出**: 生成综合从众心理分数和详细分析报告

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

## 📖 快速开始

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

### 3. 从众心理分析

#### 综合分析（推荐）
```bash
# 一键运行完整的从众心理综合分析
python text_analysis_unified.py conformity --use-cleaned-data --video-id 7306437681045654834

# 测试模式（少量数据）
python text_analysis_unified.py conformity --use-cleaned-data --test

# 不保存结果文件
python text_analysis_unified.py conformity --use-cleaned-data --no-save
```

#### 单独模块分析
```bash
# 时间从众心理分析
python text_analysis_unified.py time --use-cleaned-data --video-id 7306437681045654834

# 情感从众心理分析
python text_analysis_unified.py sentiment --use-cleaned-data --video-id 7306437681045654834

# 相似度从众心理分析
python text_analysis_unified.py similarity --use-cleaned-data --video-id 7306437681045654834

# 点赞从众心理分析
python text_analysis_unified.py like --use-cleaned-data --video-id 7306437681045654834
```

## 📚 详细文档

详细的算法原理、使用方法和API文档请参考 [docs/README.md](./docs/README.md)

## 🛠️ 技术特点

### 代码优化
- ✅ **统一API调用**: 所有模块使用统一的API管理器
- ✅ **模块化设计**: 清晰的模块分离和职责划分
- ✅ **错误处理**: 完善的异常处理和回退机制

### 性能优化
- ✅ **并发处理**: 支持批量并发API调用
- ✅ **缓存机制**: 避免重复API调用
- ✅ **限流控制**: 防止API调用频率过高

### 用户体验
- ✅ **灵活配置**: 支持多种环境变量配置方式
- ✅ **详细日志**: 完整的执行日志和错误信息
- ✅ **进度显示**: 实时显示处理进度

## 📝 更新日志

### v5.0.0 (2025-09-07)
- ✅ **综合分析功能**: 新增统一入口点，支持一键运行完整的从众心理综合分析
- ✅ **JSON序列化优化**: 修复NumPy类型序列化问题，确保结果文件正确保存
- ✅ **错误处理完善**: 增强错误处理和类型转换机制
- ✅ **用户体验优化**: 改进命令行界面和进度显示

### v4.0.0 (2025-09-07)
- ✅ **模块重构**: 删除老的分析模块，专注于从众心理分析
- ✅ **文档优化**: 分离总览和详细文档，提高可维护性
- ✅ **项目结构**: 优化项目结构，突出从众心理分析功能

### v3.6.0 (2025-09-07)
- ✅ **相似度从众心理分析**: 完成相似度从众心理分析模块
- ✅ **算法验证**: 完成算法准确性验证和文档更新
- ✅ **文件清理**: 清理不需要的测试文件

### v3.0.0 (2025-09-05)
- ✅ **集成工作流**: 数据清洗模块集成时间标准化功能
- ✅ **从众心理分析**: 新增从众心理时间分析模块
- ✅ **算法优化**: 自适应衰减因子算法，提高分析准确性

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证。