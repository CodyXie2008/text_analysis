# 社交媒体从众心理与网络暴力研究项目

## 📋 项目概述

本项目是一个专注于研究社交媒体从众心理对网络暴力影响的分析工具集。通过对抖音评论数据的深入挖掘，探索评论集中的从众行为模式如何影响网络暴力的形成与扩散。项目支持数据预处理、从众行为分析和网络暴力分析三大核心功能，构建了一个完整的研究框架，为理解社交媒体负面行为的形成机制提供数据支持和理论依据。

## 🏗️ 项目结构

```
text_analysis/
├── core/                           # 核心功能模块
│   ├── __init__.py                 # 核心模块初始化
│   ├── base_analyzer.py            # 基础分析器
│   ├── data_paths.py               # 路径管理
│   ├── aliyun_api_manager.py       # 阿里云API统一管理器
│   ├── db_config.py                # 数据库配置
│   ├── env.example                 # 环境变量示例
│   └── .env                        # API配置文件
├── modules/                        # 分析功能模块
│   ├── __init__.py                 # 模块初始化
│   ├── data_preprocessing.py       # 数据预处理模块
│   ├── group_conformity_analyzer.py # 从众行为分析模块
│   ├── group_violence_analyzer.py  # 网络暴力分析模块
│   ├── hit_stopwords.txt           # 停用词文件
│   ├── userdict.txt                # 用户词典
│   └── violence_words.txt          # 暴力词汇库
├── algorithms/                     # 统计算法模块
│   ├── __init__.py                 # 算法模块初始化
│   ├── normalization.py            # 归一化算法
│   └── README_normalization.md     # 归一化使用文档
├── tests/                          # 测试模块
│   ├── __init__.py                 # 测试模块初始化
│   ├── calculate_single_comment.py # 单条评论计算测试
│   ├── debug_aliyun_api.py         # API调试脚本
│   ├── test_group_conformity.py    # 从众分析测试
│   └── test_normalization.py       # 归一化测试脚本
├── docs/                           # 详细算法文档
│   ├── README.md                   # 算法文档索引
│   ├── API接口文档.md               # API接口文档
│   ├── data_preprocessing_tech_doc.md # 数据预处理技术文档
│   ├── group_conformity_algorithm_tech_doc.md # 从众行为算法文档
│   ├── group_violence_algorithm_tech_doc.md # 网络暴力算法文档
│   ├── 建模思路新重构1018.md        # 建模思路重构文档
│   └── 数据预处理1019.md            # 数据预处理详细说明
├── utils.py                        # 工具函数
├── config.yaml                     # 配置文件
└── README.md                       # 项目说明文档
```

## 🚀 核心功能

### 1. 数据预处理模块 (`data_preprocessing.py`)
- **功能**: 对抖音评论数据进行清洗、结构化和预处理，为后续分析提供高质量数据基础
- **特点**: 支持配置化参数、文本清洗、分词处理、时间格式化和评论树构建
- **输出**: 结构化的JSON数据，为后续分析提供标准化输入

### 2. 从众行为分析模块 (`group_conformity_analyzer.py`)
- **功能**: 分析评论集中的从众心理特征，测量群体行为的一致性程度
- **算法**: 计算语义趋同度、情绪一致性和点赞集中度，量化从众行为指标
- **输出**: 从众心理指数和详细分析报告，识别评论中的从众模式

### 3. 网络暴力分析模块 (`group_violence_analyzer.py`)
- **功能**: 检测评论中的暴力词汇，计算群体暴力强度，评估网络环境健康度
- **算法**: 词汇匹配、加权评分和阈值判断，识别不同程度的暴力表达
- **输出**: 暴力强度评分和包含暴力词汇的评论识别，为研究提供暴力行为数据

### 4. 关联性分析框架
- **功能**: 整合从众行为数据和网络暴力数据，探索两者之间的相关性和因果关系
- **方法**: 时间序列分析、相关性检验、回归分析等统计方法
- **输出**: 相关性报告和可视化分析结果，揭示从众心理对网络暴力的影响机制

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
cp core/.env.example core/.env

# 编辑配置文件，填入阿里云API密钥
NLP_AK_ENV=your_access_key_id_here
NLP_SK_ENV=your_access_key_secret_here
NLP_REGION_ENV=cn-hangzhou
```

### 2. 数据预处理

```bash
# 运行数据预处理
python modules/data_preprocessing.py

# 使用特定视频ID
python modules/data_preprocessing.py --video-id 7306437681045654834
```

### 3. 从众行为分析

```bash
# 运行从众行为分析
python modules/group_conformity_analyzer.py
```

### 4. 网络暴力分析

```bash
# 运行网络暴力分析
python modules/group_violence_analyzer.py
```

## 📚 详细文档

详细的算法原理、实现细节和技术文档请参考 [docs/README.md](./docs/README.md)，其中包含了所有核心算法的详细说明。

## 🛠️ 技术特点

### 架构设计
- ✅ **模块化设计**: 清晰的模块分离和职责划分，支持独立功能开发
- ✅ **可扩展结构**: 易于添加新的分析模块和功能，适应不同研究需求
- ✅ **统一接口**: 标准化的数据输入输出格式，确保模块间无缝协作

### 功能优势
- ✅ **配置化参数**: 支持通过配置文件灵活调整分析参数，适应不同研究场景
- ✅ **自动字段映射**: 智能识别和映射数据源字段，提高数据处理效率
- ✅ **健壮的错误处理**: 完善的异常处理和数据验证机制，确保分析结果可靠性
- ✅ **研究导向**: 功能设计围绕研究目标，提供针对性的数据支持和分析结果

### 技术栈
- **Python 3.8+**: 主要开发语言
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **jieba**: 中文分词
- **requests**: HTTP请求处理
- **阿里云NLP API**: 高级文本分析功能
- **统计分析库**: 支持相关性分析、回归分析等研究方法

## 📝 更新日志

### v7.0.0 (2025-10-24)
- ✅ **项目重构**: 简化核心模块，专注于数据预处理、从众行为分析和网络暴力分析
- ✅ **暴力词汇库增强**: 扩充暴力词汇库，提高网络暴力检测准确性
- ✅ **文档更新**: 完善算法文档，提供更详细的技术说明
- ✅ **代码优化**: 改进数据处理流程，增强错误处理机制

### v6.0.0 (2025-10-19)
- ✅ **网络暴力分析**: 新增网络暴力分析模块
- ✅ **数据预处理优化**: 改进数据预处理算法，增强文本清洗能力
- ✅ **文档结构更新**: 重新组织文档结构，提高可读性

### v5.0.0 (2025-09-07)
- ✅ **从众心理分析**: 实现多个维度的从众心理分析功能
- ✅ **统一API管理**: 整合API调用，提供统一的接口规范
- ✅ **模块化设计**: 优化项目结构，实现功能模块解耦

## 📊 研究价值

- **理论意义**: 探索社交媒体环境下从众心理与网络暴力的关系，为社会心理学和传播学研究提供新视角
- **实践价值**: 为社交媒体平台内容监管和用户行为引导提供数据支持和技术手段
- **应用场景**: 可用于学术研究、社交媒体平台治理、网络空间健康度评估等领域

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！特别是在以下方面的贡献：
- 算法优化和创新
- 新增研究维度和分析方法
- 数据集扩展和标注
- 研究结果可视化改进

## 📄 许可证

本项目采用MIT许可证。