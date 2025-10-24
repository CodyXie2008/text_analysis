# 文本分析算法文档

## 📋 文档概述

本文档集合包含了文本分析模块中所有核心算法的详细说明，包括算法原理、实现细节、使用方法和优化建议。

## 📚 算法文档索引

### 核心算法文档

#### 1. 数据预处理算法
- **文档**: [数据预处理1019.md](./数据预处理1019.md) 或 [data_preprocessing_tech_doc.md](./data_preprocessing_tech_doc.md)
- **模块**: `data_preprocessing.py`
- **功能**: 数据加载、文本清洗、分词处理、时间格式化、评论树构建
- **特点**: 支持配置化参数、自动字段映射、健壮的错误处理

#### 2. 从众行为分析算法
- **文档**: [group_conformity_algorithm_tech_doc.md](./group_conformity_algorithm_tech_doc.md)
- **模块**: `group_conformity_analyzer.py`
- **功能**: 分析评论集中的从众心理特征
- **算法**: 语义趋同、情绪一致性、点赞集中度计算
- **应用**: 社交媒体行为分析、群体心理研究

#### 3. 网络暴力分析算法
- **文档**: [group_violence_algorithm_tech_doc.md](./group_violence_algorithm_tech_doc.md)
- **模块**: `group_violence_analyzer.py`
- **功能**: 检测评论中的暴力词汇，计算群体暴力强度
- **算法**: 词汇匹配、加权评分、阈值判断
- **应用**: 网络暴力监测、内容安全评估

### 支持文档

#### 4. 建模思路文档
- **文档**: [建模思路新重构1018.md](./建模思路新重构1018.md)
- **内容**: 项目建模思路、研究框架、变量定义

#### 5. API接口文档
- **文档**: [API接口文档.md](./API接口文档.md)
- **内容**: 完整的API接口规范、参数说明、返回格式


## 🏗️ 文档结构

```
docs/
├── README.md                           # 本文档 - 算法文档索引
├── API接口文档.md                       # 完整的API接口规范
├── data_preprocessing_tech_doc.md      # 数据预处理算法技术文档
├── 数据预处理1019.md                    # 数据预处理详细说明
├── group_violence_algorithm_tech_doc.md # 网络暴力分析算法技术文档
├── group_conformity_algorithm_tech_doc.md # 从众行为算法技术文档
└── 建模思路新重构1018.md                # 建模思路重构文档
```

## 🔍 按功能选择文档

### 数据预处理
- [数据预处理1019.md](./数据预处理1019.md) - 数据预处理详细说明
- [data_preprocessing_tech_doc.md](./data_preprocessing_tech_doc.md) - 数据预处理算法技术文档

### 核心算法分析
- [group_conformity_algorithm_tech_doc.md](./group_conformity_algorithm_tech_doc.md) - 从众行为分析算法
- [group_violence_algorithm_tech_doc.md](./group_violence_algorithm_tech_doc.md) - 网络暴力分析算法

### 项目理解
- [API接口文档.md](./API接口文档.md) - 完整的API接口规范
- [建模思路新重构1018.md](./建模思路新重构1018.md) - 项目建模思路重构文档

## 🚀 快速导航

### 新手入门
1. 阅读 [API接口文档.md](./API接口文档.md) 了解接口规范
2. 阅读 [数据预处理1019.md](./数据预处理1019.md) 了解数据预处理流程
3. 阅读 [group_conformity_algorithm_tech_doc.md](./group_conformity_algorithm_tech_doc.md) 了解从众行为分析
4. 阅读 [group_violence_algorithm_tech_doc.md](./group_violence_algorithm_tech_doc.md) 了解网络暴力分析

### 算法开发者
1. 查看各算法详解文档了解实现细节
2. 参考 [API接口文档.md](./API接口文档.md) 了解接口规范
3. 根据具体需求选择合适的算法模块

### 研究人员
1. 阅读 [建模思路新重构1018.md](./建模思路新重构1018.md) 了解研究框架
2. 重点关注从众行为和网络暴力分析算法文档
3. 参考算法原理和实现细节进行深入研究

## 📊 算法对比

| 算法模块 | 分析维度 | 主要指标 | 应用场景 |
|---------|---------|---------|---------|
| 数据预处理 | 数据 | 清洗率、结构化程度 | 数据准备与标准化 |
| 从众行为分析 | 行为 | 语义趋同、情绪一致性、点赞集中度 | 群体心理研究 |
| 网络暴力分析 | 内容 | 暴力词汇匹配、强度评分 | 网络暴力监测 |

## 🔧 技术栈

### 核心依赖
- **Python 3.8+**: 主要开发语言
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **scikit-learn**: 机器学习算法
- **jieba**: 中文分词
- **requests**: HTTP请求处理

### 外部API
- **阿里云NLP API**: 中文分词、情感分析、文本向量化
- **HMAC-SHA1**: API签名算法

### 算法库
- **MinMaxNormalizer**: 自定义归一化算法
- **StandardScaler**: 标准化算法
- **PCA**: 主成分分析

## 📝 版本信息

- **v7.0.0** (2025-10-24): 更新文档结构，移除已删除模块引用，优化算法分类
- **v6.0.0** (2025-10-19): 更新文档结构，新增网络暴力分析算法和数据预处理文档

---

*核心算法文档 v6.0.0 - 2025-10-19*