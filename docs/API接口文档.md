# API接口文档

## 📋 概述

本文档详细说明了文本分析模块的API接口，包括统一入口点、各个分析模块的接口规范，以及返回数据格式。

## 🏗️ 统一入口点

### 模块: `text_analysis_unified.py`

#### 命令行接口
```bash
python text_analysis_unified.py <command> [options]
```

#### 支持的命令
- `conformity` - 综合分析
- `time` - 时间从众心理分析
- `sentiment` - 情感从众心理分析
- `similarity` - 相似度从众心理分析
- `like` - 点赞从众心理分析
- `cleaning` - 数据清洗

## 🔧 综合分析接口

### 命令: `conformity`

#### 功能描述
运行完整的从众心理综合分析，包括时间、情感、相似度、点赞四个维度的分析。

#### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--use-cleaned-data` | flag | 否 | False | 使用清洗后的数据文件 |
| `--video-id` | str | 否 | None | 视频ID，如果不指定则分析所有数据 |
| `--limit` | int | 否 | None | 限制分析数量 |
| `--cleaned-data-path` | str | 否 | None | 清洗数据文件路径 |
| `--test` | flag | 否 | False | 测试模式，只分析少量数据 |
| `--no-save` | flag | 否 | False | 不保存结果文件 |
| `--no-report` | flag | 否 | False | 不生成分析报告 |
| `--no-viz` | flag | 否 | False | 不创建可视化图表 |

#### 使用示例
```bash
# 基本用法
python text_analysis_unified.py conformity --use-cleaned-data --video-id 7306437681045654834

# 测试模式
python text_analysis_unified.py conformity --use-cleaned-data --test

# 指定数据文件
python text_analysis_unified.py conformity --cleaned-data-path data/cleaned/parent_7306470754056569635_cleaned.json
```

#### 返回数据格式
```json
{
  "analysis_metadata": {
    "analysis_time": "2025-09-07 18:15:17",
    "analyzer_version": "v1.0.0",
    "analysis_type": "comprehensive_conformity_analysis",
    "total_comments": 337,
    "parent_comment_id": "7306470754056569635"
  },
  "parent_comment_info": {
    "comment_id": "7306470754056569635",
    "content": "261，我的！seey是我发的！！！"
  },
  "comprehensive_conformity_analysis": {
    "overall_conformity_score": 0.3903,
    "conformity_level": "低从众心理",
    "score_breakdown": {
      "time_conformity_score": 0.2358,
      "sentiment_conformity_score": 0.7501,
      "similarity_conformity_score": 0.5719,
      "like_conformity_score": 0.0034
    },
    "analysis_summary": {
      "time_conformity": {
        "score": 0.2358,
        "description": "时间维度从众心理分析"
      },
      "sentiment_conformity": {
        "score": 0.7501,
        "description": "情感维度从众心理分析"
      },
      "similarity_conformity": {
        "score": 0.5719,
        "description": "文本相似度从众心理分析"
      },
      "like_conformity": {
        "score": 0.0034,
        "description": "点赞行为从众心理分析"
      }
    }
  },
  "detailed_results": {
    "time_conformity_analysis": {...},
    "sentiment_conformity_analysis": {...},
    "similarity_conformity_analysis": {...},
    "like_conformity_analysis": {...}
  }
}
```

## ⏰ 时间从众心理分析接口

### 命令: `time`

#### 功能描述
分析评论时间从众心理特征，包括时间间隔、响应速度、时间从众模式等。

#### 参数说明
与综合分析接口相同，参考上表。

#### 返回数据格式
```json
{
  "analysis_metadata": {
    "analysis_time": "2025-09-07 18:12:40",
    "analyzer_version": "v1.0.0",
    "data_source": "unknown",
    "total_comments": 337
  },
  "parent_comment_info": {
    "comment_id": "7306470754056569635",
    "content": "261，我的！seey是我发的！！！"
  },
  "parent_environment_time_analysis": {
    "parent_comment_id": "7306470754056569635",
    "child_comment_count": 336,
    "parent_conformity_score": 0.2358,
    "statistics": {
      "mean_score": 0.2358,
      "median_score": 0.0683,
      "max_score": 0.9722,
      "min_score": 0.0,
      "std_score": 0.3020
    },
    "conformity_distribution": {
      "high_conformity_count": 43,
      "high_conformity_ratio": 0.1280,
      "early_response_count": 5,
      "early_response_ratio": 0.0149,
      "quick_response_count": 0,
      "quick_response_ratio": 0.0
    },
    "time_analysis": {
      "avg_time_diff": 2245274.5,
      "median_time_diff": 88571.0,
      "min_time_diff": 929.0,
      "max_time_diff": 53323883.0
    }
  },
  "time_classification": {},
  "top_high_conformity_comments": [...]
}
```

## 😊 情感从众心理分析接口

### 命令: `sentiment`

#### 功能描述
分析情感从众心理特征，包括情感一致性、情感传播、情感从众程度等。

#### 返回数据格式
```json
{
  "analysis_metadata": {
    "analysis_time": "2025-09-07 18:14:12",
    "analyzer_version": "v1.0.0",
    "data_source": "unknown",
    "total_comments": 337
  },
  "parent_comment_info": {
    "comment_id": "7306470754056569635",
    "content": "261，我的！seey是我发的！！！"
  },
  "parent_environment_sentiment_analysis": {
    "parent_comment_id": "7306470754056569635",
    "parent_sentiment_score": 0.9926,
    "parent_sentiment_type": "正面",
    "child_comment_count": 336,
    "parent_sentiment_conformity_score": 0.7501,
    "statistics": {
      "mean_score": 0.7501,
      "median_score": 0.9513,
      "max_score": 1.0,
      "min_score": 0.0,
      "std_score": 0.3355
    },
    "conformity_distribution": {
      "high_conformity_count": 234,
      "high_conformity_ratio": 0.6964,
      "sentiment_category_distribution": {
        "高度情感从众": 184,
        "非情感从众": 80,
        "中度情感从众": 42,
        "轻度情感从众": 18,
        "低度情感从众": 12
      }
    },
    "sentiment_analysis": {
      "avg_sentiment_difference": 0.4177,
      "median_sentiment_difference": 0.0549,
      "min_sentiment_difference": 0.0001,
      "max_sentiment_difference": 1.9862
    }
  },
  "sentiment_classification": {...},
  "top_high_sentiment_comments": [...]
}
```

## 📝 相似度从众心理分析接口

### 命令: `similarity`

#### 功能描述
分析文本相似度从众心理特征，包括文本相似度、内容模仿、表达方式从众等。

#### 返回数据格式
```json
{
  "analysis_metadata": {
    "analysis_time": "2025-09-07 18:15:17",
    "analyzer_version": "v1.0.0",
    "data_source": "unknown",
    "total_comments": 337
  },
  "parent_comment_info": {
    "comment_id": "7306470754056569635",
    "content": "261，我的！seey是我发的！！！"
  },
  "parent_environment_similarity_analysis": {
    "parent_comment_id": "7306470754056569635",
    "parent_similarity_conformity_score": 0.5719,
    "child_comment_count": 336,
    "statistics": {
      "mean_score": 0.5719,
      "median_score": 0.5929,
      "std_score": 0.2506,
      "min_score": 0.0,
      "max_score": 1.0
    },
    "similarity_distribution": {
      "high_similarity_count": 119,
      "high_similarity_ratio": 0.3542,
      "medium_similarity_count": 163,
      "low_similarity_count": 54
    },
    "similarity_analysis": {
      "mean_similarity": 0.5719,
      "median_similarity": 0.5929,
      "std_similarity": 0.2506,
      "min_similarity": 0.0,
      "max_similarity": 1.0
    }
  },
  "similarity_classification": {...},
  "top_high_similarity_comments": [...]
}
```

## 👍 点赞从众心理分析接口

### 命令: `like`

#### 功能描述
分析点赞从众心理特征，包括点赞行为、互动从众、社交影响力等。

#### 返回数据格式
```json
{
  "analysis_metadata": {
    "analysis_time": "2025-09-07 18:15:17",
    "analyzer_version": "v1.0.0",
    "data_source": "unknown",
    "total_comments": 337
  },
  "parent_comment_info": {
    "comment_id": "7306470754056569635",
    "content": "261，我的！seey是我发的！！！"
  },
  "parent_environment_like_analysis": {
    "parent_comment_id": "7306470754056569635",
    "parent_like_count": 149991,
    "child_comment_count": 336,
    "parent_like_conformity_score": 0.0034,
    "statistics": {
      "mean_score": 0.0034,
      "median_score": 0.0,
      "max_score": 1.0,
      "min_score": 0.0,
      "std_score": 0.0548
    },
    "conformity_distribution": {
      "high_conformity_count": 1,
      "high_conformity_ratio": 0.0030,
      "like_category_distribution": {
        "非从众": 336
      }
    },
    "like_analysis": {
      "avg_like_difference": 149875.2,
      "median_like_difference": 149991.0,
      "min_like_difference": 115857,
      "max_like_difference": 149991
    }
  },
  "like_classification": {},
  "top_high_like_comments": [...]
}
```

## 🧹 数据清洗接口

### 命令: `cleaning`

#### 功能描述
清洗和预处理抖音评论数据，集成时间标准化功能。

#### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--video-id` | str | 是 | None | 视频ID |
| `--limit` | int | 否 | None | 限制处理数量 |
| `--segment-mode` | str | 否 | local | 分词模式：local/api |
| `--output-dir` | str | 否 | data/cleaned | 输出目录 |
| `--test` | flag | 否 | False | 测试模式 |

#### 使用示例
```bash
# 基本用法
python text_analysis_unified.py cleaning --video-id 7306437681045654834 --limit 100

# API分词模式
python text_analysis_unified.py cleaning --video-id 7306437681045654834 --segment-mode api

# 测试模式
python text_analysis_unified.py cleaning --video-id 7306437681045654834 --test
```

#### 返回数据格式
```json
{
  "cleaning_metadata": {
    "cleaning_time": "2025-09-07 18:10:00",
    "video_id": "7306437681045654834",
    "total_comments": 337,
    "processed_comments": 337,
    "segment_mode": "local"
  },
  "cleaning_statistics": {
    "parent_comments": 1,
    "child_comments": 336,
    "cleaned_comments": 337,
    "filtered_comments": 0
  },
  "comments": [
    {
      "comment_id": "7306470754056569635",
      "aweme_id": "7306437681045654834",
      "parent_comment_id": "0",
      "content": "261，我的！seey是我发的！！！",
      "create_time": 1701170293,
      "like_count": 149991,
      "sub_comment_count": 1025,
      "user_id": "110493671510",
      "nickname": "FXXXXX",
      "words": ["261", "seey"],
      "word_count": 2,
      "comment_time": "2023-11-28 11:18:13",
      "is_parent": true
    }
  ]
}
```

## 🔧 程序化接口

### Python API

#### 导入模块
```python
from text_analysis_unified import (
    run_conformity_analysis,
    run_time_analysis,
    run_sentiment_analysis,
    run_similarity_analysis,
    run_like_analysis,
    run_cleaning_analysis
)
```

#### 综合分析
```python
import argparse

# 创建参数对象
args = argparse.Namespace()
args.use_cleaned_data = True
args.video_id = "7306437681045654834"
args.test = False
args.no_save = False
args.no_report = False

# 运行综合分析
result = run_conformity_analysis(args)
```

#### 单独模块分析
```python
# 时间分析
time_result = run_time_analysis(args)

# 情感分析
sentiment_result = run_sentiment_analysis(args)

# 相似度分析
similarity_result = run_similarity_analysis(args)

# 点赞分析
like_result = run_like_analysis(args)
```

## 📊 错误处理

### 常见错误码

| 错误码 | 错误信息 | 解决方案 |
|--------|----------|----------|
| `API_TIMEOUT` | API调用超时 | 检查网络连接，重试请求 |
| `API_QUOTA_EXCEEDED` | API配额超限 | 检查API配额，等待重置 |
| `DATA_NOT_FOUND` | 数据文件不存在 | 先运行数据清洗 |
| `INVALID_FORMAT` | 数据格式错误 | 检查输入数据格式 |
| `MEMORY_ERROR` | 内存不足 | 减少数据量或增加内存 |

### 错误响应格式
```json
{
  "error": {
    "code": "API_TIMEOUT",
    "message": "API调用超时",
    "details": "HTTPSConnectionPool(host='alinlp.cn-hangzhou.aliyuncs.com', port=443): Read timed out",
    "timestamp": "2025-09-07 18:15:17"
  }
}
```

## 🔒 安全考虑

### API密钥管理
- 使用环境变量存储API密钥
- 不要在代码中硬编码密钥
- 定期轮换API密钥

### 数据隐私
- 本地处理敏感数据
- 避免在日志中记录敏感信息
- 遵守数据保护法规

### 访问控制
- 限制API调用频率
- 实施适当的访问控制
- 监控异常使用模式

## 📝 版本信息

### API版本
- **当前版本**: v5.0.0
- **发布日期**: 2025-09-07
- **兼容性**: 向后兼容v4.0.0

### 更新日志
- **v5.0.0**: 新增综合分析接口，优化错误处理
- **v4.0.0**: 统一接口规范，标准化返回格式
- **v3.0.0**: 初始版本，基础分析接口

---

*API接口文档 v5.0.0 - 2025-09-07*
