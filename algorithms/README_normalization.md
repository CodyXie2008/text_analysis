# 归一化算法使用文档

## 📋 概述

本模块提供了多种数据归一化和标准化算法，用于将不同量纲、不同范围的数值映射到统一的范围，确保在进行综合计算时公平地对待每一个指标。

## 🚀 支持的算法

### 1. Min-Max归一化 (MinMaxNormalizer)
**适用场景**: 数据分布相对均匀，无异常值
**公式**: `X_norm = (X - X_min) / (X_max - X_min) * (max - min) + min`
**特点**: 将数据线性变换到指定范围，保持原始数据分布形状

### 2. Z-Score标准化 (ZScoreNormalizer)
**适用场景**: 数据近似正态分布
**公式**: `Z = (X - μ) / σ`
**特点**: 将数据标准化为均值为0，标准差为1的分布

### 3. 稳健缩放 (RobustScaler)
**适用场景**: 数据包含异常值
**公式**: `X_scaled = (X - Q2) / (Q3 - Q1)`
**特点**: 使用四分位数进行缩放，对异常值不敏感

### 4. 小数缩放 (DecimalScaler)
**适用场景**: 需要控制数值精度
**公式**: `X_scaled = X / scale_factor`
**特点**: 将数据缩放到指定的小数位数

### 5. 对数归一化 (LogNormalizer)
**适用场景**: 右偏分布数据
**公式**: `X_log = log(X + ε)`
**特点**: 对数据进行对数变换，压缩大数值

## 📖 使用方法

### 基本用法

#### 1. 使用类方法（推荐）

```python
import numpy as np
from text_analysis.algorithms.normalization import MinMaxNormalizer

# 创建归一化器
normalizer = MinMaxNormalizer(feature_range=(0, 1))

# 拟合并变换数据
data = np.array([1, 5, 10, 15, 20])
normalized_data = normalizer.fit_transform(data)

# 对新数据进行变换
new_data = np.array([8, 12, 18])
transformed_data = normalizer.transform(new_data)

# 逆变换
original_data = normalizer.inverse_transform(transformed_data)
```

#### 2. 使用便捷函数

```python
from text_analysis.algorithms.normalization import min_max_normalize

# 直接归一化
data = np.array([1, 5, 10, 15, 20])
normalized_data = min_max_normalize(data, feature_range=(0, 1))
```

### 详细示例

#### Min-Max归一化示例

```python
import numpy as np
import pandas as pd
from text_analysis.algorithms.normalization import MinMaxNormalizer

# 示例数据
data = np.array([10, 25, 50, 75, 100])

# 创建归一化器
normalizer = MinMaxNormalizer(feature_range=(0, 1))

# 拟合并变换
normalized = normalizer.fit_transform(data)
print(f"原始数据: {data}")
print(f"归一化结果: {normalized.flatten()}")
# 输出: [0.0, 0.16666667, 0.44444444, 0.72222222, 1.0]

# 自定义范围
normalizer_custom = MinMaxNormalizer(feature_range=(-1, 1))
normalized_custom = normalizer_custom.fit_transform(data)
print(f"自定义范围结果: {normalized_custom.flatten()}")
# 输出: [-1.0, -0.66666667, -0.11111111, 0.44444444, 1.0]
```

#### Z-Score标准化示例

```python
from text_analysis.algorithms.normalization import ZScoreNormalizer

# 示例数据
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 创建标准化器
normalizer = ZScoreNormalizer()
standardized = normalizer.fit_transform(data)

print(f"原始数据: {data}")
print(f"标准化结果: {standardized.flatten()}")
print(f"标准化后均值: {np.mean(standardized):.6f}")  # 接近0
print(f"标准化后标准差: {np.std(standardized):.6f}")  # 接近1
```

#### 稳健缩放示例

```python
from text_analysis.algorithms.normalization import RobustScaler

# 包含异常值的数据
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])

# 创建稳健缩放器
scaler = RobustScaler()
scaled = scaler.fit_transform(data)

print(f"原始数据: {data}")
print(f"稳健缩放结果: {scaled.flatten()}")
print(f"缩放后中位数: {np.median(scaled):.6f}")  # 接近0
print(f"缩放后IQR: {np.percentile(scaled, 75) - np.percentile(scaled, 25):.6f}")  # 接近1
```

#### 对数归一化示例

```python
from text_analysis.algorithms.normalization import LogNormalizer

# 右偏分布数据
data = np.array([1, 2, 5, 10, 50, 100, 500, 1000])

# 创建对数归一化器
normalizer = LogNormalizer(epsilon=1e-8, base=np.e)
log_normalized = normalizer.fit_transform(data)

print(f"原始数据: {data}")
print(f"对数归一化结果: {log_normalized.flatten()}")

# 以10为底的对数
normalizer_10 = LogNormalizer(epsilon=1e-8, base=10)
log_normalized_10 = normalizer_10.fit_transform(data)
print(f"以10为底的对数归一化: {log_normalized_10.flatten()}")
```

### Pandas集成示例

```python
import pandas as pd
from text_analysis.algorithms.normalization import MinMaxNormalizer, ZScoreNormalizer

# 创建DataFrame
df = pd.DataFrame({
    'feature1': [1, 5, 10, 15, 20],
    'feature2': [100, 200, 300, 400, 500],
    'feature3': [0.1, 0.5, 1.0, 1.5, 2.0]
})

print("原始DataFrame:")
print(df)

# Min-Max归一化
normalizer = MinMaxNormalizer(feature_range=(0, 1))
normalized_df = normalizer.fit_transform(df)
print("\nMin-Max归一化后:")
print(pd.DataFrame(normalized_df, columns=df.columns))

# Z-Score标准化
z_normalizer = ZScoreNormalizer()
z_normalized_df = z_normalizer.fit_transform(df)
print("\nZ-Score标准化后:")
print(pd.DataFrame(z_normalized_df, columns=df.columns))
```

## 🔧 便捷函数

所有算法都提供了便捷函数，可以直接调用：

```python
from text_analysis.algorithms.normalization import (
    min_max_normalize,
    z_score_normalize,
    robust_scale,
    decimal_scale,
    log_normalize
)

# 便捷函数使用
data = np.array([1, 5, 10, 15, 20])

# Min-Max归一化
normalized = min_max_normalize(data, feature_range=(0, 1))

# Z-Score标准化
standardized = z_score_normalize(data)

# 稳健缩放
scaled = robust_scale(data)

# 小数缩放
decimal_scaled = decimal_scale(data, n_decimals=2)

# 对数归一化
log_normalized = log_normalize(data, epsilon=1e-8, base=np.e)
```

## 📊 算法选择指南

### 选择Min-Max归一化的情况：
- ✅ 数据分布相对均匀
- ✅ 无异常值或异常值较少
- ✅ 需要将数据映射到特定范围（如[0,1]）
- ✅ 需要保持零值的位置

### 选择Z-Score标准化的情况：
- ✅ 数据近似正态分布
- ✅ 对异常值不敏感
- ✅ 需要基于均值和标准差的统计特性
- ✅ 后续算法对数据分布有特定要求

### 选择稳健缩放的情况：
- ✅ 数据包含异常值
- ✅ 需要基于中位数和四分位数的统计特性
- ✅ 对异常值敏感的场景

### 选择小数缩放的情况：
- ✅ 需要控制数值精度
- ✅ 数据范围差异很大
- ✅ 需要保持数值的相对关系

### 选择对数归一化的情况：
- ✅ 数据呈右偏分布
- ✅ 数值范围差异很大
- ✅ 需要压缩大数值的影响

## ⚠️ 注意事项

1. **拟合顺序**: 必须先调用`fit()`方法，再调用`transform()`方法
2. **数据格式**: 支持numpy数组、列表、pandas Series和DataFrame
3. **除零处理**: 所有算法都包含除零保护机制
4. **逆变换**: 只有拟合后的归一化器才能进行逆变换
5. **异常值**: 稳健缩放和对数归一化对异常值更鲁棒

## 🧪 测试验证

运行测试脚本验证算法功能：

```bash
cd text_analysis/algorithms
python test_normalization.py
```

测试包括：
- ✅ 所有算法的基本功能
- ✅ 逆变换的正确性
- ✅ 便捷函数的一致性
- ✅ Pandas集成
- ✅ 错误处理机制

## 📈 性能特点

| 算法 | 时间复杂度 | 空间复杂度 | 对异常值敏感度 |
|------|------------|------------|----------------|
| Min-Max | O(n) | O(1) | 高 |
| Z-Score | O(n) | O(1) | 高 |
| 稳健缩放 | O(n log n) | O(1) | 低 |
| 小数缩放 | O(n) | O(1) | 中 |
| 对数归一化 | O(n) | O(1) | 低 |

## 🔗 相关链接

- [归一化算法源码](../normalization.py)
- [测试脚本](../test_normalization.py)
- [统计算法总览](../README.md)
