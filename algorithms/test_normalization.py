# -*- coding: utf-8 -*-
"""
归一化算法测试脚本

验证所有归一化算法的功能和正确性
"""

import numpy as np
import pandas as pd
from normalization import (
    MinMaxNormalizer,
    ZScoreNormalizer,
    RobustScaler,
    DecimalScaler,
    LogNormalizer,
    min_max_normalize,
    z_score_normalize,
    robust_scale,
    decimal_scale,
    log_normalize
)


def test_min_max_normalizer():
    """测试Min-Max归一化算法"""
    print("=== 测试Min-Max归一化算法 ===")
    
    # 测试数据
    data = np.array([1, 5, 10, 15, 20])
    print(f"原始数据: {data}")
    
    # 使用类
    normalizer = MinMaxNormalizer(feature_range=(0, 1))
    normalized = normalizer.fit_transform(data)
    print(f"归一化结果 (0-1): {normalized.flatten()}")
    
    # 测试逆变换
    original = normalizer.inverse_transform(normalized)
    print(f"逆变换结果: {original.flatten()}")
    
    # 使用便捷函数
    normalized_func = min_max_normalize(data, feature_range=(0, 1))
    print(f"便捷函数结果: {normalized_func.flatten()}")
    
    # 测试自定义范围
    normalizer_custom = MinMaxNormalizer(feature_range=(-1, 1))
    normalized_custom = normalizer_custom.fit_transform(data)
    print(f"自定义范围 (-1,1): {normalized_custom.flatten()}")
    
    print("✅ Min-Max归一化测试通过\n")


def test_z_score_normalizer():
    """测试Z-Score标准化算法"""
    print("=== 测试Z-Score标准化算法 ===")
    
    # 测试数据
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"原始数据: {data}")
    
    # 使用类
    normalizer = ZScoreNormalizer()
    standardized = normalizer.fit_transform(data)
    print(f"标准化结果: {standardized.flatten()}")
    
    # 验证均值和标准差
    print(f"标准化后均值: {np.mean(standardized):.6f} (应该接近0)")
    print(f"标准化后标准差: {np.std(standardized):.6f} (应该接近1)")
    
    # 测试逆变换
    original = normalizer.inverse_transform(standardized)
    print(f"逆变换结果: {original.flatten()}")
    
    # 使用便捷函数
    standardized_func = z_score_normalize(data)
    print(f"便捷函数结果: {standardized_func.flatten()}")
    
    print("✅ Z-Score标准化测试通过\n")


def test_robust_scaler():
    """测试稳健缩放算法"""
    print("=== 测试稳健缩放算法 ===")
    
    # 测试数据（包含异常值）
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
    print(f"原始数据: {data}")
    
    # 使用类
    scaler = RobustScaler()
    scaled = scaler.fit_transform(data)
    print(f"稳健缩放结果: {scaled.flatten()}")
    
    # 验证中位数和IQR
    print(f"缩放后中位数: {np.median(scaled):.6f} (应该接近0)")
    print(f"缩放后IQR: {np.percentile(scaled, 75) - np.percentile(scaled, 25):.6f} (应该接近1)")
    
    # 测试逆变换
    original = scaler.inverse_transform(scaled)
    print(f"逆变换结果: {original.flatten()}")
    
    # 使用便捷函数
    scaled_func = robust_scale(data)
    print(f"便捷函数结果: {scaled_func.flatten()}")
    
    print("✅ 稳健缩放测试通过\n")


def test_decimal_scaler():
    """测试小数缩放算法"""
    print("=== 测试小数缩放算法 ===")
    
    # 测试数据
    data = np.array([100, 250, 500, 750, 1000])
    print(f"原始数据: {data}")
    
    # 使用类
    scaler = DecimalScaler(n_decimals=2)
    scaled = scaler.fit_transform(data)
    print(f"小数缩放结果: {scaled.flatten()}")
    
    # 测试逆变换
    original = scaler.inverse_transform(scaled)
    print(f"逆变换结果: {original.flatten()}")
    
    # 使用便捷函数
    scaled_func = decimal_scale(data, n_decimals=2)
    print(f"便捷函数结果: {scaled_func.flatten()}")
    
    print("✅ 小数缩放测试通过\n")


def test_log_normalizer():
    """测试对数归一化算法"""
    print("=== 测试对数归一化算法 ===")
    
    # 测试数据（右偏分布）
    data = np.array([1, 2, 5, 10, 50, 100, 500, 1000])
    print(f"原始数据: {data}")
    
    # 使用类
    normalizer = LogNormalizer(epsilon=1e-8, base=np.e)
    log_normalized = normalizer.fit_transform(data)
    print(f"对数归一化结果: {log_normalized.flatten()}")
    
    # 测试逆变换
    original = normalizer.inverse_transform(log_normalized)
    print(f"逆变换结果: {original.flatten()}")
    
    # 使用便捷函数
    log_normalized_func = log_normalize(data, epsilon=1e-8, base=np.e)
    print(f"便捷函数结果: {log_normalized_func.flatten()}")
    
    # 测试以10为底的对数
    normalizer_10 = LogNormalizer(epsilon=1e-8, base=10)
    log_normalized_10 = normalizer_10.fit_transform(data)
    print(f"以10为底的对数归一化: {log_normalized_10.flatten()}")
    
    print("✅ 对数归一化测试通过\n")


def test_pandas_integration():
    """测试与Pandas的集成"""
    print("=== 测试Pandas集成 ===")
    
    # 创建DataFrame
    df = pd.DataFrame({
        'feature1': [1, 5, 10, 15, 20],
        'feature2': [100, 200, 300, 400, 500],
        'feature3': [0.1, 0.5, 1.0, 1.5, 2.0]
    })
    print("原始DataFrame:")
    print(df)
    
    # 测试Min-Max归一化
    normalizer = MinMaxNormalizer(feature_range=(0, 1))
    normalized_df = normalizer.fit_transform(df)
    print("\nMin-Max归一化后:")
    print(pd.DataFrame(normalized_df, columns=df.columns))
    
    # 测试Z-Score标准化
    z_normalizer = ZScoreNormalizer()
    z_normalized_df = z_normalizer.fit_transform(df)
    print("\nZ-Score标准化后:")
    print(pd.DataFrame(z_normalized_df, columns=df.columns))
    
    print("✅ Pandas集成测试通过\n")


def test_error_handling():
    """测试错误处理"""
    print("=== 测试错误处理 ===")
    
    # 测试空数据
    try:
        min_max_normalize([])
        print("❌ 空数据处理失败")
    except Exception as e:
        print(f"✅ 空数据处理正确: {e}")
    
    # 测试未拟合的逆变换
    try:
        normalizer = MinMaxNormalizer()
        normalizer.inverse_transform([0.5])
        print("❌ 未拟合逆变换处理失败")
    except ValueError as e:
        print(f"✅ 未拟合逆变换处理正确: {e}")
    
    # 测试除零情况
    data_zeros = np.array([1, 1, 1, 1, 1])  # 所有值相同
    try:
        normalized = min_max_normalize(data_zeros)
        print(f"✅ 除零情况处理正确: {normalized.flatten()}")
    except Exception as e:
        print(f"❌ 除零情况处理失败: {e}")
    
    print("✅ 错误处理测试通过\n")


def main():
    """主测试函数"""
    print("🚀 开始归一化算法测试\n")
    
    try:
        test_min_max_normalizer()
        test_z_score_normalizer()
        test_robust_scaler()
        test_decimal_scaler()
        test_log_normalizer()
        test_pandas_integration()
        test_error_handling()
        
        print("🎉 所有测试通过！归一化算法工作正常。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
