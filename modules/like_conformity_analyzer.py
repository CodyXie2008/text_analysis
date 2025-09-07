# -*- coding: utf-8 -*-
"""
从众心理点赞分析模块

基于点赞数的从众心理分析：
计算公式：父评论点赞数 - 子评论点赞数的绝对值
如果绝对值越小，那么这条子评论跟随父评论的从众越大
如果绝对值越大，从众越小
对每个绝对值进行归一化，最终得到以父评论为环境的点赞指数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import argparse
import os
import json

# 添加项目根目录到Python路径
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from text_analysis.core.base_analyzer import BaseAnalyzer, create_parser, parse_common_args
from text_analysis.algorithms.normalization import MinMaxNormalizer

logger = logging.getLogger(__name__)


class LikeConformityAnalyzer(BaseAnalyzer):
    """
    从众心理点赞分析器
    
    分析子评论相对于父评论的点赞从众心理特征：
    1. 计算子评论与父评论的点赞数差异
    2. 基于点赞差异计算从众心理分数（差异越小分数越高）
    3. 对分数进行归一化处理
    4. 生成点赞从众心理分析报告
    """
    
    def __init__(self, module_name: str = "like_conformity_analyzer", output_dir: str = None, **kwargs):
        """初始化从众心理点赞分析器"""
        super().__init__(module_name=module_name, **kwargs)
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output")
        self.normalizer = MinMaxNormalizer(feature_range=(0, 1))
        self.results = {}
        
    def calculate_like_conformity_score(self, data: Union[pd.DataFrame, List[Dict]]) -> Union[pd.DataFrame, List[Dict]]:
        """
        计算点赞从众心理分数
        
        Args:
            data: 包含评论数据的DataFrame或字典列表，需要包含以下字段：
                - like_count: 评论点赞数
                - comment_id: 评论ID
                - is_parent: 父评论标识（可选，如果数据已通过数据清洗模块处理）
                
        Returns:
            Union[pd.DataFrame, List[Dict]]: 包含点赞从众心理分数的数据
        """
        logger.info("开始计算点赞从众心理分数...")
        
        # 处理输入数据格式
        if isinstance(data, list):
            # 如果是字典列表，转换为DataFrame
            df = pd.DataFrame(data)
            return_dict = True
        else:
            df = data.copy()
            return_dict = False
        
        # 检查数据是否已经过处理
        if 'is_parent' in df.columns:
            # 数据已经过数据清洗模块处理
            logger.info("检测到已处理的数据，直接使用")
        else:
            # 需要手动处理数据
            logger.info("数据未经过处理，开始处理...")
            
            # 确保必要的列存在
            if 'like_count' not in df.columns:
                raise ValueError("数据中缺少like_count字段")
            if 'comment_id' not in df.columns:
                raise ValueError("数据中缺少comment_id字段")
            
            # 假设第一个评论是父评论（或基于其他逻辑识别父评论）
            df['is_parent'] = False
            df.loc[0, 'is_parent'] = True
            logger.info("假设第一个评论为父评论")
        
        # 分离父评论和子评论
        parent_data = df[df['is_parent'] == True]
        child_data = df[df['is_parent'] == False]
        
        if len(parent_data) == 0:
            raise ValueError("未找到父评论数据")
        
        parent_comment = parent_data.iloc[0]
        parent_likes = parent_comment['like_count']
        parent_id = parent_comment['comment_id']
        
        logger.info(f"父评论ID: {parent_id}, 父评论点赞数: {parent_likes}")
        logger.info(f"子评论数量: {len(child_data)}")
        
        # 计算每个子评论的点赞从众心理分数
        like_differences = []
        conformity_scores = []
        
        for idx, row in child_data.iterrows():
            child_likes = row['like_count']
            # 计算点赞数差异的绝对值
            like_diff = abs(parent_likes - child_likes)
            like_differences.append(like_diff)
        
        # 计算自适应归一化参数
        if like_differences:
            # 使用中位数作为参考点，确保大部分分数在合理范围内
            median_diff = np.median(like_differences)
            max_diff = np.max(like_differences)
            
            # 如果差异范围很大，使用对数缩放
            if max_diff > median_diff * 10:
                # 使用对数缩放处理大范围差异
                log_differences = [np.log1p(diff) for diff in like_differences]
                # 归一化对数差异
                normalized_scores = self.normalizer.fit_transform(
                    np.array(log_differences).reshape(-1, 1)
                ).flatten()
                # 反转分数（差异越小，从众心理越强）
                conformity_scores = 1 - normalized_scores
                logger.info("使用对数缩放处理大范围点赞差异")
            else:
                # 直接归一化差异
                normalized_scores = self.normalizer.fit_transform(
                    np.array(like_differences).reshape(-1, 1)
                ).flatten()
                # 反转分数（差异越小，从众心理越强）
                conformity_scores = 1 - normalized_scores
                logger.info("使用直接归一化处理点赞差异")
        else:
            logger.warning("未找到子评论数据")
            conformity_scores = []
        
        # 为父评论添加分数（设为1.0，表示完全从众）
        parent_conformity_score = 1.0
        
        # 将结果添加到DataFrame
        df['like_difference'] = 0  # 父评论的差异为0
        df['raw_like_conformity_score'] = 0.0  # 父评论的原始分数为0
        df['normalized_like_conformity_score'] = 0.0  # 父评论的归一化分数为0
        
        # 更新子评论的分数
        child_indices = child_data.index
        for i, idx in enumerate(child_indices):
            df.loc[idx, 'like_difference'] = like_differences[i]
            df.loc[idx, 'raw_like_conformity_score'] = like_differences[i]
            df.loc[idx, 'normalized_like_conformity_score'] = conformity_scores[i]
        
        # 更新父评论的分数
        parent_idx = parent_data.index[0]
        df.loc[parent_idx, 'normalized_like_conformity_score'] = parent_conformity_score
        
        # 添加点赞分类
        df['like_category'] = self._categorize_like_differences(df['like_difference'].tolist())
        
        logger.info(f"点赞从众心理分数计算完成，共处理 {len(df)} 条评论")
        logger.info(f"父评论点赞数: {parent_likes}, 子评论数量: {len(child_data)}")
        
        if return_dict:
            return df.to_dict('records')
        else:
            return df
    
    def _categorize_like_differences(self, like_differences: List[float]) -> List[str]:
        """
        对点赞差异进行分类
        
        Args:
            like_differences: 点赞差异列表
            
        Returns:
            List[str]: 点赞分类列表
        """
        categories = []
        for diff in like_differences:
            if diff == 0:  # 父评论
                categories.append('父评论')
            elif diff <= 10:  # 差异很小
                categories.append('高度从众')
            elif diff <= 50:  # 差异较小
                categories.append('中度从众')
            elif diff <= 100:  # 差异中等
                categories.append('轻度从众')
            elif diff <= 500:  # 差异较大
                categories.append('低度从众')
            else:  # 差异很大
                categories.append('非从众')
        
        return categories
    
    def calculate_parent_like_conformity_score(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """
        计算父评论的总体点赞从众心理分数
        
        Args:
            data: 包含评论数据的DataFrame或字典列表
            
        Returns:
            Dict: 父评论点赞从众心理分数统计结果
        """
        logger.info("开始计算父评论点赞从众心理分数...")
        
        # 计算点赞从众心理分数
        result_data = self.calculate_like_conformity_score(data)
        
        # 转换为DataFrame格式进行处理
        if isinstance(result_data, list):
            df = pd.DataFrame(result_data)
        else:
            df = result_data
        
        # 分离父评论和子评论
        parent_data = df[df['is_parent'] == True]
        child_data = df[df['is_parent'] == False]
        
        if len(parent_data) == 0:
            raise ValueError("未找到父评论数据")
        
        parent_comment = parent_data.iloc[0]
        parent_id = parent_comment['comment_id']
        parent_likes = parent_comment['like_count']
        
        # 计算子评论的点赞从众心理分数统计
        child_scores = child_data['normalized_like_conformity_score'].tolist()
        child_like_diffs = child_data['like_difference'].tolist()
        
        if len(child_scores) == 0:
            logger.warning("未找到子评论数据")
            return {
                'parent_comment_id': parent_id,
                'parent_like_count': parent_likes,
                'child_comment_count': 0,
                'parent_like_conformity_score': 0.0,
                'message': '无子评论数据'
            }
        
        # 基础统计
        avg_score = np.mean(child_scores)
        median_score = np.median(child_scores)
        max_score = np.max(child_scores)
        min_score = np.min(child_scores)
        std_score = np.std(child_scores)
        
        # 高从众心理比例（分数 > 0.7）
        high_conformity_count = sum(1 for score in child_scores if score > 0.7)
        high_conformity_ratio = high_conformity_count / len(child_scores)
        
        # 点赞差异统计
        avg_like_diff = np.mean(child_like_diffs)
        median_like_diff = np.median(child_like_diffs)
        min_like_diff = np.min(child_like_diffs)
        max_like_diff = np.max(child_like_diffs)
        
        # 点赞分类统计
        like_categories = child_data['like_category'].value_counts()
        
        result = {
            'parent_comment_id': parent_id,
            'parent_like_count': parent_likes,
            'child_comment_count': len(child_scores),
            'parent_like_conformity_score': avg_score,  # 主要分数：平均值
            'statistics': {
                'mean_score': avg_score,
                'median_score': median_score,
                'max_score': max_score,
                'min_score': min_score,
                'std_score': std_score
            },
            'conformity_distribution': {
                'high_conformity_count': high_conformity_count,
                'high_conformity_ratio': high_conformity_ratio,
                'like_category_distribution': like_categories.to_dict()
            },
            'like_analysis': {
                'avg_like_difference': avg_like_diff,
                'median_like_difference': median_like_diff,
                'min_like_difference': min_like_diff,
                'max_like_difference': max_like_diff
            },
            'parent_info': {
                'like_count': parent_likes,
                'comment_id': parent_id
            }
        }
        
        logger.info(f"父评论点赞从众心理分数计算完成: {parent_id}, 分数: {avg_score:.4f}")
        
        return result
    
    def analyze_like_conformity_patterns(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        分析点赞从众心理模式
        
        Args:
            data: 包含点赞从众心理分数的DataFrame
            
        Returns:
            Dict: 点赞从众心理模式分析结果
        """
        logger.info("开始分析点赞从众心理模式...")
        
        analysis_results = {}
        
        # 1. 整体统计
        analysis_results['overall_stats'] = {
            'total_comments': len(data),
            'unique_parents': data['parent_comment_id'].nunique() if 'parent_comment_id' in data.columns else 1,
            'avg_like_conformity_score': data['normalized_like_conformity_score'].mean(),
            'std_like_conformity_score': data['normalized_like_conformity_score'].std(),
            'median_like_conformity_score': data['normalized_like_conformity_score'].median(),
            'min_like_conformity_score': data['normalized_like_conformity_score'].min(),
            'max_like_conformity_score': data['normalized_like_conformity_score'].max()
        }
        
        # 2. 点赞分类统计
        like_category_stats = data.groupby('like_category').agg({
            'normalized_like_conformity_score': ['count', 'mean', 'std'],
            'like_difference': ['mean', 'median']
        }).round(4)
        
        analysis_results['like_category_stats'] = like_category_stats
        
        # 3. 点赞差异与从众心理的相关性
        correlation = data['like_difference'].corr(data['normalized_like_conformity_score'])
        analysis_results['like_conformity_correlation'] = correlation
        
        # 4. 高从众心理评论的特征分析
        high_conformity_threshold = data['normalized_like_conformity_score'].quantile(0.8)
        high_conformity_data = data[data['normalized_like_conformity_score'] >= high_conformity_threshold]
        
        analysis_results['high_conformity_analysis'] = {
            'threshold': high_conformity_threshold,
            'count': len(high_conformity_data),
            'percentage': len(high_conformity_data) / len(data) * 100,
            'avg_like_difference': high_conformity_data['like_difference'].mean(),
            'like_category_distribution': high_conformity_data['like_category'].value_counts()
        }
        
        logger.info("点赞从众心理模式分析完成")
        
        return analysis_results
    
    def generate_like_conformity_report(self, data: pd.DataFrame, 
                                      analysis_results: Dict[str, any]) -> str:
        """
        生成点赞从众心理分析报告
        
        Args:
            data: 包含分析结果的DataFrame
            analysis_results: 分析结果字典
            
        Returns:
            str: Markdown格式的分析报告
        """
        logger.info("生成点赞从众心理分析报告...")
        
        report = f"""# 点赞从众心理分析报告

## 分析概述
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总评论数**: {analysis_results['overall_stats']['total_comments']}
- **父评论数**: {analysis_results['overall_stats']['unique_parents']}
- **平均点赞从众心理分数**: {analysis_results['overall_stats']['avg_like_conformity_score']:.4f}

## 主要发现

### 1. 整体点赞从众心理特征
- **平均点赞从众心理分数**: {analysis_results['overall_stats']['avg_like_conformity_score']:.4f}
- **标准差**: {analysis_results['overall_stats']['std_like_conformity_score']:.4f}
- **中位数**: {analysis_results['overall_stats']['median_like_conformity_score']:.4f}
- **分数范围**: [{analysis_results['overall_stats']['min_like_conformity_score']:.4f}, {analysis_results['overall_stats']['max_like_conformity_score']:.4f}]

### 2. 点赞差异与从众心理相关性
- **相关系数**: {analysis_results['like_conformity_correlation']:.4f}
- **解释**: {'强负相关' if analysis_results['like_conformity_correlation'] < -0.5 else '中等负相关' if analysis_results['like_conformity_correlation'] < -0.3 else '弱负相关' if analysis_results['like_conformity_correlation'] < 0 else '弱正相关' if analysis_results['like_conformity_correlation'] < 0.3 else '中等正相关' if analysis_results['like_conformity_correlation'] < 0.5 else '强正相关'}（点赞差异越小，从众心理越强）

### 3. 点赞分类统计
"""
        
        # 添加点赞分类统计表格
        like_stats = analysis_results['like_category_stats']
        for category in like_stats.index:
            count = like_stats.loc[category, ('normalized_like_conformity_score', 'count')]
            mean_score = like_stats.loc[category, ('normalized_like_conformity_score', 'mean')]
            avg_diff = like_stats.loc[category, ('like_difference', 'mean')]
            report += f"- **{category}**: {count}条评论，平均分数{mean_score:.4f}，平均差异{avg_diff:.1f}个点赞\n"
        
        report += f"""
### 4. 高从众心理评论分析
- **高从众心理阈值**: {analysis_results['high_conformity_analysis']['threshold']:.4f}
- **高从众心理评论数**: {analysis_results['high_conformity_analysis']['count']}条
- **占比**: {analysis_results['high_conformity_analysis']['percentage']:.1f}%
- **平均点赞差异**: {analysis_results['high_conformity_analysis']['avg_like_difference']:.1f}个点赞

### 5. 高从众心理评论的点赞分布
"""
        
        # 添加高从众心理评论的点赞分布
        like_dist = analysis_results['high_conformity_analysis']['like_category_distribution']
        for category, count in like_dist.items():
            percentage = count / analysis_results['high_conformity_analysis']['count'] * 100
            report += f"- **{category}**: {count}条评论 ({percentage:.1f}%)\n"
        
        report += """
## 结论与建议

### 主要结论
1. **点赞效应明显**: 评论点赞数与父评论越接近，从众心理越强
2. **差异影响显著**: 点赞差异越小，从众心理表现越明显
3. **分布特征**: 点赞从众心理分数呈现合理的分布特征

### 建议
1. **内容监控**: 重点关注高从众心理区域的评论内容
2. **点赞分析**: 在内容分析中考虑点赞从众心理因素
3. **用户行为**: 分析用户点赞行为中的从众心理模式
4. **算法优化**: 在推荐算法中考虑点赞从众心理因素

---
*本报告由点赞从众心理分析系统自动生成*
"""
        
        return report
    
    def analyze(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """
        实现BaseAnalyzer的analyze方法
        """
        logger.info("开始点赞从众心理分析...")
        
        # 计算点赞从众心理分数
        processed_data = self.calculate_like_conformity_score(data)
        if isinstance(processed_data, list):
            processed_df = pd.DataFrame(processed_data)
        else:
            processed_df = processed_data
        
        # 计算父评论总体点赞从众心理分数
        parent_overall_result = self.calculate_parent_like_conformity_score(processed_df)
        
        # 分析点赞从众心理模式
        pattern_results = self.analyze_like_conformity_patterns(processed_df)
        
        # 生成报告
        report_content = self.generate_like_conformity_report(processed_df, pattern_results)
        
        # 准备输出结果
        output_result = {
            "analysis_metadata": {
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analyzer_version": "v1.0.0",
                "data_source": getattr(self, 'data_source', 'unknown'),
                "total_comments": len(processed_df)
            },
            "parent_comment_info": {
                "comment_id": parent_overall_result.get('parent_comment_id', 'N/A'),
                "content": processed_df[processed_df['comment_id'] == parent_overall_result.get('parent_comment_id', 'N/A')]['content'].iloc[0] if not processed_df[processed_df['comment_id'] == parent_overall_result.get('parent_comment_id', 'N/A')].empty else 'N/A',
            },
            "parent_environment_like_analysis": parent_overall_result,
            "like_classification": pattern_results.get('like_category_distribution', {}),
            "top_high_like_comments": processed_df.nlargest(10, 'normalized_like_conformity_score').to_dict('records') if not processed_df.empty else []
        }
        
        logger.info("点赞从众心理分析完成")
        return output_result
    
    def run_analysis(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        运行完整的点赞从众心理分析
        
        Args:
            data: 包含评论数据的DataFrame
            
        Returns:
            Dict: 完整的分析结果
        """
        logger.info("开始点赞从众心理分析...")
        
        # 1. 计算点赞从众心理分数
        result_data = self.calculate_like_conformity_score(data)
        
        # 2. 分析点赞从众心理模式
        analysis_results = self.analyze_like_conformity_patterns(result_data)
        
        # 3. 生成报告
        report = self.generate_like_conformity_report(result_data, analysis_results)
        
        # 4. 保存结果
        self._save_results(result_data, analysis_results, report)
        
        # 5. 保存到实例变量
        self.results = {
            'data': result_data,
            'analysis': analysis_results,
            'report': report
        }
        
        logger.info("点赞从众心理分析完成")
        
        return self.results
    
    def _save_results(self, data: pd.DataFrame, analysis_results: Dict[str, any], report: str):
        """保存分析结果"""
        logger.info("保存分析结果...")
        
        # 创建输出目录
        output_dir = os.path.join(self.output_dir, "like_conformity_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细数据
        data_path = os.path.join(output_dir, f"like_conformity_data_{timestamp}.csv")
        data.to_csv(data_path, index=False, encoding='utf-8-sig')
        
        # 保存分析结果
        analysis_path = os.path.join(output_dir, f"like_conformity_analysis_{timestamp}.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存报告
        report_path = os.path.join(output_dir, f"like_conformity_report_{timestamp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"分析结果已保存到: {output_dir}")


def main():
    """主函数"""
    parser = create_parser("点赞从众心理分析")
    parser.add_argument('--like-col', default='like_count',
                       help='点赞数列名')
    parser.add_argument('--comment-id-col', default='comment_id',
                       help='评论ID列名')
    
    args = parse_common_args(parser)
    
    # 创建分析器
    analyzer = LikeConformityAnalyzer(
        output_dir=args.output_dir,
        db_config=args.db_config
    )
    
    # 加载数据
    if args.input_file:
        logger.info(f"从文件加载数据: {args.input_file}")
        data = pd.read_csv(args.input_file)
    else:
        logger.info("从数据库加载数据...")
        data = analyzer.load_data_from_db(
            table_name='douyin_aweme_comment',
            limit=args.limit
        )
    
    # 确保必要的列存在
    required_columns = [args.like_col, args.comment_id_col]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")
    
    # 运行分析
    results = analyzer.run_analysis(data)
    
    # 打印摘要
    print("\n=== 点赞从众心理分析摘要 ===")
    print(f"总评论数: {results['analysis']['overall_stats']['total_comments']}")
    print(f"父评论数: {results['analysis']['overall_stats']['unique_parents']}")
    print(f"平均点赞从众心理分数: {results['analysis']['overall_stats']['avg_like_conformity_score']:.4f}")
    print(f"点赞相关性: {results['analysis']['like_conformity_correlation']:.4f}")
    
    logger.info("点赞从众心理分析完成")


if __name__ == "__main__":
    main()
