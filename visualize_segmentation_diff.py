import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置文件路径（根据实际生成的文件名修改）
COMPARISON_FILE = "data/analysis/segmentation_comparison_20250830_180216.json"
OUTPUT_DIR = "data/visualizations/segmentation_diff"


def load_comparison_results(file_path):
    """加载比较结果文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return None


def create_word_count_comparison_chart(results, output_file):
    """创建分词数量比较图表"""
    comment_ids = [item['comment_id'] for item in results['detailed_comparison']]
    local_counts = [item['local_words_count'] for item in results['detailed_comparison']]
    api_counts = [item['api_words_count'] for item in results['detailed_comparison']]

    plt.figure(figsize=(12, 6))
    x = range(len(comment_ids))
    width = 0.35

    plt.bar([i - width/2 for i in x], local_counts, width, label='本地模式')
    plt.bar([i + width/2 for i in x], api_counts, width, label='API模式')

    plt.xlabel('评论ID')
    plt.ylabel('分词数量')
    plt.title('本地模式与API模式分词数量比较')
    plt.xticks(x, comment_ids, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"分词数量比较图表已保存到: {output_file}")


def create_unique_words_chart(results, output_file):
    """创建独有词汇数量图表"""
    comment_ids = [item['comment_id'] for item in results['detailed_comparison']]
    only_local = [item['only_local_words_count'] for item in results['detailed_comparison']]
    only_api = [item['only_api_words_count'] for item in results['detailed_comparison']]

    plt.figure(figsize=(12, 6))
    x = range(len(comment_ids))
    width = 0.35

    plt.bar([i - width/2 for i in x], only_local, width, label='仅本地模式识别')
    plt.bar([i + width/2 for i in x], only_api, width, label='仅API模式识别')

    plt.xlabel('评论ID')
    plt.ylabel('独有词汇数量')
    plt.title('本地模式与API模式独有词汇数量比较')
    plt.xticks(x, comment_ids, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"独有词汇数量图表已保存到: {output_file}")


def create_same_words_chart(results, output_file):
    """创建相同词汇占比图表"""
    comment_ids = [item['comment_id'] for item in results['detailed_comparison']]
    same_words = [item['same_words_count'] for item in results['detailed_comparison']]
    total_words = [max(item['local_words_count'], item['api_words_count']) for item in results['detailed_comparison']]
    same_ratio = [s/t if t > 0 else 0 for s, t in zip(same_words, total_words)]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=comment_ids, y=same_ratio)

    plt.xlabel('评论ID')
    plt.ylabel('相同词汇占比')
    plt.title('本地模式与API模式相同词汇占比')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"相同词汇占比图表已保存到: {output_file}")


def main():
    print("开始可视化本地模式和API模式的分词差异...")

    # 加载比较结果
    comparison_results = load_comparison_results(COMPARISON_FILE)

    if comparison_results:
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 创建各种图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        word_count_file = os.path.join(OUTPUT_DIR, f"word_count_comparison_{timestamp}.png")
        create_word_count_comparison_chart(comparison_results, word_count_file)

        unique_words_file = os.path.join(OUTPUT_DIR, f"unique_words_comparison_{timestamp}.png")
        create_unique_words_chart(comparison_results, unique_words_file)

        same_words_file = os.path.join(OUTPUT_DIR, f"same_words_ratio_{timestamp}.png")
        create_same_words_chart(comparison_results, same_words_file)

        print("\n==== 可视化结果摘要 ====")
        print(f"共同评论数量: {comparison_results['common_comments_count']}")
        print(f"仅本地模式保留的评论: {comparison_results['only_local_comments_count']}")
        print(f"仅API模式保留的评论: {comparison_results['only_api_comments_count']}")
        print(f"图表已保存到目录: {OUTPUT_DIR}")
        print("========================")
    else:
        print("可视化失败，请检查比较结果文件是否存在或格式是否正确。")


if __name__ == "__main__":
    main()