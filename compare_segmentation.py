import json
import os
import difflib
from datetime import datetime

# 配置文件路径（根据实际生成的文件名修改）
LOCAL_FILE = "data/processed/processed_cleaning_7306437681045654834_20250830_180041.json"
API_FILE = "data/processed/processed_cleaning_7306437681045654834_20250830_180111.json"
OUTPUT_FILE = "data/analysis/segmentation_comparison_{}.json".format(datetime.now().strftime("%Y%m%d_%H%M%S"))


def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return None


def compare_segmentation(local_data, api_data):
    """比较本地模式和API模式的分词结果"""
    if not local_data or not api_data:
        return None

    # 创建以评论ID为键的字典
    local_comments = {item['comment_id']: item for item in local_data}
    api_comments = {item['comment_id']: item for item in api_data}

    # 找出共同的评论ID
    common_ids = set(local_comments.keys()) & set(api_comments.keys())
    only_local_ids = set(local_comments.keys()) - set(api_comments.keys())
    only_api_ids = set(api_comments.keys()) - set(local_comments.keys())

    comparison_results = {
        "common_comments_count": len(common_ids),
        "only_local_comments_count": len(only_local_ids),
        "only_api_comments_count": len(only_api_ids),
        "detailed_comparison": []
    }

    # 比较共同评论的分词结果
    for comment_id in common_ids:
        local_comment = local_comments[comment_id]
        api_comment = api_comments[comment_id]

        local_words = local_comment.get('words', [])
        api_words = api_comment.get('words', [])

        # 计算分词数量差异
        word_count_diff = len(api_words) - len(local_words)

        # 找出相同和不同的词
        same_words = set(local_words) & set(api_words)
        only_local_words = set(local_words) - set(api_words)
        only_api_words = set(api_words) - set(local_words)

        # 生成差异报告
        diff_report = {
            "comment_id": comment_id,
            "original_text": local_comment.get('text', 'N/A'),
            "local_words_count": len(local_words),
            "api_words_count": len(api_words),
            "word_count_diff": word_count_diff,
            "same_words_count": len(same_words),
            "only_local_words_count": len(only_local_words),
            "only_api_words_count": len(only_api_words),
            "only_local_words": list(only_local_words),
            "only_api_words": list(only_api_words),
            "local_words": local_words,
            "api_words": api_words
        }

        comparison_results['detailed_comparison'].append(diff_report)

    return comparison_results


def save_comparison_results(results, output_file):
    """保存比较结果到文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"比较结果已保存到: {output_file}")


def display_summary(results):
    """显示比较结果摘要"""
    print("\n==== 分词模式比较摘要 ====")
    print(f"共同评论数量: {results['common_comments_count']}")
    print(f"仅本地模式保留的评论: {results['only_local_comments_count']}")
    print(f"仅API模式保留的评论: {results['only_api_comments_count']}")

    # 计算平均分词数量差异
    if results['common_comments_count'] > 0:
        total_diff = sum(item['word_count_diff'] for item in results['detailed_comparison'])
        avg_diff = total_diff / results['common_comments_count']
        print(f"平均分词数量差异 (API - 本地): {avg_diff:.2f}")

    print(f"详细比较结果已保存到: {OUTPUT_FILE}")
    print("==========================")


def main():
    print("开始比较本地模式和API模式的分词结果...")

    # 加载数据
    local_data = load_json_file(LOCAL_FILE)
    api_data = load_json_file(API_FILE)

    # 比较分词结果
    comparison_results = compare_segmentation(local_data, api_data)

    if comparison_results:
        # 保存结果
        save_comparison_results(comparison_results, OUTPUT_FILE)

        # 显示摘要
        display_summary(comparison_results)
    else:
        print("比较失败，请检查文件是否存在或格式是否正确。")


if __name__ == "__main__":
    main()