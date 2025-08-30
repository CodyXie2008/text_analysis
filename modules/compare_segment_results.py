#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
from collections import Counter
from typing import Dict, List


def load_cleaned(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_stats(items: List[Dict]) -> Dict:
    words_lists = [x.get('words') or [] for x in items]
    word_counts = [len(w) for w in words_lists]
    vocab = Counter()
    for ws in words_lists:
        vocab.update(ws)
    avg_wc = sum(word_counts) / len(word_counts) if word_counts else 0.0
    return {
        'num_records': len(items),
        'avg_word_count': avg_wc,
        'vocab_size': len(vocab),
        'top_words': vocab.most_common(50),
        'vocab_set': set(vocab.keys()),
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main():
    if len(sys.argv) < 3:
        print('Usage: compare_segment_results.py <local_cleaned.json> <api_cleaned.json> [output.json]')
        sys.exit(1)
    p_local = sys.argv[1]
    p_api = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    data_local = load_cleaned(p_local)
    data_api = load_cleaned(p_api)

    s_local = build_stats(data_local)
    s_api = build_stats(data_api)

    jac = jaccard(s_local['vocab_set'], s_api['vocab_set'])
    only_local = list(s_local['vocab_set'] - s_api['vocab_set'])[:50]
    only_api = list(s_api['vocab_set'] - s_local['vocab_set'])[:50]

    report = {
        'local': {
            'num_records': s_local['num_records'],
            'avg_word_count': s_local['avg_word_count'],
            'vocab_size': s_local['vocab_size'],
            'top_words': s_local['top_words'],
            'source_path': p_local,
        },
        'api': {
            'num_records': s_api['num_records'],
            'avg_word_count': s_api['avg_word_count'],
            'vocab_size': s_api['vocab_size'],
            'top_words': s_api['top_words'],
            'source_path': p_api,
        },
        'comparison': {
            'vocab_jaccard': jac,
            'only_in_local_sample': only_local,
            'only_in_api_sample': only_api,
        }
    }

    # 移除集合以便序列化
    del s_local['vocab_set']
    del s_api['vocab_set']

    if not out_path:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(p_local))), 'data', 'results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'compare_segment_report.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'Report saved: {out_path}')


if __name__ == '__main__':
    main()



