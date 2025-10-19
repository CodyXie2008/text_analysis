import os
import json
from typing import List, Optional, Set

from text_analysis.core.data_paths import PROJECT_ROOT, resolve_latest_cleaned_data


def _enumerate_aweme_ids_from_cleaned(cleaned_data_path: Optional[str]) -> List[str]:
    path = cleaned_data_path or resolve_latest_cleaned_data()
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ids: Set[str] = set()
        for item in data:
            vid = str(item.get('aweme_id')) if item.get('aweme_id') is not None else None
            if vid:
                ids.add(vid)
        return sorted(ids)
    except Exception:
        return []


def _enumerate_aweme_ids_from_db() -> List[str]:
    try:
        from text_analysis.core.db_config import get_db_conn
        import pandas as pd
        conn = get_db_conn()
        df = pd.read_sql_query("SELECT DISTINCT aweme_id FROM douyin_aweme_comment WHERE aweme_id IS NOT NULL", conn)
        conn.close()
        return sorted([str(x) for x in df['aweme_id'].dropna().unique().tolist()])
    except Exception:
        return []


def enumerate_aweme_ids(use_cleaned_data: bool = False, cleaned_data_path: Optional[str] = None) -> List[str]:
    """返回可用的所有视频ID列表。

    优先来源：清洗数据；若不可用则回退数据库。
    """
    if use_cleaned_data:
        ids = _enumerate_aweme_ids_from_cleaned(cleaned_data_path)
        if ids:
            return ids
    # 回退数据库
    return _enumerate_aweme_ids_from_db()


