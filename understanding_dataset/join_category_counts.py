import json
import os
from typing import Any, Dict

ROOT = os.path.dirname(__file__)
SUMMARY_PATH = os.path.join(ROOT, 'category_summary.json')
COUNTS_PATH = os.path.join(ROOT, 'categories_ds2_dense_tmn.json')
CATEGORIES_PATH = os.path.join(ROOT, 'categories.json')
OUT_PATH = os.path.join(ROOT, 'category_summary_with_counts.json')


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    summary = load_json(SUMMARY_PATH)
    counts = load_json(COUNTS_PATH)
    categories: Dict[str, Dict[str, Any]] = load_json(CATEGORIES_PATH)

    # Build id->count and id->name maps
    id_to_count: Dict[int, int] = {}
    items = counts.get('items', [])
    for it in items:
        cid = it.get('category_id')
        cnt = it.get('count')
        if isinstance(cid, int) and isinstance(cnt, int):
            id_to_count[cid] = cnt

    id_to_name: Dict[int, str] = {}
    for k, v in categories.items():
        try:
            cid = int(k)
            name = v.get('name')
            if name:
                id_to_name[cid] = name
        except Exception:
            pass

    # Update items with counts and names
    for it in summary.get('items', []):
        cid = it.get('category_id')
        if isinstance(cid, int):
            it['count'] = id_to_count.get(cid, 0)
            it['name'] = it.get('name') or id_to_name.get(cid)

    # Fill category_names aligned with category_ids order
    cat_ids = summary.get('category_ids', [])
    summary['category_names'] = [id_to_name.get(cid) for cid in cat_ids]

    # Write output
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH}")


if __name__ == '__main__':
    main()
