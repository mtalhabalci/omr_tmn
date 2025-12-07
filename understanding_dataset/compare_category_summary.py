import json
import os
from typing import Dict, Any

ROOT = os.path.dirname(__file__)
SUMMARY_PATH = os.path.join(ROOT, 'category_summary.json')
CATEGORIES_PATH = os.path.join(ROOT, 'categories.json')
COUNTS_PATH = os.path.join(ROOT, 'categories_ds2_dense_tmn.json')  # optional
OUT_PATH = os.path.join(ROOT, 'unseen_categories.json')


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    summary = load_json(SUMMARY_PATH)
    categories: Dict[str, Dict[str, Any]] = load_json(CATEGORIES_PATH)

    superset_ids = set(int(k) for k in categories.keys())
    observed_ids = set(summary.get('category_ids', []))

    unseen_ids = sorted(list(superset_ids - observed_ids))
    observed_not_in_superset = sorted(list(observed_ids - superset_ids))

    unseen_items = []
    for cid in unseen_ids:
        c = categories.get(str(cid), {})
        unseen_items.append({
            'id': cid,
            'name': c.get('name'),
            'annotation_set': c.get('annotation_set'),
        })

    tmn_presence = {}
    for cid in range(209, 217):  # 209..216
        tmn_presence[str(cid)] = (cid in observed_ids)

    result = {
        'superset_total': len(superset_ids),
        'observed_total': len(observed_ids),
        'coverage_ratio': (len(observed_ids) / len(superset_ids)) if superset_ids else 0.0,
        'unseen_count': len(unseen_ids),
        'unseen_ids': unseen_ids,
        'unseen_items': unseen_items,
        'observed_not_in_superset': observed_not_in_superset,
        'tmn_presence': tmn_presence,
    }

    # Optionally enrich with counts if available
    if os.path.exists(COUNTS_PATH):
        try:
            counts = load_json(COUNTS_PATH)
            # Build a map id->count if the structure is an array of items
            id_to_count = {}
            items = counts.get('items') if isinstance(counts, dict) else None
            if items and isinstance(items, list):
                for it in items:
                    cid = it.get('category_id')
                    cnt = it.get('count')
                    if cid is not None and cnt is not None:
                        id_to_count[int(cid)] = int(cnt)
            # Add TMN counts
            tmn_counts = {}
            for cid in range(209, 217):
                tmn_counts[str(cid)] = id_to_count.get(cid)
            result['tmn_counts'] = tmn_counts
        except Exception:
            # Keep minimal output if parsing fails
            pass

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print brief summary to console
    print(json.dumps({
        'superset_total': result['superset_total'],
        'observed_total': result['observed_total'],
        'coverage_ratio': result['coverage_ratio'],
        'unseen_count': result['unseen_count'],
        'tmn_presence': result['tmn_presence'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
