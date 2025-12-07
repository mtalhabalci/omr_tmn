import argparse
import json
import os
from collections import Counter
from pathlib import Path


def find_json_files(root: Path):
    json_dir = root / "jsonlar"
    if json_dir.is_dir():
        files = sorted(json_dir.glob("*.json"))
        if files:
            return files
    # Fallback: search a bit wider (shallow to avoid huge scans)
    files = sorted(root.glob("*.json"))
    if files:
        return files
    return sorted(root.rglob("*.json"))


def load_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e)}


def build_category_summary(json_files, limit=None):
    candidate_name_keys = [
        "category_name", "cat_name", "name", "class_name", "class",
        "symbol", "label", "semantic", "semantic_name", "cat", "category",
    ]

    counts = Counter()
    ann_name_hits = {}
    names_from_categories = {}
    annotation_keys_union = set()

    scanned = []
    for i, p in enumerate(json_files):
        if limit is not None and i >= limit:
            break
        data = load_json(p)
        scanned.append(p.name)
        if "__error__" in data:
            continue

        cats = data.get("categories") or []
        if isinstance(cats, dict):
            cats = list(cats.values())
        for c in cats:
            cid = c.get("id")
            try:
                cid = int(cid)
            except Exception:
                continue
            nm = c.get("name") or c.get("category_name")
            if nm and cid not in names_from_categories:
                names_from_categories[cid] = str(nm)

        anns = data.get("annotations") or []
        if isinstance(anns, dict):
            anns = list(anns.values())

        for a in anns:
            if not isinstance(a, dict):
                continue
            annotation_keys_union.update(a.keys())
            cid = None
            cid_val = a.get("cat_id") or a.get("category_id")
            # cat_id/category_id bazen liste olabilir -> ilk öğeyi al
            if isinstance(cid_val, list) and cid_val:
                try:
                    cid = int(cid_val[0])
                except Exception:
                    cid = None
            else:
                try:
                    cid = int(cid_val)
                except Exception:
                    cid = None
            if cid is None:
                continue
            counts[cid] += 1
            if cid not in ann_name_hits:
                for k in candidate_name_keys:
                    v = a.get(k)
                    if isinstance(v, str) and v:
                        ann_name_hits[cid] = v
                        break

    # Merge names: prefer categories[] names, fallback to annotation-derived
    name_by_cid = dict(names_from_categories)
    for cid, nm in ann_name_hits.items():
        name_by_cid.setdefault(cid, nm)

    items = []
    for cid, cnt in counts.most_common():
        items.append({
            "category_id": int(cid),
            "count": int(cnt),
            "name": name_by_cid.get(cid),
        })

    summary = {
        "files_scanned": scanned,
        "total_categories": len({int(k) for k in counts.keys()}),
        "with_names": sum(1 for cid in counts.keys() if cid in name_by_cid),
        "annotation_keys_union": sorted(list(annotation_keys_union))[:200],
        "items": items,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize categories in ds2_dense_tmn locally")
    default_root = (Path(__file__).resolve().parents[1] / "ds2_dense_tmn")
    parser.add_argument("--root", type=Path, default=default_root,
                        help="Path to ds2_dense_tmn root (default: repo_root/ds2_dense_tmn)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "understanding_dataset" / "categories_ds2_dense_tmn.json",
                        help="Output JSON path for the summary")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of JSON files to scan (None for all)")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        print({"error": f"root not found: {str(root)}"})
        raise SystemExit(2)

    json_files = find_json_files(root)
    if not json_files:
        print({"root": str(root), "json_count": 0})
        raise SystemExit(1)

    print({"root": str(root), "json_count": len(json_files)})
    summary = build_category_summary(json_files, limit=args.limit if args.limit and args.limit > 0 else None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console preview
    print("saved:", str(args.out))
    print("categories:", summary.get("total_categories"))
    print("with_names:", summary.get("with_names"))
    print("top items (first 25):")
    for it in summary.get("items", [])[:25]:
        print(" ", it)


if __name__ == "__main__":
    main()
