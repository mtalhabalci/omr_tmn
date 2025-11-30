import json
from pathlib import Path
import argparse

root = Path(r"c:\projects\yl\omr_copilot")
json_path = root/"ds2_dense_tmn"/"jsonlar"/"deepscores_train.json"

ap = argparse.ArgumentParser()
ap.add_argument("--filename", required=True, help="Target image filename in ds2_dense_tmn/images")
ap.add_argument("--out-prefix", default="", help="Optional prefix for output file name")
args = ap.parse_args()
TARGET_FILENAME = args.filename

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])
annotations = data.get("annotations", {})
categories = data.get("categories", {})

img_rec = None
for img in images:
    if img.get("filename") == TARGET_FILENAME:
        img_rec = img
        break
if img_rec is None:
    raise SystemExit(f"Image not found: {TARGET_FILENAME}")

ann_ids = img_rec.get("ann_ids", [])
ann_dict = {}
# Primary: gather listed ann_ids
for ann_id in ann_ids:
    key = str(ann_id)
    rec = annotations.get(key)
    if rec is not None:
        ann_dict[key] = rec

# Secondary: augmentation may have added annotations without updating ann_ids.
# Scan all annotations for matching img_id and add missing ones.
img_id_str = str(img_rec.get("id"))
for a_id, rec in annotations.items():
    try:
        if str(rec.get("img_id")) == img_id_str and a_id not in ann_dict:
            ann_dict[a_id] = rec
    except Exception:
        continue

used_cat_ids = set()
for rec in ann_dict.values():
    for cid in rec.get("cat_id", []):
        try:
            used_cat_ids.add(int(cid))
        except Exception:
            pass
cat_subset = {str(cid): categories.get(str(cid)) for cid in used_cat_ids if str(cid) in categories}

output = {
    "image": img_rec,
    "annotations": ann_dict,
    "categories_subset": cat_subset,
}

out_name = (args.out_prefix + TARGET_FILENAME.replace('.png','') + "_subset.json")
out_path = root/"understanding_dataset"/out_name
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("written:", out_path)
