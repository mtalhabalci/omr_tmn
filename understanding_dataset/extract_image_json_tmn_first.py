import json
from pathlib import Path

root = Path(r"c:\projects\yl\omr_copilot")
# Augmented dataset JSON (with TMN categories)
json_path = root/"deepscores_train_tmn_aug.json"
# Images directory under ds2_dense_tmn
images_dir = root/"ds2_dense_tmn"/"images"

# Determine first image filename lexicographically
image_files = sorted(p.name for p in images_dir.glob("*.png"))
if not image_files:
    raise SystemExit("No images found in ds2_dense_tmn/images")
TARGET_FILENAME = image_files[0]

# Output path
output_path = root/"understanding_dataset"/f"{TARGET_FILENAME.replace('.png','')}_ds2_dense_tmn.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])
ann_source = data.get("annotations")
if isinstance(ann_source, dict):
    annotations = ann_source
else:
    annotations = data
categories = data.get("categories", {})

# Find image record
img_rec = None
for img in images:
    if img.get("filename") == TARGET_FILENAME:
        img_rec = img
        break
if img_rec is None:
    raise SystemExit(f"Image not found in JSON: {TARGET_FILENAME}")

ann_ids = img_rec.get("ann_ids", [])
ann_dict = {}
for ann_id in ann_ids:
    key = str(ann_id)
    rec = annotations.get(key)
    if rec is not None:
        ann_dict[key] = rec

used_cat_ids = set()
for rec in ann_dict.values():
    for cid in rec.get("cat_id", []):
        used_cat_ids.add(cid)
cat_subset = {cid: categories.get(cid) for cid in used_cat_ids if cid in categories}

output = {
    "image": img_rec,
    "annotations": ann_dict,
    "categories_subset": cat_subset,
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("written:", output_path)
