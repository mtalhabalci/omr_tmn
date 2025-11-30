import json
from pathlib import Path

root = Path(r"c:\projects\yl\omr_copilot")
json_path = root/"ds2_dense"/"deepscores_train.json"
output_path = root/"understanding_dataset"/"lg-2267728-aug-beethoven--page-2.json"
TARGET_FILENAME = "lg-2267728-aug-beethoven--page-2.png"

# Load JSON (it may be large)
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])
# Anotasyonlar ya top-level'da olabilir ya da data["annotations"] altında.
ann_source = data.get("annotations")
if isinstance(ann_source, dict):
    annotations = ann_source
else:
    annotations = data
categories = data.get("categories", {})

# Find the image record
img_rec = None
for img in images:
    if img.get("filename") == TARGET_FILENAME:
        img_rec = img
        break

if img_rec is None:
    raise SystemExit(f"Image not found: {TARGET_FILENAME}")

# Collect annotations by ann_ids
ann_ids = img_rec.get("ann_ids", [])
ann_dict = {}
for ann_id in ann_ids:
    # Anahtarları stringleştirerek güvence altına al
    key = str(ann_id)
    rec = annotations.get(key)
    if rec is not None:
        ann_dict[key] = rec

# Build output: include image record, annotations, and optionally category subset used
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
