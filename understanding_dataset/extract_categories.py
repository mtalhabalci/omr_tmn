import json
from pathlib import Path

root = Path(r"c:\projects\yl\omr_copilot")
json_path = root/"ds2_dense"/"deepscores_train.json"
out_path = root/"understanding_dataset"/"categories.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

cats = data.get("categories", {})

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cats, f, ensure_ascii=False, indent=2)

print("written:", out_path)
