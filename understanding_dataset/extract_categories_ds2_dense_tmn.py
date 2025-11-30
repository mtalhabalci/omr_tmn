import json
from pathlib import Path

root = Path(r"c:\projects\yl\omr_copilot")
json_path = root/"ds2_dense_tmn"/"jsonlar"/"deepscores_train.json"
out_path = root/"understanding_dataset"/"categories_ds2_dense_tmn.json"

data = json.loads(json_path.read_text(encoding="utf-8"))
cats = data.get("categories", {})
with out_path.open("w", encoding="utf-8") as f:
    json.dump(cats, f, ensure_ascii=False, indent=2)
print("written:", out_path)
