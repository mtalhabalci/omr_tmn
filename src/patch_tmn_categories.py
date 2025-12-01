import json, os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_JSON = os.path.join(BASE_DIR, 'ds2_dense_tmn', 'jsonlar', 'deepscores_train.json')
TMN_CATS = [
    (209,'tmn_1_bemol'), (210,'tmn_1_diyez'), (211,'tmn_4_bemol'), (212,'tmn_5_diyez'),
    (213,'tmn_8_bemol'), (214,'tmn_8_diyez'), (215,'tmn_9_bemol'), (216,'tmn_9_diyez')
]

if not os.path.isfile(OUT_JSON):
    raise SystemExit(f'Not found: {OUT_JSON}')

with open(OUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

cats = data.get('categories') or {}
if not isinstance(cats, dict):
    cats = {}

changed = 0
for cid, name in TMN_CATS:
    k = str(cid)
    entry = cats.get(k) or {}
    # Preserve existing name; if missing, set it
    if 'name' not in entry:
        entry['name'] = name
        changed += 1
    # Add or normalize annotation_set
    if entry.get('annotation_set') != 'tmn':
        entry['annotation_set'] = 'tmn'
        changed += 1
    # Add color using identity to palette index conventions
    if entry.get('color') != cid:
        entry['color'] = cid
        changed += 1
    cats[k] = entry

if changed:
    data['categories'] = cats
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

print({'updated_entries': changed, 'out_json': OUT_JSON})
