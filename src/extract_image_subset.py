import os, json, sys
from typing import Dict, Any

BASE = os.path.dirname(os.path.dirname(__file__))
FULL_JSON = os.path.join(BASE, 'ds2_dense_tmn', 'jsonlar', 'deepscores_train.json')
OUT_DIR = os.path.join(BASE, 'understanding_dataset')


def extract_subset(filename: str) -> Dict[str, Any]:
    with open(FULL_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images = data.get('images')
    anns = data.get('annotations')
    cats = data.get('categories')
    # Normalize collections to dict/list
    if isinstance(images, dict):
        images_list = list(images.values())
    else:
        images_list = images or []
    if isinstance(anns, dict):
        anns_dict = anns
    else:
        anns_dict = {str(i): a for i, a in enumerate(anns or [])}
    # Find image
    target_img = None
    for img in images_list:
        if img.get('filename') == filename:
            target_img = img
            break
    if not target_img:
        raise RuntimeError(f'Image not found: {filename}')
    # Collect annotations for this image
    img_id = str(target_img.get('id')) if target_img.get('id') is not None else None
    selected_anns: Dict[str, Any] = {}
    for aid, a in anns_dict.items():
        a_img_id = a.get('img_id') or a.get('image_id')
        if str(a_img_id) == img_id:
            selected_anns[str(aid)] = a
    subset = {
        'images': [target_img],
        'annotations': selected_anns,
        'categories': cats,
    }
    return subset


def main():
    if len(sys.argv) < 2:
        print('Usage: extract_image_subset.py <filename>')
        sys.exit(1)
    filename = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)
    subset = extract_subset(filename)
    out_path = os.path.join(OUT_DIR, f'{os.path.splitext(filename)[0]}_subset.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False)
    # Also write categories standalone
    cats_out = os.path.join(OUT_DIR, 'categories.json')
    with open(cats_out, 'w', encoding='utf-8') as f:
        json.dump(subset['categories'], f, ensure_ascii=False)
    print({'subset_json': out_path, 'categories_json': cats_out, 'images_count': 1, 'annotations_count': len(subset['annotations'])})


if __name__ == '__main__':
    main()
