import os, json, argparse
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_JSON = os.path.join(BASE_DIR, 'ds2_dense', 'deepscores_train.json')
IMAGES_DIR = os.path.join(BASE_DIR, 'ds2_dense', 'images')
OUT_DIR = os.path.join(BASE_DIR, 'tmn_debug_bbox')
os.makedirs(OUT_DIR, exist_ok=True)

H_MARGIN = 2
MAX_VSHIFT = 12

ACC_NAT_NAMES = {
    'accidentalsharp','accidentalsharpsmall','accidentaldoublesharp',
    'accidentalflat','accidentaldoubleflat',
    'accidentalnatural','accidentalnaturalsmall'
}
STEM_NAMES = {'stem'}
def is_flag_name(n: str) -> bool:
    n = (n or '').lower()
    return n.startswith('flag') or 'flag' in n

def median(vals):
    if not vals:
        return 0
    s = sorted(vals)
    return s[len(s)//2]

def bw(b):
    return max(1, int(b[2]-b[0]))

def bh(b):
    return max(1, int(b[3]-b[1]))

def overlap(a,b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

def collides_any(rect, boxes):
    for ob in boxes:
        if overlap(rect, ob):
            return True
    return False

def load_dataset():
    with open(DATASET_JSON,'r',encoding='utf-8') as f:
        data = json.load(f)
    images = data.get('images') or []
    if isinstance(images, dict):
        images = list(images.values())
    cats = data.get('categories') or {}
    id2name = {}
    if isinstance(cats, dict):
        for cid, meta in cats.items():
            try:
                cid_int = int(cid)
            except Exception:
                continue
            name = meta.get('name') if isinstance(meta, dict) else None
            if isinstance(name, str):
                id2name[cid_int] = name.lower()
    anns = data.get('annotations') or []
    if isinstance(anns, dict):
        anns = list(anns.values())
    return images, id2name, anns

def collect_for_image(filename, images, anns, id2name):
    img_id = None
    for img in images:
        fname = img.get('filename') or img.get('file_name')
        if fname == filename:
            img_id = int(img.get('id'))
            break
    if img_id is None:
        raise SystemExit(f'Image not found: {filename}')
    items = []
    for a in anns:
        a_img_id = a.get('img_id') or a.get('image_id')
        try:
            aid = int(a_img_id)
        except Exception:
            continue
        if aid != img_id:
            continue
        bbox = a.get('a_bbox') or a.get('bbox')
        cats = a.get('cat_id') or a.get('category_ids') or []
        ids=[]
        if isinstance(cats, (list,tuple)):
            for c in cats:
                try:
                    ids.append(int(c))
                except Exception:
                    pass
        elif isinstance(cats,str):
            try:
                ids.append(int(cats))
            except Exception:
                pass
        names = [id2name.get(ci,str(ci)).lower() for ci in ids]
        if bbox and len(bbox)>=4:
            items.append({'bbox': tuple(bbox), 'names': names})
    return img_id, items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--filename', required=True)
    args = ap.parse_args()

    images, id2name, anns = load_dataset()
    img_id, items = collect_for_image(args.filename, images, anns, id2name)

    img_path = os.path.join(IMAGES_DIR, args.filename)
    base = Image.open(img_path).convert('RGBA')
    overlay = base.copy()
    d = ImageDraw.Draw(overlay)

    # Collect boxes
    noteheads, sharps, accnat = [], [], []
    stems, flags = [], []
    for it in items:
        b = it['bbox']
        nl = [n.lower() for n in it['names'] if isinstance(n,str)]
        if any('notehead' in n for n in nl):
            noteheads.append(b)
        elif any(n in ('accidentalsharp','accidentalsharpsmall','accidentaldoublesharp') for n in nl):
            sharps.append(b)
        if any(n in ACC_NAT_NAMES for n in nl):
            accnat.append(b)
        if any(n in STEM_NAMES for n in nl):
            stems.append(b)
        if any(is_flag_name(n) for n in nl):
            flags.append(b)

    # Slot size from this page's sharps
    slot_w = median([bw(s) for s in sharps])
    slot_h = median([bh(s) for s in sharps])
    if slot_w == 0 or slot_h == 0:
        nh_h_med = median([bh(nh) for nh in noteheads])
        if slot_h == 0:
            slot_h = max(6, int(0.85 * nh_h_med))
        if slot_w == 0:
            slot_w = max(6, int(0.7 * slot_h))

    free=0; blocked=0
    chosen_slots = []
    for nh in noteheads:
        x1n,y1n,x2n,y2n = nh
        sw, sh = int(slot_w), int(slot_h)
        if sw<=0 or sh<=0:
            continue
        x2 = int(x1n) - H_MARGIN
        total_w = sw + 2*H_MARGIN
        x1 = x2 - total_w
        if x1 < 0:
            continue
        yc = int((y1n+y2n)/2 - sh/2)
        y_candidates = [yc] + [yc-d for d in range(1,MAX_VSHIFT+1)] + [yc+d for d in range(1,MAX_VSHIFT+1)]
        chosen=None
        for y1 in y_candidates:
            y2 = y1 + sh
            rect = (x1,y1,x2,y2)
            # No structural obstacles; block accidental/natural, stem, and flags
            if collides_any(rect, accnat):
                continue
            # New rule: slot cannot overlap ANY notehead
            if collides_any(rect, noteheads):
                continue
            # New rule: slot cannot overlap stem
            if collides_any(rect, stems):
                continue
            # New rule: slot cannot overlap any flag category
            if collides_any(rect, flags):
                continue
            # New rule: slot cannot overlap previously chosen slots
            if collides_any(rect, chosen_slots):
                continue
            chosen = rect
            break
        if chosen:
            # paint blue box
            d.rectangle([(chosen[0],chosen[1]),(chosen[2],chosen[3])], outline=(0,90,200,255), width=2, fill=(30,144,255,100))
            chosen_slots.append(chosen)
            free += 1
        else:
            blocked += 1

    out_path = os.path.join(OUT_DIR, f'slots_blue_{os.path.basename(args.filename)}')
    overlay.save(out_path)
    print({'filename': args.filename, 'slot_size': (slot_w,slot_h), 'free_slots': free, 'blocked_slots': blocked, 'output': out_path})

if __name__=='__main__':
    main()
