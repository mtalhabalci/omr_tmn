import os, json, argparse
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_JSON = os.path.join(BASE_DIR, 'ds2_dense', 'deepscores_train.json')
IMAGES_DIR = os.path.join(BASE_DIR, 'ds2_dense', 'images')
OUT_DIR = os.path.join(BASE_DIR, 'bbox_slot_tmn')
os.makedirs(OUT_DIR, exist_ok=True)

H_MARGIN = 2
MAX_VSHIFT = 12

ACC_NAT_NAMES = {
    'accidentalflat','accidentaldoubleflat',
    'accidentalnatural','accidentalnaturalsmall',
    'accidentalsharp','accidentalsharpsmall','accidentaldoublesharp'
}
STEM_NAMES = {'stem'}

def is_flag_name(n: str) -> bool:
    n = (n or '').lower()
    return n.startswith('flag') or 'flag' in n

def is_rest_name(n: str) -> bool:
    n = (n or '').lower()
    return n.startswith('rest') or 'rest' in n

def is_augmentation_dot(n: str) -> bool:
    n = (n or '').lower()
    # Augmentation dot (note value extender), avoid matching repeatDot
    return 'augmentationdot' in n

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

def collect_for_image(image_obj, anns, id2name):
    fname = image_obj.get('filename') or image_obj.get('file_name')
    img_id = int(image_obj.get('id'))
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
    return fname, img_id, items


def process_image(fname, items, static_slot=None):
    img_path = os.path.join(IMAGES_DIR, fname)
    base = Image.open(img_path).convert('RGBA')
    overlay = base.copy()
    d = ImageDraw.Draw(overlay)

    noteheads, sharps, accnat, stems, flags, rests, augdots = [], [], [], [], [], [], []
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
        if any(is_rest_name(n) for n in nl):
            rests.append(b)
        if any(is_augmentation_dot(n) for n in nl):
            augdots.append(b)

    # slot size: static overrides, else from page sharps with fallback
    if static_slot and static_slot[0] and static_slot[1]:
        slot_w = max(1, int(static_slot[0]))
        slot_h = max(1, int(static_slot[1]))
    else:
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
            # block accidental/natural, any notehead, stem, flags, rests, augmentation dots, and chosen slots
            if collides_any(rect, accnat):
                continue
            if collides_any(rect, noteheads):
                continue
            if collides_any(rect, stems):
                continue
            if collides_any(rect, flags):
                continue
            if collides_any(rect, rests):
                continue
            if collides_any(rect, augdots):
                continue
            if collides_any(rect, chosen_slots):
                continue
            chosen = rect
            break
        if chosen:
            d.rectangle([(chosen[0],chosen[1]),(chosen[2],chosen[3])], outline=(0,90,200,255), width=2, fill=(30,144,255,100))
            chosen_slots.append(chosen)
            free += 1
        else:
            blocked += 1

    out_path = os.path.join(OUT_DIR, f'slots_blue_{os.path.basename(fname)}')
    overlay.save(out_path)
    return free, blocked, out_path, (slot_w, slot_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=20, help='Islenecek goruntu sayisi (train)')
    ap.add_argument('--slot-w', type=int, default=None, help='Statik slot genisligi (px)')
    ap.add_argument('--slot-h', type=int, default=None, help='Statik slot yuksekligi (px)')
    ap.add_argument('--offset', type=int, default=0, help='Gorsel baslangic ofseti (train)')
    args = ap.parse_args()

    images, id2name, anns = load_dataset()
    start = max(0, int(args.offset))
    end = start + args.limit if args.limit > 0 else None
    images = images[start:end]

    total_free=0; total_blocked=0
    static_slot = (args.slot_w, args.slot_h) if (args.slot_w and args.slot_h) else None
    for img in images:
        fname, img_id, items = collect_for_image(img, anns, id2name)
        free, blocked, out_path, slot_size = process_image(fname, items, static_slot=static_slot)
        total_free += free
        total_blocked += blocked
        print({'filename': fname, 'slot_size': slot_size, 'free_slots': free, 'blocked_slots': blocked, 'output': out_path})

    print({'batch_free_slots': total_free, 'batch_blocked_slots': total_blocked, 'out_dir': OUT_DIR})

if __name__=='__main__':
    main()
