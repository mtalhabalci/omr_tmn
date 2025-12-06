import os, json, argparse, random
import glob
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Default single JSON (legacy). Can be overridden with args.
DATASET_JSON = os.path.join(BASE_DIR, 'ds2_dense', 'deepscores_train.json')
IMAGES_DIR = os.path.join(BASE_DIR, 'ds2_dense', 'images')
SYMBOLS_DIR = os.path.join(BASE_DIR, 'tmn_symbols_png')
OUT_ROOT = os.path.join(BASE_DIR, 'ds2_dense_tmn')
OUT_IMAGES = os.path.join(OUT_ROOT, 'images')
OUT_JSONLAR = os.path.join(OUT_ROOT, 'jsonlar')
OUT_SEG = os.path.join(OUT_ROOT, 'segmentation')
OUT_INST = os.path.join(OUT_ROOT, 'instance')
for _p in (OUT_IMAGES, OUT_JSONLAR, OUT_SEG, OUT_INST):
    os.makedirs(_p, exist_ok=True)
PROGRESS_LOG = os.path.join(OUT_JSONLAR, 'progress.jsonl')

H_MARGIN = 2
MAX_VSHIFT = 12

ACC_NAT_NAMES = {
    'accidentalflat','accidentaldoubleflat',
    'accidentalnatural','accidentalnaturalsmall',
    'accidentalsharp','accidentalsharpsmall','accidentaldoublesharp'
}

TMN_CATS = [
    (209,'tmn_1_bemol'), (210,'tmn_1_diyez'), (211,'tmn_4_bemol'), (212,'tmn_5_diyez'),
    (213,'tmn_8_bemol'), (214,'tmn_8_diyez'), (215,'tmn_9_bemol'), (216,'tmn_9_diyez')
]


def erode_mask(alpha_img: Image.Image, iterations: int = 1) -> Image.Image:
    """Erode a binary mask (alpha>0) by 1-pixel 8-neighborhood, repeated iterations times.
    Returns an 'L' image with values 0 or 255.
    """
    src = alpha_img.convert('L')
    # Binarize
    w, h = src.size
    src_px = src.load()
    for y in range(h):
        for x in range(w):
            src_px[x,y] = 255 if src_px[x,y] > 0 else 0
    cur = src
    for _ in range(max(1, int(iterations))):
        nxt = Image.new('L', (w,h), 0)
        np = nxt.load()
        cp = cur.load()
        for y in range(1, h-1):
            for x in range(1, w-1):
                # Keep pixel only if all neighbors are foreground (basic erosion)
                all_fg = True
                for yy in (-1,0,1):
                    for xx in (-1,0,1):
                        if cp[x+xx, y+yy] == 0:
                            all_fg = False
                            break
                    if not all_fg:
                        break
                np[x,y] = 255 if all_fg else 0
        # border set to 0
        cur = nxt
    return cur

def make_binary_mask(alpha_img: Image.Image, threshold: int = 64) -> Image.Image:
    """Convert alpha to a binary mask with given threshold (0-255)."""
    src = alpha_img.convert('L')
    w, h = src.size
    sp = src.load()
    t = max(0, min(255, int(threshold)))
    for y in range(h):
        for x in range(w):
            sp[x, y] = 255 if sp[x, y] >= t else 0
    return src

def dilate_mask(bin_mask: Image.Image, iterations: int = 1) -> Image.Image:
    """Dilate a binary mask (values {0,255}) by 1-pixel 8-neighborhood, repeated iterations times."""
    w, h = bin_mask.size
    cur = bin_mask.convert('L')
    for _ in range(max(1, int(iterations))):
        nxt = Image.new('L', (w,h), 0)
        np = nxt.load()
        cp = cur.load()
        for y in range(1, h-1):
            for x in range(1, w-1):
                any_fg = False
                for yy in (-1,0,1):
                    for xx in (-1,0,1):
                        if cp[x+xx, y+yy] >= 128:
                            any_fg = True
                            break
                    if any_fg:
                        break
                np[x,y] = 255 if any_fg else 0
        cur = nxt
    return cur


def is_flag_name(n: str) -> bool:
    n = (n or '').lower()
    return n.startswith('flag') or 'flag' in n

def is_rest_name(n: str) -> bool:
    n = (n or '').lower()
    return n.startswith('rest') or 'rest' in n

def is_augmentation_dot(n: str) -> bool:
    n = (n or '').lower()
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


def load_dataset(json_path: str = None, json_glob: str = None):
    """Load dataset from a single JSON or merge multiple shard JSONs.
    - json_path: explicit JSON file to load
    - json_glob: glob pattern relative to BASE_DIR, e.g., 'ds2_dense/deepscores-complete-*_train.json'
    Merging policy:
      images: concatenated unique by filename (first occurrence kept)
      annotations: concatenated list
      categories: last shard wins per id
    """
    datas = []
    shards = []  # list of tuples (path, data)
    if json_path and os.path.isfile(json_path):
        with open(json_path,'r',encoding='utf-8') as f:
            d = json.load(f)
            datas.append(d)
            shards.append((json_path, d))
    elif json_glob:
        # Support absolute or relative glob
        hits = glob.glob(json_glob if os.path.isabs(json_glob) else os.path.join(BASE_DIR, json_glob))
        hits.sort()
        for jp in hits:
            try:
                with open(jp,'r',encoding='utf-8') as f:
                    d = json.load(f)
                    datas.append(d)
                    shards.append((jp, d))
            except Exception:
                pass
    else:
        # fallbacks: train + test in root
        for cand in ['ds2_dense/deepscores_train.json','ds2_dense/deepscores_test.json']:
            p = os.path.join(BASE_DIR, cand)
            if os.path.isfile(p):
                with open(p,'r',encoding='utf-8') as f:
                    d = json.load(f)
                    datas.append(d)
                    shards.append((p, d))
    if not datas:
        # final fallback to legacy constant
        with open(DATASET_JSON,'r',encoding='utf-8') as f:
            d = json.load(f)
            datas.append(d)
            shards.append((DATASET_JSON, d))
    images_out = []
    seen = set()
    anns_out = []
    cats_out = {}
    for data in datas:
        imgs = data.get('images') or []
        if isinstance(imgs, dict):
            imgs = list(imgs.values())
        for im in imgs:
            fn = im.get('filename') or im.get('file_name')
            if fn and fn not in seen:
                images_out.append(im)
                seen.add(fn)
        anns = data.get('annotations') or []
        if isinstance(anns, dict):
            anns = list(anns.values())
        anns_out.extend(anns)
        cats = data.get('categories') or {}
        for k, v in cats.items():
            cats_out[str(k)] = v
    id2name = {}
    for k, v in cats_out.items():
        try:
            cid_int = int(k)
        except Exception:
            continue
        name = v.get('name') if isinstance(v, dict) else None
        if isinstance(name, str):
            id2name[cid_int] = name.lower()
    merged = {'images': images_out, 'annotations': anns_out, 'categories': cats_out}

    # Build per-shard membership by filename
    shard_membership = []  # list of sets of filenames per shard
    for (_, d) in shards:
        imgs = d.get('images') or []
        if isinstance(imgs, dict):
            imgs = list(imgs.values())
        shard_membership.append(set([im.get('filename') or im.get('file_name') for im in imgs if (im.get('filename') or im.get('file_name'))]))

    return merged, images_out, id2name, anns_out, shards, shard_membership


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


def ensure_tmn_categories(data):
    cats = data.get('categories') or {}
    if not isinstance(cats, dict):
        cats = {}
    for cid, name in TMN_CATS:
        k = str(cid)
        entry = cats.get(k) or {}
        # Always ensure name is present
        if 'name' not in entry:
            entry['name'] = name
        # Add annotation_set marker for TMN categories
        if entry.get('annotation_set') != 'tmn':
            entry['annotation_set'] = 'tmn'
        # Use identity color to align with segmentation palette mapping
        if entry.get('color') != cid:
            entry['color'] = cid
        cats[k] = entry
    data['categories'] = cats

def load_barline_mask_for_image(fname: str, barline_rgb_hex: str):
    """Load segmentation image for fname and build a binary mask (L) for the given barline color.
    Returns an 'L' image with 0/255, or None if segmentation not found or color not present.
    """
    try_paths = []
    # ds2_dense (original)
    try_paths.append(os.path.join(os.path.dirname(IMAGES_DIR), 'segmentation', fname.replace('.png','_seg.png')))
    # ds2_dense_tmn (augmented)
    try_paths.append(os.path.join(OUT_SEG, fname.replace('.png','_seg.png')))
    seg_path = None
    for p in try_paths:
        if os.path.isfile(p):
            seg_path = p
            break
    if not seg_path:
        return None
    im = Image.open(seg_path)
    # parse hex
    s = barline_rgb_hex.strip()
    if s.startswith('#'):
        s = s[1:]
    try:
        target_rgb = (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16))
    except Exception:
        return None
    # determine palette index for target_rgb
    idx = None
    if im.mode == 'P':
        pal = im.getpalette()
        if pal:
            for i in range(256):
                base = i*3
                rgb = (pal[base], pal[base+1], pal[base+2])
                if rgb == target_rgb:
                    idx = i
                    break
    # build mask
    w,h = im.size
    mask = Image.new('L', (w,h), 0)
    mp = mask.load()
    px = im.load()
    if im.mode == 'P' and idx is not None:
        for y in range(h):
            for x in range(w):
                mp[x,y] = 255 if px[x,y] == idx else 0
    else:
        # fallback: compare RGB values directly
        rgb_im = im.convert('RGB')
        rp = rgb_im.load()
        for y in range(h):
            for x in range(w):
                mp[x,y] = 255 if rp[x,y] == target_rgb else 0
    return mask

def intersects_mask_at(sx: int, sy: int, smask: Image.Image, other_mask: Image.Image) -> bool:
    """Check if placing smask at (sx,sy) overlaps other_mask (both 'L' 0/255)."""
    if other_mask is None:
        return False
    a_px = smask.load()
    o_px = other_mask.load()
    w,h = smask.size
    W,H = other_mask.size
    for yy in range(h):
        oy = sy + yy
        if oy < 0 or oy >= H:
            continue
        for xx in range(w):
            ox = sx + xx
            if ox < 0 or ox >= W:
                continue
            if a_px[xx,yy] > 0 and o_px[ox,oy] > 0:
                return True
    return False


def symbol_paths():
    paths = [os.path.join(SYMBOLS_DIR, n) for n in sorted(os.listdir(SYMBOLS_DIR)) if n.lower().endswith('.png')]
    if not paths:
        raise SystemExit(f'No PNG symbols found in {SYMBOLS_DIR}')
    return paths

SYM_TO_CAT = {
    '1_bemol.png': 209,
    '1_diyez.png': 210,
    '4_bemol.png': 211,
    '5_diyez.png': 212,
    '8_bemol.png': 213,
    '8_diyez.png': 214,
    '9_bemol.png': 215,
    '9_diyez.png': 216,
}


def process_one(fname, img_id, items, slot_w, slot_h, syms, anns_out_start_id, alpha_th, erode_iter, dilate_iter, save_image=True):
    img_path = os.path.join(IMAGES_DIR, fname)
    if not os.path.isfile(img_path):
        return 0, anns_out_start_id, None
    base = Image.open(img_path).convert('RGBA')
    canvas = base.copy()

    noteheads, accnat, stems, flags, rests, augdots = [], [], [], [], [], []
    for it in items:
        b = it['bbox']
        nl = [n.lower() for n in it['names'] if isinstance(n,str)]
        if any('notehead' in n for n in nl):
            noteheads.append(b)
        if any(n in ACC_NAT_NAMES for n in nl):
            accnat.append(b)
        if any(n == 'stem' for n in nl):
            stems.append(b)
        if any(is_flag_name(n) for n in nl):
            flags.append(b)
        if any(is_rest_name(n) for n in nl):
            rests.append(b)
        if any(is_augmentation_dot(n) for n in nl):
            augdots.append(b)

    placed = 0
    obstacles_dyn = []  # placed TMNs to avoid slot-slot
    anns_new = []
    new_ids = []  # track new annotation ids generated (assigned later)
    tmn_rects = []  # (rect, cat_id) for later mask overlays

    # Optional barline mask (from segmentation) using color #00acc6
    bar_mask = load_barline_mask_for_image(fname, '#00acc6')

    for idx, nh in enumerate(noteheads):
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
            # block accidental/natural, any notehead, stem, flags, rests, augdots, and placed TMNs
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
            if collides_any(rect, obstacles_dyn):
                continue
            chosen = rect
            break
        if not chosen:
            continue
        # paste TMN symbol scaled to chosen rect (inner fill)
        ix1,iy1,ix2,iy2 = chosen
        avail_w = max(1, ix2-ix1)
        avail_h = max(1, iy2-iy1)
        sym_path = syms[placed % len(syms)]
        sym = Image.open(sym_path).convert('RGBA')
        sw0, sh0 = sym.size
        scale = min(avail_w / sw0, avail_h / sh0)
        new_w = max(1, int(sw0 * scale))
        new_h = max(1, int(sh0 * scale))
        sym_resized = sym.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Use original symbol pixels as-is (no RGB/alpha normalization) for images
        # Build binary mask (threshold) and optionally erode/dilate for visual thickness control
        alpha = sym_resized.getchannel('A')
        mask = make_binary_mask(alpha, threshold=alpha_th)
        if erode_iter > 0:
            mask = erode_mask(mask, iterations=erode_iter)
        if dilate_iter > 0:
            mask = dilate_mask(mask, iterations=dilate_iter)
        px = ix1 + (avail_w - new_w)//2
        py = iy1 + (avail_h - new_h)//2
        # If overlaps barline mask, skip placement
        if bar_mask is not None and intersects_mask_at(px, py, mask, bar_mask):
            continue
        # Images: place the symbol directly with natural transparency
        canvas.alpha_composite(sym_resized, (px, py))
        new_rect = (px, py, px+new_w, py+new_h)
        obstacles_dyn.append(new_rect)
        # add annotation
        cat_id = SYM_TO_CAT.get(os.path.basename(sym_path))
        if cat_id:
            area = max(1, int((new_rect[2]-new_rect[0])*(new_rect[3]-new_rect[1])))
            x1, y1, x2, y2 = new_rect[0], new_rect[1], new_rect[2], new_rect[3]
            # o_bbox polygon uses same rectangle corners in order similar to existing dataset examples (top-right, top-left, bottom-left, bottom-right)
            o_poly = [float(x2), float(y1), float(x1), float(y1), float(x1), float(y2), float(x2), float(y2)]
            anns_new.append({
                'a_bbox': [float(x1), float(y1), float(x2), float(y2)],
                'o_bbox': o_poly,
                'cat_id': [str(cat_id)],
                'area': area,
                'img_id': str(img_id),
                'comments': ''
            })
            # Keep eroded mask and position for mask overlays
            tmn_rects.append(((px, py, mask), cat_id))
        placed += 1

    if placed > 0 and save_image:
        out_img = os.path.join(OUT_IMAGES, fname)
        canvas.save(out_img)
    # return new annotations to be merged later
    next_id = anns_out_start_id
    anns_dict_updates = {}
    for a in anns_new:
        ann_id_str = str(next_id)
        anns_dict_updates[ann_id_str] = a
        new_ids.append(ann_id_str)
        next_id += 1
    return placed, next_id, anns_dict_updates, new_ids, tmn_rects


def extend_palette(im, needed_indices):
    """Ensure palette indices exist with distinct colors for segmentation mask."""
    if im.mode != 'P':
        return im
    pal = im.getpalette()  # list of 768 ints (256*3)
    if pal is None:
        # initialize palette with zeros
        pal = [0]*768
    # assign colors for each needed index if not already non-zero
    for idx in needed_indices:
        if idx < 0 or idx > 255:
            continue
        base = idx*3
        # if color already set (non-zero triple), skip
        if pal[base] or pal[base+1] or pal[base+2]:
            continue
        # generate deterministic bright color
        r = (idx*53) % 256
        g = (idx*97) % 256
        b = (idx*193) % 256
        # avoid very dark colors
        if r+g+b < 60:
            r = (r+120) % 256
            g = (g+80) % 256
            b = (b+40) % 256
        pal[base] = r
        pal[base+1] = g
        pal[base+2] = b
    im.putpalette(pal)
    return im

def gen_instance_color(existing):
    """Generate a unique RGBA color not in existing set (avoid transparent)."""
    for _ in range(1000):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        col = (r,g,b,255)
        if col not in existing and (r+g+b) > 30:
            return col
    # fallback
    return (255,0,0,255)

def outputs_status(fname: str):
    """Return simple status booleans for existing outputs of an image."""
    img_ok = os.path.isfile(os.path.join(OUT_IMAGES, fname))
    seg_ok = os.path.isfile(os.path.join(OUT_SEG, fname.replace('.png','_seg.png')))
    inst_ok = os.path.isfile(os.path.join(OUT_INST, fname.replace('.png','_inst.png')))
    return {'image': img_ok, 'seg': seg_ok, 'inst': inst_ok}

def append_progress(rec: dict):
    try:
        with open(PROGRESS_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--slot-w', type=int, default=17)
    ap.add_argument('--slot-h', type=int, default=48)
    ap.add_argument('--only', type=str, default='', help='Comma-separated list of filenames to process (overrides offset/limit)')
    ap.add_argument('--alpha-th', type=int, default=64, help='Alpha threshold (0-255) to binarize symbol mask')
    ap.add_argument('--erode', type=int, default=0, help='Erode iterations for symbol mask (0 disables)')
    ap.add_argument('--dilate', type=int, default=0, help='Dilate iterations for symbol mask (0 disables)')
    ap.add_argument('--feather-radius', type=float, default=0.0, help='Gaussian blur radius for soft mask edges')
    ap.add_argument('--knockout-strength', type=float, default=1.0, help='Strength (0-1) of background darkening under symbol')
    ap.add_argument('--images-only', dest='images_only', action='store_true', help='Generate only composited images; skip segmentation/instance/json updates')
    ap.add_argument('--segmentation-only', dest='segmentation_only', action='store_true', help='Generate only segmentation masks; skip images/instance/json updates')
    ap.add_argument('--instance-only', dest='instance_only', action='store_true', help='Generate only instance masks; skip images/segmentation/json updates')
    ap.add_argument('--checkpoint', type=int, default=100, help='Flush JSON/progress every N processed images (0 disables checkpoint flush)')
    ap.add_argument('--force', action='store_true', help='Force reprocessing even if outputs exist')
    ap.add_argument('--from-fs-missing', dest='from_fs_missing', action='store_true', help='Process only files present in dataset JSON + source images but missing in TMN output images')
    ap.add_argument('--json-path', type=str, default='', help='Explicit dataset JSON file path')
    ap.add_argument('--json-glob', type=str, default='', help='Glob for single/multiple JSONs. Absolute or relative to repo root.')
    ap.add_argument('--images-dir', type=str, default='', help='Override source images directory (defaults to repo ds2_dense/images)')
    ap.add_argument('--out-root', type=str, default='', help='Override output root directory (will create images/segmentation/instance/jsonlar under it)')
    ap.add_argument('--symbols-dir', type=str, default='', help='Override TMN symbols PNG directory')
    ap.add_argument('--json-out-mode', type=str, default='single', choices=['single','per-shard'], help='Write a single merged JSON or one per input shard (when using json-glob)')
    args = ap.parse_args()

    # Apply path overrides early
    global IMAGES_DIR, OUT_ROOT, OUT_IMAGES, OUT_JSONLAR, OUT_SEG, OUT_INST, SYMBOLS_DIR, PROGRESS_LOG
    if args.images_dir:
        IMAGES_DIR = args.images_dir
    if args.out_root:
        OUT_ROOT = args.out_root
        OUT_IMAGES = os.path.join(OUT_ROOT, 'images')
        OUT_JSONLAR = os.path.join(OUT_ROOT, 'jsonlar')
        OUT_SEG = os.path.join(OUT_ROOT, 'segmentation')
        OUT_INST = os.path.join(OUT_ROOT, 'instance')
        for _p in (OUT_IMAGES, OUT_JSONLAR, OUT_SEG, OUT_INST):
            os.makedirs(_p, exist_ok=True)
        PROGRESS_LOG = os.path.join(OUT_JSONLAR, 'progress.jsonl')
    if args.symbols_dir:
        SYMBOLS_DIR = args.symbols_dir

    data, images, id2name, anns, shards, shard_membership = load_dataset(json_path=args.json_path or None, json_glob=args.json_glob or None)
    ensure_tmn_categories(data)
    start = max(0, int(args.offset))
    end = start + args.limit if args.limit > 0 else None
    only_list = []
    if args.only:
        only_list = [x.strip() for x in args.only.split(',') if x.strip()]
    if only_list:
        target_set = set(only_list)
        # preserve order of only_list
        filename_to_img = {img.get('filename'): img for img in images if img.get('filename') in target_set}
        images_sel = [filename_to_img[f] for f in only_list if f in filename_to_img]
    else:
        # Optional mode: compute intersection of JSON images and filesystem source, minus TMN outputs
        if args.from_fs_missing:
            src_dir = IMAGES_DIR
            out_images_dir = OUT_IMAGES
            try:
                fs_names = set([n for n in os.listdir(src_dir) if n.lower().endswith('.png')])
            except Exception:
                fs_names = set()
            try:
                tmn_names = set([n for n in os.listdir(out_images_dir) if n.lower().endswith('.png')])
            except Exception:
                tmn_names = set()
            # Dataset JSON image names
            json_names = [img.get('filename') for img in images if img.get('filename')]
            json_set = set(json_names)
            # candidates = in JSON AND on FS AND not already in TMN images
            candidates = [n for n in json_names if (n in fs_names) and (n not in tmn_names)]
            # Build selection preserving dataset order
            filename_to_img = {img.get('filename'): img for img in images if img.get('filename') in candidates}
            images_sel = [filename_to_img[f] for f in candidates if f in filename_to_img]
            print({'selection_mode': 'from-fs-missing', 'candidates': len(candidates)})
        else:
            images_sel = images[start:end]

    # prepare annotations dict with next id (unless images-only)
    anns_out = data.get('annotations') or {}
    if not isinstance(anns_out, dict):
        # convert list to dict if needed
        anns_out = {str(i): a for i, a in enumerate(anns)}
    try:
        next_id = max(int(k) for k in anns_out.keys()) + 1
    except Exception:
        next_id = 1

    syms = symbol_paths()

    total_placed = 0
    processed_counter = 0
    # Mapping of TMN cat_ids to palette indices (identity mapping to keep dataset consistent)
    CAT_TO_PALETTE = {209:209,210:210,211:211,212:212,213:213,214:214,215:215,216:216}

    for img in images_sel:
        fname, img_id, items = collect_for_image(img, anns, id2name)
        # Resume/idempotent skip: if outputs already exist, skip unless forced
        st = outputs_status(fname)
        if not args.force:
            # Determine required outputs based on mode
            needs_image = not args.segmentation_only and not args.instance_only
            needs_seg = not args.images_only and not args.instance_only
            needs_inst = not args.images_only and not args.segmentation_only
            already_done = True
            if needs_image:
                already_done = already_done and st['image']
            if needs_seg:
                already_done = already_done and st['seg']
            if needs_inst:
                already_done = already_done and st['inst']
            if already_done:
                print({'filename': fname, 'skipped': True, 'reason': 'outputs-exist'})
                continue
        placed, next_id, updates, new_ids, tmn_rects = process_one(
            fname, img_id, items,
            args.slot_w, args.slot_h,
            syms, next_id,
            args.alpha_th, args.erode, args.dilate, save_image=(not args.segmentation_only and not args.instance_only)
        )
        total_placed += placed
        processed_counter += 1 if placed > 0 else 0
        # Update annotations/json only if not in any *_only mode
        if not args.images_only and not args.segmentation_only and not args.instance_only:
            anns_out.update(updates)
            # Update ann_ids list for image to include new ids
            if placed > 0:
                ann_ids_list = img.get('ann_ids')
                if not isinstance(ann_ids_list, list):
                    ann_ids_list = []
                ann_ids_list.extend(new_ids)
                img['ann_ids'] = ann_ids_list
        # Append lightweight progress log for resume diagnostics
        append_progress({
            'filename': fname,
            'placed': placed,
            'outputs': st,
            'required': {
                'image': not args.segmentation_only and not args.instance_only,
                'seg': not args.images_only and not args.instance_only,
                'inst': not args.images_only and not args.segmentation_only
            }
        })
        # Generate masks depending on mode
        if placed > 0 and (args.segmentation_only or args.instance_only or not args.images_only):
            # Overlay segmentation and instance masks using actual symbol shapes
            base_seg_path = os.path.join(os.path.dirname(IMAGES_DIR), 'segmentation', fname.replace('.png','_seg.png'))
            base_inst_path = os.path.join(os.path.dirname(IMAGES_DIR), 'instance', fname.replace('.png','_inst.png'))
            out_seg_path = os.path.join(OUT_SEG, fname.replace('.png','_seg.png'))
            out_inst_path = os.path.join(OUT_INST, fname.replace('.png','_inst.png'))
            # Determine output canvas sizes
            img_w, img_h = 0, 0
            try:
                with Image.open(os.path.join(IMAGES_DIR, fname)) as _im:
                    img_w, img_h = _im.size
            except Exception:
                pass
            # Segmentation (fallback to new blank paletted image if base missing)
            if os.path.isfile(base_seg_path):
                seg_im = Image.open(base_seg_path).copy()
            else:
                # create blank paletted segmentation if base not available
                seg_im = Image.new('P', (img_w, img_h), 0)
            # Ensure palette extended
            if not args.instance_only:
                seg_im = extend_palette(seg_im, [CAT_TO_PALETTE[cid] for _, cid in tmn_rects if cid in CAT_TO_PALETTE])
                seg_px = seg_im.load()
                for ((sx, sy, smask), cid) in tmn_rects:
                    pal_idx = CAT_TO_PALETTE.get(cid)
                    if pal_idx is None:
                        continue
                    # Use eroded mask to set segmentation index
                    a_px = smask.load()
                    w, h = smask.size
                    for yy in range(h):
                        ay = sy + yy
                        if ay < 0 or ay >= seg_im.size[1]:
                            continue
                        for xx in range(w):
                            ax = sx + xx
                            if ax < 0 or ax >= seg_im.size[0]:
                                continue
                            if a_px[xx, yy] > 0:
                                seg_px[ax, ay] = pal_idx
                seg_im.save(out_seg_path)
                if not args.segmentation_only:
                    # Instance (fallback to blank transparent RGBA if base missing)
                    if os.path.isfile(base_inst_path):
                        inst_im = Image.open(base_inst_path).convert('RGBA')
                    else:
                        inst_im = Image.new('RGBA', (img_w, img_h), (0,0,0,0))
                    existing_colors = set(inst_im.getdata())
                    # assign colors per new annotation id (1:1 with tmn_rects order)
                    ann_colors = {}
                    for aid, (rectinfo, cid) in zip(new_ids, tmn_rects):
                        col = gen_instance_color(existing_colors)
                        existing_colors.add(col)
                        ann_colors[aid] = col
                    # Update comments for new annotations with instance color hex (RRGGBB)
                    for aid2, col in ann_colors.items():
                        r, g, b, _ = col
                        hex_str = f"#{r:02x}{g:02x}{b:02x}"
                        if aid2 in updates:
                            cmt = updates[aid2].get('comments') or ''
                            sep = ';' if (cmt and not cmt.endswith(';')) else ''
                            updates[aid2]['comments'] = f"{cmt}{sep}instance:{hex_str};"
                    inst_px = inst_im.load()
                    for aid, ((sx, sy, smask), cid) in zip(new_ids, tmn_rects):
                        color = ann_colors[aid]
                        a_px = smask.load()
                        w, h = smask.size
                        for yy in range(h):
                            ay = sy + yy
                            if ay < 0 or ay >= inst_im.size[1]:
                                continue
                            for xx in range(w):
                                ax = sx + xx
                                if ax < 0 or ax >= inst_im.size[0]:
                                    continue
                                if a_px[xx, yy] > 0:
                                    inst_px[ax, ay] = color
                    inst_im.save(out_inst_path)
        print({
            'filename': fname,
            'placed': placed,
            'new_annotation_ids': new_ids[:5] if new_ids else [],
            'slot_size': (args.slot_w, args.slot_h),
            'out_image': os.path.join(OUT_IMAGES, fname)
        })

        # Checkpoint flush: write JSON incrementally every N processed images
        if processed_counter > 0 and args.checkpoint and (processed_counter % args.checkpoint == 0) and (not args.images_only and not args.segmentation_only and not args.instance_only):
            try:
                if args.json_out_mode == 'per-shard' and shards:
                    # build per-shard outputs
                    imgid_to_fname = {}
                    for im in data.get('images') or []:
                        try:
                            imgid_to_fname[int(im.get('id'))] = im.get('filename') or im.get('file_name')
                        except Exception:
                            continue
                    fname_to_annids = {}
                    for k, v in anns_out.items():
                        try:
                            fid = int(v.get('img_id'))
                        except Exception:
                            continue
                        fn = imgid_to_fname.get(fid)
                        if not fn:
                            continue
                        fname_to_annids.setdefault(fn, []).append(k)
                    for (spath, sdata), sset in zip(shards, shard_membership):
                        sd = dict(sdata)  # shallow copy
                        ensure_tmn_categories(sd)
                        # Filter annotations
                        ann_filtered = {k: v for k, v in anns_out.items() if (imgid_to_fname.get(int(v.get('img_id'))) in sset if v.get('img_id') else False)}
                        sd['annotations'] = ann_filtered
                        # Update images' ann_ids if present
                        imgs = sd.get('images') or []
                        if isinstance(imgs, dict):
                            imgs = list(imgs.values())
                        for im in imgs:
                            fn = im.get('filename') or im.get('file_name')
                            if fn and fn in fname_to_annids:
                                im['ann_ids'] = fname_to_annids[fn]
                        out_name = os.path.basename(spath)
                        out_json_ck = os.path.join(OUT_JSONLAR, out_name)
                        with open(out_json_ck, 'w', encoding='utf-8') as f:
                            json.dump(sd, f, ensure_ascii=False)
                    print({'checkpoint': processed_counter, 'out_json_mode': 'per-shard', 'count': len(shards)})
                else:
                    out_json_ck = os.path.join(OUT_JSONLAR, 'deepscores_train.json')
                    data['annotations'] = anns_out
                    with open(out_json_ck, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False)
                    print({'checkpoint': processed_counter, 'out_json': out_json_ck})
            except Exception:
                pass

    # Write json only if not any *_only mode
    if not args.images_only and not args.segmentation_only and not args.instance_only:
        # Final JSON write
        if args.json_out_mode == 'per-shard' and shards:
            try:
                imgid_to_fname = {}
                for im in data.get('images') or []:
                    try:
                        imgid_to_fname[int(im.get('id'))] = im.get('filename') or im.get('file_name')
                    except Exception:
                        continue
                fname_to_annids = {}
                for k, v in anns_out.items():
                    try:
                        fid = int(v.get('img_id'))
                    except Exception:
                        continue
                    fn = imgid_to_fname.get(fid)
                    if not fn:
                        continue
                    fname_to_annids.setdefault(fn, []).append(k)
                out_paths = []
                for (spath, sdata), sset in zip(shards, shard_membership):
                    sd = dict(sdata)
                    ensure_tmn_categories(sd)
                    ann_filtered = {k: v for k, v in anns_out.items() if (imgid_to_fname.get(int(v.get('img_id'))) in sset if v.get('img_id') else False)}
                    sd['annotations'] = ann_filtered
                    imgs = sd.get('images') or []
                    if isinstance(imgs, dict):
                        imgs = list(imgs.values())
                    for im in imgs:
                        fn = im.get('filename') or im.get('file_name')
                        if fn and fn in fname_to_annids:
                            im['ann_ids'] = fname_to_annids[fn]
                    out_name = os.path.basename(spath)
                    out_json = os.path.join(OUT_JSONLAR, out_name)
                    with open(out_json, 'w', encoding='utf-8') as f:
                        json.dump(sd, f, ensure_ascii=False)
                    out_paths.append(out_json)
                # palette mapping write
                try:
                    um_dir = os.path.join(BASE_DIR, 'understanding_dataset')
                    os.makedirs(um_dir, exist_ok=True)
                    cat_to_palette_out = {}
                    cats = data.get('categories') or {}
                    for k in cats.keys():
                        try:
                            cid = int(k)
                        except Exception:
                            continue
                        if cid in CAT_TO_PALETTE:
                            cat_to_palette_out[str(cid)] = CAT_TO_PALETTE[cid]
                        else:
                            if 0 <= cid <= 255:
                                cat_to_palette_out[str(cid)] = cid
                    for tmn_cid, pal_idx in CAT_TO_PALETTE.items():
                        cat_to_palette_out.setdefault(str(tmn_cid), pal_idx)
                    with open(os.path.join(um_dir, 'palette_mapping.json'), 'w', encoding='utf-8') as f_map:
                        json.dump({'cat_to_palette': cat_to_palette_out}, f_map, ensure_ascii=False)
                except Exception:
                    pass
                print({'total_placed': total_placed, 'out_images': OUT_IMAGES, 'out_jsons': out_paths, 'palette_mapping': os.path.join(BASE_DIR, 'understanding_dataset', 'palette_mapping.json')})
            except Exception:
                # Fallback to single write if something goes wrong
                data['annotations'] = anns_out
                out_json = os.path.join(OUT_JSONLAR, 'deepscores_train.json')
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                print({'total_placed': total_placed, 'out_images': OUT_IMAGES, 'out_json': out_json})
        else:
            data['annotations'] = anns_out
            out_json = os.path.join(OUT_JSONLAR, 'deepscores_train.json')
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            # Also write palette mapping for segmentation to understanding_dataset
            try:
                um_dir = os.path.join(BASE_DIR, 'understanding_dataset')
                os.makedirs(um_dir, exist_ok=True)
                # Build full mapping: default index = cat_id (if in [0,255]),
                # override TMN categories using CAT_TO_PALETTE.
                cat_to_palette_out = {}
                cats = data.get('categories') or {}
                # categories may be dict with string keys -> {'name': ...}
                for k in cats.keys():
                    try:
                        cid = int(k)
                    except Exception:
                        continue
                    if cid in CAT_TO_PALETTE:
                        cat_to_palette_out[str(cid)] = CAT_TO_PALETTE[cid]
                    else:
                        # use identity mapping for original dataset categories within palette range
                        if 0 <= cid <= 255:
                            cat_to_palette_out[str(cid)] = cid
                # ensure all TMN are present
                for tmn_cid, pal_idx in CAT_TO_PALETTE.items():
                    cat_to_palette_out.setdefault(str(tmn_cid), pal_idx)
                with open(os.path.join(um_dir, 'palette_mapping.json'), 'w', encoding='utf-8') as f_map:
                    json.dump({'cat_to_palette': cat_to_palette_out}, f_map, ensure_ascii=False)
            except Exception:
                pass
            print({'total_placed': total_placed, 'out_images': OUT_IMAGES, 'out_json': out_json, 'palette_mapping': os.path.join(BASE_DIR, 'understanding_dataset', 'palette_mapping.json')})
    else:
        print({'total_placed': total_placed, 'out_images': OUT_IMAGES, 'out_segmentation': OUT_SEG, 'out_instance': OUT_INST})

if __name__=='__main__':
    main()
