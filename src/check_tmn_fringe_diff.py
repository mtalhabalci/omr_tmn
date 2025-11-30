import os
from PIL import Image

TMN_PAL = {248,249,250,251,252,253,254,255}

def check_page(page_png: str, base_seg_png: str, tmn_seg_png: str):
    img = Image.open(page_png).convert('RGBA')
    seg0 = Image.open(base_seg_png)
    seg1 = Image.open(tmn_seg_png)
    if seg0.mode != 'P': seg0 = seg0.convert('P')
    if seg1.mode != 'P': seg1 = seg1.convert('P')
    if not (img.size == seg0.size == seg1.size):
        raise RuntimeError(f"Size mismatch: img {img.size} seg0 {seg0.size} seg1 {seg1.size}")
    w, h = img.size
    img_px = img.load()
    s0 = seg0.load()
    s1 = seg1.load()
    total = 0
    non_black = 0
    sample = None
    for y in range(h):
        for x in range(w):
            v1 = s1[x,y]
            if v1 not in TMN_PAL:
                continue
            v0 = s0[x,y]
            if v0 == v1:
                # not our overlay; skip
                continue
            total += 1
            r,g,b,a = img_px[x,y]
            if (r,g,b) != (0,0,0):
                non_black += 1
                if sample is None:
                    sample = (x,y,(r,g,b,a),v0,v1)
    return total, non_black, sample

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(__file__))
    page = os.path.join(root, 'ds2_dense_tmn', 'images', 'lg-101766503886095953-aug-beethoven--page-1.png')
    base_seg = os.path.join(root, 'ds2_dense', 'segmentation', 'lg-101766503886095953-aug-beethoven--page-1_seg.png')
    tmn_seg = os.path.join(root, 'ds2_dense_tmn', 'segmentation', 'lg-101766503886095953-aug-beethoven--page-1_seg.png')
    if not (os.path.isfile(page) and os.path.isfile(base_seg) and os.path.isfile(tmn_seg)):
        print('Required files not found')
    else:
        total, non_black, sample = check_page(page, base_seg, tmn_seg)
        print({'tmn_overlay_pixels': total, 'non_black': non_black, 'sample': sample})
