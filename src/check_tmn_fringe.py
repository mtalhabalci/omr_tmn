import os
from PIL import Image

# TMN palette indices as used in place_tmn_batch.py
TMN_PAL = {248,249,250,251,252,253,254,255}

def check_page(page_png: str, seg_png: str):
    img = Image.open(page_png).convert('RGBA')
    seg = Image.open(seg_png)
    if seg.mode != 'P':
        seg = seg.convert('P')
    if img.size != seg.size:
        raise RuntimeError(f"Size mismatch: {img.size} vs {seg.size}")
    w, h = img.size
    img_px = img.load()
    seg_px = seg.load()
    total = 0
    non_black = 0
    sample = None
    for y in range(h):
        for x in range(w):
            if seg_px[x, y] in TMN_PAL:
                total += 1
                r, g, b, a = img_px[x, y]
                if (r, g, b) != (0, 0, 0):
                    non_black += 1
                    if sample is None:
                        sample = (x, y, (r, g, b, a), seg_px[x,y])
    return total, non_black, sample

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(__file__))
    page = os.path.join(root, 'ds2_dense_tmn', 'images', 'lg-101766503886095953-aug-beethoven--page-1.png')
    seg = os.path.join(root, 'ds2_dense_tmn', 'segmentation', 'lg-101766503886095953-aug-beethoven--page-1_seg.png')
    if not (os.path.isfile(page) and os.path.isfile(seg)):
        print('Required files not found')
    else:
        total, non_black, sample = check_page(page, seg)
        print({'tmn_pixels': total, 'non_black': non_black, 'sample': sample})
