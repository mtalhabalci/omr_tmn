import os
from PIL import Image

def check_page(page_png: str, inst_png: str):
    img = Image.open(page_png).convert('RGBA')
    inst = Image.open(inst_png).convert('RGBA')
    if img.size != inst.size:
        raise RuntimeError(f"Size mismatch: {img.size} vs {inst.size}")
    w, h = img.size
    img_px = img.load()
    inst_px = inst.load()
    fringe = 0
    sample = None
    for y in range(h):
        for x in range(w):
            ir, ig, ib, ia = inst_px[x, y]
            if ia == 0:
                continue
            r, g, b, a = img_px[x, y]
            # Any non-black RGB within instance region indicates residual color
            if (r, g, b) != (0, 0, 0):
                fringe += 1
                if sample is None:
                    sample = (x, y, (r, g, b, a))
    return fringe, sample

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(__file__))
    page = os.path.join(root, 'ds2_dense_tmn', 'images', 'lg-101766503886095953-aug-beethoven--page-1.png')
    inst = os.path.join(root, 'ds2_dense_tmn', 'instance', 'lg-101766503886095953-aug-beethoven--page-1_inst.png')
    if not (os.path.isfile(page) and os.path.isfile(inst)):
        print('Required files not found')
    else:
        fringe, sample = check_page(page, inst)
        print({'fringe_pixels': fringe, 'sample': sample})
