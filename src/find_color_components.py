import os, sys, argparse, json
from PIL import Image, ImageDraw

"""
Scan a paletted segmentation PNG, find pixels matching a given RGB color,
extract connected components, and write:
- JSON with bounding boxes
- Overlay PNG with rectangles drawn

Usage:
  python src/find_color_components.py --file <seg_png> --color #00acc6 --min-size 10

Notes:
- Works for both paletted ('P') and RGB(A) PNGs; for 'P', the palette is used
  to map indices to RGB and match the requested color.
- Connected components via 4-neighborhood flood fill.
"""

def parse_color_hex(s: str):
    s = s.strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) == 6:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r,g,b)
    raise ValueError('Expected color like #RRGGBB')


def load_palette_rgb(im: Image.Image):
    if im.mode != 'P':
        return None
    pal = im.getpalette()
    if pal is None:
        return None
    # Return list of 256 RGB tuples
    rgb = []
    for i in range(256):
        base = i*3
        rgb.append((pal[base], pal[base+1], pal[base+2]))
    return rgb


def find_components(im: Image.Image, target_rgb, min_size=1):
    W, H = im.size
    px = im.load()
    palette = load_palette_rgb(im)
    visited = [[False]*W for _ in range(H)]
    comps = []

    def matches(x, y):
        val = px[x,y]
        if palette is not None:
            rgb = palette[val]
        else:
            # If not paletted, assume RGB(A)
            if im.mode in ('RGB','RGBA'):
                rgb = px[x,y] if im.mode=='RGB' else px[x,y][:3]
            else:
                # Fallback: treat val as gray
                rgb = (val, val, val)
        return rgb == target_rgb

    for y in range(H):
        for x in range(W):
            if visited[y][x]:
                continue
            if not matches(x,y):
                visited[y][x] = True
                continue
            # flood fill
            stack = [(x,y)]
            visited[y][x] = True
            minx = maxx = x
            miny = maxy = y
            count = 0
            while stack:
                cx, cy = stack.pop()
                count += 1
                if cx < minx: minx = cx
                if cx > maxx: maxx = cx
                if cy < miny: miny = cy
                if cy > maxy: maxy = cy
                for nx, ny in ((cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1)):
                    if 0 <= nx < W and 0 <= ny < H and not visited[ny][nx] and matches(nx,ny):
                        visited[ny][nx] = True
                        stack.append((nx,ny))
            if count >= min_size:
                comps.append({'x1':minx,'y1':miny,'x2':maxx+1,'y2':maxy+1,'pixels':count})
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Path to segmentation PNG (paletted or RGB)')
    ap.add_argument('--color', required=True, help='Hex color like #00acc6')
    ap.add_argument('--min-size', type=int, default=10, help='Minimum component pixel count to keep')
    args = ap.parse_args()

    seg_path = args.file
    if not os.path.isfile(seg_path):
        print({'error': f'File not found: {seg_path}'})
        sys.exit(1)
    target_rgb = parse_color_hex(args.color)
    im = Image.open(seg_path)
    comps = find_components(im, target_rgb, min_size=max(1,args.min_size))

    # Write JSON next to file
    base_dir = os.path.dirname(seg_path)
    base_name = os.path.basename(seg_path)
    name_noext = os.path.splitext(base_name)[0]
    out_json = os.path.join(base_dir, f"{name_noext}__{args.color.replace('#','')}_components.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'file': seg_path, 'color': f"#{args.color.strip('#')}", 'components': comps, 'count': len(comps)}, f, ensure_ascii=False)

    # Overlay rectangles on a copy of the image (convert to RGB for drawing)
    rgb_im = im.convert('RGB')
    dr = ImageDraw.Draw(rgb_im)
    for c in comps:
        dr.rectangle([(c['x1'], c['y1']), (c['x2'], c['y2'])], outline=(255,0,0), width=2)
    out_png = os.path.join(base_dir, f"{name_noext}__{args.color.replace('#','')}_overlay.png")
    rgb_im.save(out_png)

    print({'components': len(comps), 'json': out_json, 'overlay': out_png})

if __name__ == '__main__':
    main()
