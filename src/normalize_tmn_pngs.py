import os
from PIL import Image

def normalize_folder(folder_path: str) -> int:
    count = 0
    for name in os.listdir(folder_path):
        if not name.lower().endswith('.png'):
            continue
        fp = os.path.join(folder_path, name)
        try:
            im = Image.open(fp).convert('RGBA')
        except Exception as e:
            print(f"Skip {name}: {e}")
            continue
        pixels = im.load()
        w, h = im.size
        for y in range(h):
            for x in range(w):
                r, g, b, a = pixels[x, y]
                # Set RGB to pure black, preserve alpha
                pixels[x, y] = (0, 0, 0, a)
        im.save(fp)
        count += 1
    return count

if __name__ == '__main__':
    folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tmn_symbols_png')
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
    else:
        n = normalize_folder(folder)
        print(f"Normalized {n} PNG files in {folder}")
