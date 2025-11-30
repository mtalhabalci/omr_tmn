from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

root = Path(r"c:\projects\yl\omr_copilot")
img_path = root/"ds2_dense"/"images"/"lg-2267728-aug-beethoven--page-2.png"
out_path = root/"understanding_dataset"/"overlay_lg-2267728-aug-beethoven--page-2.png"

examples = [
    {"ann_id":"567855", "a_bbox":[93,130,1866,197],  "instance":"#000006", "label":"staff"},
    {"ann_id":"567856", "a_bbox":[93,296,1866,362],  "instance":"#000014", "label":"staff"},
    {"ann_id":"567857", "a_bbox":[1354,278,1384,281], "instance":"#000015", "label":"ledgerLine"},
]

im = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(im)

colors = [(255,0,0),(0,160,255),(0,200,100)]
for i, ex in enumerate(examples):
    x1,y1,x2,y2 = ex["a_bbox"]
    color = colors[i % len(colors)]
    draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
    text = f"{ex['ann_id']} {ex['label']} {ex['instance']}"
    tx,ty = x1, max(0, y1-18)
    # basit bir zemin kutusu
    w = max(120, len(text)*7)
    draw.rectangle([tx,ty,tx+w,ty+16], fill=(0,0,0))
    draw.text((tx+2, ty+1), text, fill=(255,255,255))

im.save(out_path)
print("saved:", out_path)
