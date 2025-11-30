# DeepScoresV2: Bir Görsel Üzerinden Yapı Anlatımı

Örnek görsel: `lg-2267728-aug-beethoven--page-2.png`

Aşağıda bu sayfanın görüntüsü, JSON kaydı, instance/segmentation maskeleri ve kategori renkleri üzerinden tam bir akış veriyorum.

## Dosya Konumları
- Görsel: `ds2_dense/images/lg-2267728-aug-beethoven--page-2.png`
- Instance mask: `ds2_dense/instance/lg-2267728-aug-beethoven--page-2_inst.png`
- Segmentation mask: `ds2_dense/segmentation/lg-2267728-aug-beethoven--page-2_seg.png`
- JSON: `ds2_dense/deepscores_train.json`

## JSON Yapısı: images
`images` içinde ilgili kayıt:
```
{"id": 35, "filename": "lg-2267728-aug-beethoven--page-2.png", "width": 1960, "height": 2772, "ann_ids": [
  "567855", "567856", "567857", "567858", ...
]}
```
- `id`: 35 → Bu sayfayı eşsiz tanımlıyor.
- `ann_ids`: Bu sayfaya ait annotation kimlikleri.

## JSON Yapısı: annotations (örnekler)
Annotationlar, kimlik→kayıt sözlüğü şeklinde tutulur. Örn. ilk üç annotation:
```
"567855": {
  "a_bbox": [93.0, 130.0, 1866.0, 197.0],
  "o_bbox": [1866.0, 197.0, 1866.0, 130.0, 93.0, 130.0, 93.0, 197.0],
  "cat_id": ["135", "208"],
  "area": 16456,
  "img_id": "35",
  "comments": "instance:#000006;"
},
"567856": {
  "a_bbox": [93.0, 296.0, 1866.0, 362.0],
  "o_bbox": [1866.0, 362.0, 1866.0, 296.0, 93.0, 296.0, 93.0, 362.0],
  "cat_id": ["135", "208"],
  "area": 11336,
  "img_id": "35",
  "comments": "instance:#000014;"
},
"567857": {
  "a_bbox": [1354.0, 278.0, 1384.0, 281.0],
  "o_bbox": [1384.0, 281.0, 1384.0, 278.0, 1354.0, 278.0, 1354.0, 281.0],
  "cat_id": ["2", "138"],
  "area": 43,
  "img_id": "35",
  "comments": "instance:#000015;"
}
```
- `a_bbox`: Axis-aligned bbox [x1, y1, x2, y2].
- `o_bbox`: Objeyi çevreleyen çokgenin köşe koordinatları.
- `cat_id`: Çiftli gelir ("deepscores" ve "muscima++" eşlemeleri). Örn. `135` ve `208` ikisi de “staff”.
- `comments.instance`: Instance maskede bu örneğe ayrılmış benzersiz renk (örn. `#000006`). Instance PNG’de bu piksel rengi o objeyi temsil eder.

## JSON Yapısı: categories (segmentation renkleri)
`categories` sözlüğünde her kategori için ad ve segmentation rengi (palette index) vardır. Örn. `135` ve `208`:
```
"135": {"name": "staff", "annotation_set": "deepscores", "color": 165},
"208": {"name": "staff", "annotation_set": "muscima++", "color": 165}
```
- `color`: Segmentation maskesinde bu kategoriye karşılık gelen palet index’idir. Yani segmentation tarafındaki renk JSON’da kategori düzeyinde kayıtlıdır; instance’taki renk ise annotation düzeyinde `comments` içinde tutulur.

Örn. `567857` içinde `cat_id: ["2","138"]` görüyoruz. Bunların `categories` karşılığı:
```
"2":   {"name": "ledgerLine", "annotation_set": "deepscores", "color": 2},
"138": {"name": "legerLine",  "annotation_set": "muscima++",   "color": 2}
```
Bu da segmentation maskesinde ledgerLine piksellerinin palet rengi 2 anlamına gelir.

## Instance vs Segmentation Özet
- Instance: Her obje kendi benzersiz rengi ile `..._inst.png` içinde işaretlenir. Bu renk, annotation’daki `comments` alanında `instance:#RRGGBB` olarak yazılıdır.
- Segmentation: Her piksel, ait olduğu kategoriye göre palet değerine (kategori `color`) boyanır. Bu palet rengi, `categories` altında tutulur; annotation düzeyinde renk yazılmaz.

## İsteğe Bağlı Görsel Üretim
Aşağıdaki küçük script, örnek sayfa üzerinde birkaç bbox’ı (567855, 567856, 567857) çizer ve instance renklerini başlıkta gösterir. Çıktı: `understanding_dataset/overlay_lg-2267728-aug-beethoven--page-2.png`

```
# Windows PowerShell’de çalıştırma (proje kökünden):
# Python venv zaten ayarlıysa:
# C:/projects/yl/omr_copilot/.venv/Scripts/python.exe understanding_dataset/overlay.py
```

```python
# understanding_dataset/overlay.py
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Girdi yolları
root = Path(r"c:\\projects\\yl\\omr_copilot")
img_path = root/"ds2_dense"/"images"/"lg-2267728-aug-beethoven--page-2.png"
out_path = root/"understanding_dataset"/"overlay_lg-2267728-aug-beethoven--page-2.png"

# Örnek annotation altkümesi (JSON’dan alınmış koordinatlar)
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
    draw.rectangle([tx,ty,tx+len(text)*7,ty+16], fill=(0,0,0,180))
    draw.text((tx+2, ty+1), text, fill=(255,255,255))

im.save(out_path)
print("saved:", out_path)
```

Notlar:
- BBox’lar `a_bbox` değerleriyle çizildi.
- Instance rengi (ör. `#000006`) metin olarak gösterildi; gerçekte bu renk, `..._inst.png` dosyasında ilgili objenin piksel rengi.
- Segmentation rengi annotation’da değil, kategori sözlüğünde (`categories[cat_id]['color']`). Örneğin `staff` için `color=165`.

## Kısa Cevaplar
- “Bu görüntünün instance ve segmentation örnekleri var mı?” → Evet, karşılıkları `_inst.png` ve `_seg.png` olarak var.
- “JSON’da segmentation rengi yazıyor mu?” → Evet, kategori düzeyinde `categories[id].color` altında var (palette index). Instance rengi ise annotation `comments` içinde `instance:#RRGGBB` olarak var.
- “Bir görselin JSON’da nasıl tutulduğu” → `images` içinde tek kayıt, `annotations` kısmında bu `img_id`’ye ait çok sayıda obje kaydı; `categories` kısmında her sınıf için isim ve segmentation palet rengi.
