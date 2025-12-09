# Colab: Mount Google Drive
import sys, os
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted at /content/drive')
else:
    print('Not in Colab; skipping drive mount.')

    # Colab: Bağımlılıkları yükle (gerekliyse)
import importlib, sys, subprocess

def ensure(pkg, import_name=None, spec=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
        print(f"ok: {name}")
    except Exception:
        cmd = [sys.executable, '-m', 'pip', 'install']
        if spec:
            cmd.append(spec)
        else:
            cmd.append(pkg)
        print('installing:', ' '.join(cmd))
        subprocess.check_call(cmd)
        importlib.invalidate_caches()
        importlib.import_module(name)
        print(f"installed: {name}")

# COCO eval için
ensure('pycocotools', 'pycocotools')
# Görselleştirme
ensure('matplotlib', 'matplotlib')


# Proje Yapılandırması ve Kütüphanelerin İçe Aktarılması
import os, sys, json, glob, time
import torch, torchvision
from PIL import Image
from pathlib import Path
print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

# Parametreleştirme: Config Sözlüğü ve Dataclass
from dataclasses import dataclass
from typing import Optional
import sys
from pathlib import Path

@dataclass
class Config:
    # Eğitim girdileri: TMN dataset
    OUT_ROOT: str = '/content/drive/MyDrive/omr_dataset/dataset/ds2/ds2_dense_tmn'
    IMG_ROOT: str = '/content/drive/MyDrive/omr_dataset/dataset/ds2/ds2_dense_tmn/images'
    # Eğitim çıktıları: ayrı bir train klasöründe kalıcı
    TRAIN_ROOT: str = '/content/drive/MyDrive/omr_dataset/dataset/ds2/train/ds2_dense_tmn'
    # Model özel alt klasörü
    MASKRCNN_ROOT: str = '/content/drive/MyDrive/omr_dataset/dataset/ds2/train/ds2_dense_tmn/maskrcnn'
    # Bellek için daha konservatif başlangıç batch boyutu
    BATCH_SIZE: int = 1
    EPOCHS: int = 3
    LR: float = 0.005
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005
    # Bellek kontrolleri
    MAX_INSTANCES: Optional[int] = None   # Görsel başına sınır yok (RAM yeterliyse)
    DATA_LOADER_PIN_MEMORY: bool = False  # RAM tüketimini azaltmak için kapalı
    # Değerlendirme ayarları
    EVAL_SPLIT: str = 'test'   # 'test' yoksa 'train' olarak ayarla
    EVAL_SCORE_THR: float = 0.05

cfg = Config()

# Yerel Windows ortamı: repo kökünde ds2_dense_tmn varsa yolları otomatik yerelle
try:
    IN_COLAB = 'google.colab' in sys.modules
    repo_root = Path(os.getcwd())
    local_ds = repo_root / 'ds2_dense_tmn'
    if not IN_COLAB and local_ds.exists():
        cfg.OUT_ROOT = str(local_ds)
        cfg.IMG_ROOT = str(local_ds / 'images')
        cfg.TRAIN_ROOT = str(repo_root / 'train' / 'ds2_dense_tmn')
        cfg.MASKRCNN_ROOT = str(repo_root / 'train' / 'ds2_dense_tmn' / 'maskrcnn')
        print({'local_dataset_detected': True, 'OUT_ROOT': cfg.OUT_ROOT})
except Exception as e:
    print('Yerel yol ayarlama atlandı:', e)

print(cfg)

# Çalışma klasörü: tarih/saat etiketli alt klasör (maskrcnn altında)
import time, os
os.makedirs(cfg.MASKRCNN_ROOT, exist_ok=True)
RUN_DIR = os.path.join(cfg.MASKRCNN_ROOT, time.strftime('%Y%m%d_%H%M'))
os.makedirs(RUN_DIR, exist_ok=True)
print('RUN_DIR:', RUN_DIR)


# Kategori Haritası: category_id -> eğitim etiketi (1..K), 0 arkaplan
import json, glob, os

train_jsons = sorted(glob.glob(f"{cfg.OUT_ROOT}/jsonlar/*train*.json"))
test_jsons = sorted(glob.glob(f"{cfg.OUT_ROOT}/jsonlar/*test*.json"))

def build_category_maps(json_paths):
    cats = set()
    for jp in json_paths:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        anns = data.get('annotations') or {}
        ann_iter = anns.values() if isinstance(anns, dict) else anns
        for a in ann_iter:
            cats_val = a.get('cat_id') or a.get('category_id')
            if isinstance(cats_val, list):
                for c in cats_val:
                    try:
                        cats.add(int(c))
                    except Exception:
                        pass
            elif cats_val is not None:
                try:
                    cats.add(int(cats_val))
                except Exception:
                    pass
    cats = sorted(cats)
    cat_map = {orig: i+1 for i, orig in enumerate(cats)}  # 0: background
    inv_cat_map = {v: k for k, v in cat_map.items()}
    return cat_map, inv_cat_map

ALL_JSONS = train_jsons + test_jsons
CAT_MAP, INV_CAT_MAP = build_category_maps(ALL_JSONS)
NUM_CLASSES = 1 + len(CAT_MAP)
print({'num_classes': NUM_CLASSES, 'categories': len(CAT_MAP)})


# Gelişmiş Kategori Özeti: isim ve adetleri annotasyonlardan da topla
import json
from collections import defaultdict
import os

# 1) categories alanından isimleri topla (varsa)
def _collect_category_names_from_categories(json_paths):
    names = {}
    for jp in json_paths:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        cats = data.get('categories') or []
        if isinstance(cats, dict):
            cats = list(cats.values())
        for c in cats:
            try:
                cid = int(c.get('id'))
            except Exception:
                continue
            nm = c.get('name') or c.get('category_name') or None
            if cid not in names and nm:
                names[cid] = str(nm)
    return names

# 2) annotations içinden olası isim alanlarından topla ve adetleri say
def _collect_from_annotations(json_paths):
    counts = defaultdict(int)
    ann_names = {}
    candidate_keys = [
        'category_name', 'cat_name', 'name', 'class_name', 'class',
        'symbol', 'label', 'semantic', 'semantic_name', 'cat', 'category'
    ]
    for jp in json_paths:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        anns = data.get('annotations') or {}
        ann_iter = anns.values() if isinstance(anns, dict) else anns
        for a in ann_iter:
            # id
            cid_val = a.get('cat_id') or a.get('category_id')
            try:
                cid = int(cid_val)
            except Exception:
                continue
            counts[cid] += 1
            # isim ipuçları
            for k in candidate_keys:
                v = a.get(k)
                if isinstance(v, str) and len(v) > 0:
                    if cid not in ann_names:
                        ann_names[cid] = v
                    break
    return counts, ann_names

NAMES_FROM_CATS = _collect_category_names_from_categories(ALL_JSONS)
COUNTS_BY_CAT, ANN_NAME_BY_CAT = _collect_from_annotations(ALL_JSONS)

# Öncelik: categories -> annotation isimleri
NAME_BY_CAT_ID = dict(NAMES_FROM_CATS)
for cid, nm in ANN_NAME_BY_CAT.items():
    NAME_BY_CAT_ID.setdefault(cid, nm)

# 3) Referans isimler (understanding_dataset/categories.json) -> eksik isimler için yedek
try:
    ref_path = os.path.join(os.getcwd(), 'understanding_dataset', 'categories.json')
    REF_NAME_BY_ID = {}
    if os.path.exists(ref_path):
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref = json.load(f)
        for k, v in ref.items():
            try:
                cid = int(k)
            except Exception:
                continue
            nm = v.get('name')
            if isinstance(nm, str) and nm:
                REF_NAME_BY_ID[cid] = nm
        # Eksik olanları referanstan tamamla (dataset/annotation isimleri öncelikli)
        for cid, nm in REF_NAME_BY_ID.items():
            NAME_BY_CAT_ID.setdefault(cid, nm)
        print(f"Referans isimler yüklendi: {len(REF_NAME_BY_ID)} id")
    else:
        print('Referans isim dosyası yok: understanding_dataset/categories.json')
except Exception as e:
    print('Referans isimler okunamadı:', e)

ORIG_CATEGORY_IDS = sorted(CAT_MAP.keys())
TRAIN_LABEL_TO_CATEGORY = {tr: INV_CAT_MAP[tr] for tr in sorted(INV_CAT_MAP.keys())}

print('Toplam kategori:', len(ORIG_CATEGORY_IDS))
print('İsmi bilinen kategori sayısı:', sum(1 for cid in ORIG_CATEGORY_IDS if cid in NAME_BY_CAT_ID))

# Listede çok kategori olabilir; çıktıyı sınırlamak için ayar
max_rows = 120  # gerekirse artır/azalt
printed = 0
print('\nEğitim label -> category_id | adet | isim')
for tr in range(1, NUM_CLASSES):
    cid = TRAIN_LABEL_TO_CATEGORY.get(tr)
    cnt = int(COUNTS_BY_CAT.get(cid, 0))
    nm = NAME_BY_CAT_ID.get(cid)
    nm_disp = nm if nm is not None else '-'
    print(f"  {tr:3d} -> {cid:4d} | {cnt:6d} | {nm_disp}")
    printed += 1
    if printed >= max_rows and NUM_CLASSES-1 > max_rows:
        remaining = (NUM_CLASSES-1) - max_rows
        print(f"... {remaining} satır daha (max_rows={max_rows}). Değeri arttırabilirsiniz.")
        break

# Dizi olarak saklama (tam liste)
TRAIN_CATEGORY_IDS = [TRAIN_LABEL_TO_CATEGORY[i] for i in range(1, NUM_CLASSES)]
TRAIN_CATEGORY_NAMES = [NAME_BY_CAT_ID.get(cid) for cid in TRAIN_CATEGORY_IDS]
print('\nTRAIN_CATEGORY_IDS (ilk 50):', TRAIN_CATEGORY_IDS[:50])
print('TRAIN_CATEGORY_NAMES (ilk 50):', TRAIN_CATEGORY_NAMES[:50])

# Kategori özetini JSON olarak kaydet (RUN_DIR/reports/category_summary.json)
import os, json

if 'RUN_DIR' not in globals():
    print('RUN_DIR tanımsız. Önce Config hücresini çalıştırın.')
else:
    REPORT_DIR = os.path.join(RUN_DIR, 'reports')
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, 'category_summary.json')

    items = []
    for tr in range(1, NUM_CLASSES):
        cid = TRAIN_LABEL_TO_CATEGORY.get(tr)
        cnt = int(COUNTS_BY_CAT.get(cid, 0))
        nm = NAME_BY_CAT_ID.get(cid)
        items.append({
            'train_label': int(tr),
            'category_id': int(cid) if cid is not None else None,
            'count': cnt,
            'name': nm
        })

    summary = {
        'total_categories': int(len(ORIG_CATEGORY_IDS)),
        'with_names': int(sum(1 for cid in ORIG_CATEGORY_IDS if cid in NAME_BY_CAT_ID)),
        'train_labels': int(NUM_CLASSES - 1),
        'category_ids': [int(x) for x in TRAIN_CATEGORY_IDS],
        'category_names': TRAIN_CATEGORY_NAMES,
        'items': items
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('Kategori özeti kaydedildi ->', out_path)

# Kategori Özeti: ID ve (varsa) isimleri yazdır
import json

def _collect_category_names(json_paths):
    names = {}
    for jp in json_paths:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        cats = data.get('categories') or []
        if isinstance(cats, dict):
            cats = list(cats.values())
        for c in cats:
            try:
                cid = int(c.get('id'))
            except Exception:
                continue
            nm = c.get('name') or c.get('category_name') or None
            if cid not in names and nm:
                names[cid] = str(nm)
    return names

NAME_BY_CAT_ID = _collect_category_names(ALL_JSONS)

ORIG_CATEGORY_IDS = sorted(CAT_MAP.keys())
TRAIN_LABEL_TO_CATEGORY = {tr: INV_CAT_MAP[tr] for tr in sorted(INV_CAT_MAP.keys())}

print('Orijinal category_id listesi (sıralı):', ORIG_CATEGORY_IDS)
print('Toplam kategori:', len(ORIG_CATEGORY_IDS))
print('\nEğitim label -> category_id (ve ad):')
for tr in range(1, NUM_CLASSES):
    cid = TRAIN_LABEL_TO_CATEGORY.get(tr)
    nm = NAME_BY_CAT_ID.get(cid)
    if nm is not None:
        print(f'  {tr:2d} -> {cid}  ({nm})')
    else:
        print(f'  {tr:2d} -> {cid}')

# Opsiyonel: dizi olarak sakla
TRAIN_CATEGORY_IDS = [TRAIN_LABEL_TO_CATEGORY[i] for i in range(1, NUM_CLASSES)]
TRAIN_CATEGORY_NAMES = [NAME_BY_CAT_ID.get(cid) for cid in TRAIN_CATEGORY_IDS]
print('\nTRAIN_CATEGORY_IDS:', TRAIN_CATEGORY_IDS)
print('TRAIN_CATEGORY_NAMES:', TRAIN_CATEGORY_NAMES)

# Günlükleme (Logging)
import logging, os
LOG_DIR = os.path.join(RUN_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler(os.path.join(LOG_DIR, 'train.log')),
    logging.StreamHandler()
])
logging.info('Logging initialized at %s', LOG_DIR)

# Veri İşleme Akışı ve Dataset (Cell 12) - LAZY LOADING VERSION
import torch
from torch.utils.data import Dataset, DataLoader

class DS2TMNDataset(Dataset):
    def __init__(self, images_dir, json_paths, transform=None, max_instances=None, category_map=None):
        import json
        self.images_dir = images_dir
        self.transform = transform
        self.max_instances = max_instances
        self.category_map = category_map
        self.json_paths = json_paths  # JSON yollarını sakla (lazy loading için)
        self.items = []  # {'filename': str, 'image_id': int, 'json_idx': int}
        
        # ✅ LAZY LOADING: Sadece görüntü listesini topla, annotation'ları yükleme
        seen_fn = set()
        for json_idx, jp in enumerate(json_paths):
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)

            imgs = data.get('images') or []
            if isinstance(imgs, dict):
                imgs = list(imgs.values())
            
            for im in imgs:
                fn = im.get('filename') or im.get('file_name')
                if not fn or fn in seen_fn:
                    continue
                seen_fn.add(fn)
                
                try:
                    iid = int(im.get('id')) if im.get('id') is not None else -1
                except Exception:
                    iid = -1
                
                # Hangi JSON'da olduğunu kaydet (lazy loading için)
                self.items.append({
                    'filename': fn, 
                    'image_id': iid,
                    'json_idx': json_idx
                })

        # Görselleri alfabetik sırala (tekrar üretilebilirlik için)
        self.items.sort(key=lambda x: x['filename'])
        self.to_tensor = torchvision.transforms.ToTensor()
        
        # ✅ BOŞ GÖRÜNTÜ FİLTRESİ: Annotation'ı olmayan görüntüleri baştan filtrele
        if category_map is not None:
            print(f"⏳ Filtering images without annotations...")
            valid_items = []
            for item in self.items:
                # Her görüntü için annotation var mı kontrol et (lazy check)
                has_anns = self._check_has_annotations(item['image_id'])
                if has_anns:
                    valid_items.append(item)
            
            filtered_count = len(self.items) - len(valid_items)
            self.items = valid_items
            print(f"✅ Filtered out {filtered_count} images without annotations")
        
        print(f"✅ Lazy Loading: {len(self.items)} images indexed (annotations will load on-demand)")

    def _check_has_annotations(self, image_id):
        """Bir görüntünün en az 1 valid annotation'ı olup olmadığını kontrol et"""
        import json
        for jp in self.json_paths:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            anns = data.get('annotations') or {}
            ann_iter = anns.values() if isinstance(anns, dict) else anns
            
            for a in ann_iter:
                img_id = a.get('img_id') or a.get('image_id')
                if img_id != image_id:
                    continue
                
                b = a.get('a_bbox') or a.get('bbox')
                if not b or len(b) < 4:
                    continue
                
                # En az 1 valid bbox varsa True döndür
                cats = a.get('cat_id') or a.get('category_id') or []
                if isinstance(cats, list) and len(cats) > 0:
                    orig_lab = int(cats[0])
                elif isinstance(cats, (int, str)):
                    orig_lab = int(cats)
                else:
                    continue
                
                # Category mapping kontrolü
                if self.category_map is not None:
                    mapped = self.category_map.get(orig_lab)
                    if mapped is None:
                        continue
                
                # Valid annotation bulundu!
                return True
        
        return False  # Hiç valid annotation yok

    def _load_annotations_for_image(self, filename, image_id):
        """Lazy loading: Sadece bu görüntünün annotation'larını JSON'dan oku"""
        import json
        annotations = []
        
        # İlgili JSON dosyasını bul ve sadece bu görüntünün annotation'larını yükle
        for jp in self.json_paths:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Bu JSON'daki annotation'ları tara
            anns = data.get('annotations') or {}
            ann_iter = anns.values() if isinstance(anns, dict) else anns
            
            for a in ann_iter:
                img_id = a.get('img_id') or a.get('image_id')
                if img_id != image_id:
                    continue  # Bu annotation başka bir görüntüye ait
                
                b = a.get('a_bbox') or a.get('bbox')
                cats = a.get('cat_id') or a.get('category_id') or []
                if not b or len(b) < 4:
                    continue
                
                # Etiket çözümleme (liste/tekil)
                if isinstance(cats, list) and len(cats) > 0:
                    orig_lab = int(cats[0])
                elif isinstance(cats, (int, str)):
                    orig_lab = int(cats)
                else:
                    orig_lab = 0
                
                # Eğitim için map'lenmiş etiket (1..K), 0 arkaplan kullanılmaz
                if self.category_map is not None:
                    mapped = self.category_map.get(orig_lab)
                    if mapped is None:
                        continue  # bilinmeyense atla
                    lab_to_use = int(mapped)
                else:
                    lab_to_use = int(orig_lab)
                
                annotations.append([
                    float(b[0]), float(b[1]), float(b[2]), float(b[3]), lab_to_use
                ])
        
        return annotations

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        im = self.items[idx]
        fn = im['filename']
        img_id = int(im['image_id'])
        path = os.path.join(self.images_dir, fn)
        
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            W = H = 1
            img = Image.new('RGB', (W, H), (0, 0, 0))
            lst = []
        else:
            W, H = img.size
            # ✅ LAZY LOADING: Annotation'ları ihtiyaç anında yükle
            lst = self._load_annotations_for_image(fn, img_id)

        boxes_labels = []
        for rec in lst:
            x1, y1, x2, y2, lab = rec
            x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W));     y2 = max(0, min(y2, H))
            if x2 > x1 and y2 > y1:
                boxes_labels.append((x1, y1, x2, y2, lab))

        # ⚠️ BOŞ ANNOTATION FİLTRESİ: Eğer hiç bbox yoksa, bir sonraki görüntüyü dene
        if len(boxes_labels) == 0:
            # Rastgele bir sonraki indeksi dene (sonsuz döngüyü önlemek için max 10 deneme)
            for attempt in range(10):
                next_idx = (idx + attempt + 1) % len(self.items)
                try:
                    return self.__getitem__(next_idx)
                except Exception:
                    continue
            # 10 denemede de bulamadıysak, dummy bir bbox döndür (model çalışmaya devam etsin)
            boxes_labels = [(0, 0, min(10, W), min(10, H), 1)]  # Minimal dummy bbox

        # Instance sınırı (bellek için)
        if self.max_instances and len(boxes_labels) > self.max_instances:
            boxes_labels = boxes_labels[: self.max_instances]

        boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in boxes_labels]
        labels = [lab for *_, lab in boxes_labels]

        if len(boxes) > 0:
            masks = torch.zeros((len(boxes), H, W), dtype=torch.bool)
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                masks[i, int(y1):int(y2), int(x1):int(x2)] = True
        else:
            masks = torch.zeros((0, H, W), dtype=torch.bool)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': masks,
            'image_id': torch.tensor([img_id], dtype=torch.int64)
        }
        img = self.transform(img) if self.transform else self.to_tensor(img)
        return img, target

train_jsons = sorted(glob.glob(f"{cfg.OUT_ROOT}/jsonlar/*train*.json"))
test_jsons = sorted(glob.glob(f"{cfg.OUT_ROOT}/jsonlar/*test*.json"))
train_ds = DS2TMNDataset(images_dir=cfg.IMG_ROOT, json_paths=train_jsons, max_instances=cfg.MAX_INSTANCES, category_map=CAT_MAP)
test_ds = DS2TMNDataset(images_dir=cfg.IMG_ROOT, json_paths=test_jsons, max_instances=cfg.MAX_INSTANCES, category_map=CAT_MAP)

def collate_fn(batch):
    return tuple(zip(*batch))

# RAM tüketimini kontrol altında tutmak için otomatik büyütmeyi kaldırdık
bs = cfg.BATCH_SIZE
# Colab'de worker öldürme sorunlarını önlemek için num_workers=0; pin_memory bellek için kapalı
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=cfg.DATA_LOADER_PIN_MEMORY, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=cfg.DATA_LOADER_PIN_MEMORY, collate_fn=collate_fn)
print({'batch_size': bs, 'train_len': len(train_ds), 'test_len': len(test_ds)})

# Etiket aralığı hızlı kontrol (örneklem)
def _check_label_ranges(ds, name, max_samples=100):
    seen_min, seen_max, checked, bad = None, None, 0, 0
    for i in range(min(max_samples, len(ds))):
        _, t = ds[i]
        l = t['labels']
        if l.numel() == 0:
            continue
        lmin = int(l.min())
        lmax = int(l.max())
        seen_min = lmin if seen_min is None else min(seen_min, lmin)
        seen_max = lmax if seen_max is None else max(seen_max, lmax)
        if (l < 1).any() or (l > (NUM_CLASSES-1)).any():
            bad += 1
        checked += 1
    print({name: {'sample_min': seen_min, 'sample_max': seen_max, 'checked': checked, 'out_of_range_samples': bad}})

_check_label_ranges(train_ds, 'train_labels')

# Model (Cell 13): Build lightweight Mask R-CNN (MobileNetV3-FPN) with version fallback (Önerilen)
import os, torch, torchvision
from typing import Optional

# Ensure NUM_CLASSES is available (compute from CAT_MAP if possible)
if 'NUM_CLASSES' not in globals():
    if 'CAT_MAP' in globals():
        NUM_CLASSES = 1 + len(CAT_MAP)
        print({'computed_NUM_CLASSES': NUM_CLASSES})
    else:
        raise RuntimeError("NUM_CLASSES tanımlı değil. Lütfen önce 'Config' ve 'Kategori Haritası' hücrelerini çalıştırın.")

print({'torch': torch.__version__, 'torchvision': torchvision.__version__})
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('medium')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: Optional[torch.nn.Module] = None
used_impl = None

# Try factory API first; fallback to manual backbone build if unavailable
try:
    from torchvision.models.detection import maskrcnn_mobilenet_v3_large_fpn
    model = maskrcnn_mobilenet_v3_large_fpn(num_classes=NUM_CLASSES)
    used_impl = 'factory'
except Exception as e:
    print('maskrcnn_mobilenet_v3_large_fpn not available, falling back:', e)
    from torchvision.models.detection.backbone_utils import mobilenet_backbone
    from torchvision.models.detection import MaskRCNN
    backbone = mobilenet_backbone('mobilenet_v3_large', pretrained=True, fpn=True, trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=NUM_CLASSES)
    used_impl = 'backbone_utils'

# Tight memory settings
if hasattr(model, 'transform'):
    if hasattr(model.transform, 'min_size'):
        model.transform.min_size = [256]  # ✅ AZALTILDI: 384 -> 256 (RAM tasarrufu)
    if hasattr(model.transform, 'max_size'):
        model.transform.max_size = 512  # ✅ AZALTILDI: 640 -> 512 (RAM tasarrufu)
if hasattr(model, 'roi_heads'):
    if hasattr(model.roi_heads, 'detections_per_img'):
        model.roi_heads.detections_per_img = 20
    if hasattr(model.roi_heads, 'box_batch_size_per_image'):
        model.roi_heads.box_batch_size_per_image = 96

model.to(device)

# Anchor generator guard to match backbone feature maps
from torchvision.models.detection.rpn import AnchorGenerator
def _configure_anchor_generator(model, device, im_size=256):
    try:
        with torch.no_grad():
            x = torch.zeros(1, 3, im_size, im_size, device=device)
            feats = model.backbone(x)
        if isinstance(feats, dict):
            n = len(feats)
        elif isinstance(feats, (list, tuple)):
            n = len(feats)
        else:
            n = 1
        base_sizes = (32, 64, 128, 256, 512, 1024)
        sizes = tuple((base_sizes[i],) for i in range(n))
        ratios = tuple(((0.5, 1.0, 2.0),) for _ in range(n))
        model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
        print(f'AnchorGenerator configured: levels={n}, sizes={[list(s) for s in sizes]}')
    except Exception as e:
        print('Anchor guard skipped:', e)
    finally:
        try:
            del x, feats
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

_configure_anchor_generator(model, device)

print(f'Mask R-CNN (MobileNetV3-FPN via {used_impl}) ready:', 'CUDA' if torch.cuda.is_available() else 'CPU')

# Memory-safe overrides (Cell 14) — modelden hemen sonra çalıştırın
import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('medium')
try:
    from torchvision.models.detection.mask_rcnn import MaskRCNN
    if 'model' in globals() and isinstance(model, MaskRCNN):
        if hasattr(model.transform, 'min_size'):
            model.transform.min_size = [256]  # ✅ AZALTILDI: 448 -> 256
        if hasattr(model.transform, 'max_size'):
            model.transform.max_size = 512  # ✅ AZALTILDI: 768 -> 512
        if hasattr(model, 'roi_heads'):
            if hasattr(model.roi_heads, 'detections_per_img'):
                model.roi_heads.detections_per_img = 30
            if hasattr(model.roi_heads, 'box_batch_size_per_image'):
                model.roi_heads.box_batch_size_per_image = 128
        print('Memory overrides applied (transform + ROI).')
    else:
        print('Model not yet defined; will apply when available.')
except Exception as e:
    print('OOM overrides skipped:', e)

    # Modülerleştirme ve Model Kurulumu (Cell 15) — Opsiyonel ağır alternatif
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

num_classes = NUM_CLASSES  # background(0) + K kategori (1..K)
model = maskrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('Model on', device, '| num_classes =', num_classes)


# Eğitim Döngüsü (AMP) ve Checkpoint (Cell 16)
from torch.optim import SGD
from torch.amp import GradScaler, autocast
import os, time, gc

optimizer = SGD(model.parameters(), lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
CKPT_DIR = os.path.join(RUN_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

model.train()
for epoch in range(cfg.EPOCHS):
    t0 = time.time(); total = 0.0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=torch.cuda.is_available()):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        total += losses.item()
        # Batch sonu temizlik (özellikle RAM/VRAM baskısını azaltmak için)
        del images, targets, loss_dict, losses
    dur = time.time()-t0
    avg = total / max(1,len(train_loader))
    logging.info({'epoch': epoch+1, 'loss': round(avg,3), 'sec': round(dur,1)})
    ckpt = os.path.join(CKPT_DIR, f'maskrcnn_epoch{epoch+1}.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1}, ckpt)
    logging.info({'saved': ckpt})
    # Epoch sonu bellek temizliği
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
print('Done')

# Değerlendirme ve Görselleştirme (Cell 17) — Opsiyonel
import matplotlib.pyplot as plt
model.eval()
@torch.no_grad()
def eval_show(n=3, thr=0.5):
    shown = 0
    for images, targets in test_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for img, out in zip(images, outputs):
            if shown>=n: return
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(img.permute(1,2,0).cpu().numpy())
            boxes = out['boxes'].cpu().numpy(); scores = out['scores'].cpu().numpy()
            for b,s in zip(boxes, scores):
                if s<thr: continue
                x1,y1,x2,y2 = b
                ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color='y', linewidth=2))
            ax.set_title(f'detections >= {thr}')
            plt.show(); plt.close(fig)  # Bellek sızıntılarını önlemek için figürü kapat
            shown += 1

eval_show(3, 0.5)

# COCO mAP Evaluation (train/test sabit eşleme, RAM dostu akış) (Cell 18)
import json, os, gc
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Split seçimi: 'train' -> train_jsons & train_loader, 'test' -> test_jsons & test_loader
split = (cfg.EVAL_SPLIT or 'test').lower()
if split == 'test':
    assert len(test_jsons) > 0, "test split için JSON bulunamadı (ör. deepscores_test.json)"
    EVAL_JSON = test_jsons[0]
    eval_loader = test_loader
elif split == 'train':
    assert len(train_jsons) > 0, "train split için JSON bulunamadı (ör. deepscores_train.json)"
    EVAL_JSON = train_jsons[0]
    eval_loader = train_loader
else:
    raise ValueError(f"Bilinmeyen EVAL_SPLIT: {split}. 'train' veya 'test' olmalı.")

print({'eval_split': split, 'json': EVAL_JSON})

# Eğer GT JSON'da categories yoksa veya isimler boşsa -> geçici bir GT yaz ve onu kullan
try:
    with open(EVAL_JSON, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
except Exception as e:
    gt_data = None
    print('GT json okunamadı:', e)

def _extract_unique_cat_ids(annotations):
    ids = set()
    if isinstance(annotations, dict):
        ann_iter = annotations.values()
    else:
        ann_iter = annotations or []
    for a in ann_iter:
        if not isinstance(a, dict):
            continue
        cid_val = a.get('cat_id') or a.get('category_id')
        if isinstance(cid_val, list) and cid_val:
            try:
                ids.add(int(cid_val[0]))
            except Exception:
                pass
        else:
            try:
                if cid_val is not None:
                    ids.add(int(cid_val))
            except Exception:
                pass
    return sorted(ids)

# Referans isimler (understanding_dataset/categories.json) -> opsiyonel
REF_NAME_BY_ID = {}
try:
    ref_path = os.path.join(os.getcwd(), 'understanding_dataset', 'categories.json')
    if os.path.exists(ref_path):
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref = json.load(f)
        for k, v in ref.items():
            try:
                cid = int(k)
            except Exception:
                continue
            nm = v.get('name')
            if isinstance(nm, str) and nm:
                REF_NAME_BY_ID[cid] = nm
except Exception as e:
    print('Referans isimler okunamadı:', e)

USE_JSON = EVAL_JSON
if gt_data is not None:
    cats = gt_data.get('categories')
    need_patch = False
    if not cats:
        need_patch = True
    else:
        # categories varsa ama isimleri eksikse yine patchle
        def _has_name(c):
            return isinstance(c, dict) and (c.get('name') or c.get('category_name'))
        if isinstance(cats, dict):
            cats_list = list(cats.values())
        else:
            cats_list = cats
        if not all(_has_name(c) for c in cats_list):
            need_patch = True
    if need_patch:
        cat_ids = _extract_unique_cat_ids(gt_data.get('annotations') or [])
        patched_cats = []
        for cid in cat_ids:
            name = REF_NAME_BY_ID.get(cid, f'cat_{cid}')
            patched_cats.append({'id': int(cid), 'name': name})
        gt_data['categories'] = patched_cats
        REPORT_DIR = os.path.join(RUN_DIR, 'reports')
        os.makedirs(REPORT_DIR, exist_ok=True)
        patched_path = os.path.join(REPORT_DIR, f'gt_with_categories_{split}.json')
        with open(patched_path, 'w', encoding='utf-8') as fw:
            json.dump(gt_data, fw, ensure_ascii=False)
        USE_JSON = patched_path
        print({'gt_patched': True, 'path': patched_path, 'cat_count': len(patched_cats)})

cocoGt = COCO(USE_JSON)

# Akış tabanlı tespit toplama: JSONL dosyasına yaz, bellekte tutma
REPORT_DIR = os.path.join(RUN_DIR, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)
DETS_JSONL = os.path.join(REPORT_DIR, f'detections_{split}.jsonl')
DETS_JSON = os.path.join(REPORT_DIR, f'detections_{split}.json')  # COCO loadRes JSON array ister

model.eval()
import numpy as np
@torch.no_grad()
def collect_detections_stream(score_thr=0.05):
    # Var olan dosyayı temizle
    if os.path.exists(DETS_JSONL):
        os.remove(DETS_JSONL)
    processed = 0
    with open(DETS_JSONL, 'w', encoding='utf-8') as fw:
        for images, targets in eval_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                img_id = int(tgt['image_id'].item())
                boxes = out['boxes'].detach().cpu().numpy()
                scores = out['scores'].detach().cpu().numpy()
                labels = out['labels'].detach().cpu().numpy()
                for b, s, lab in zip(boxes, scores, labels):
                    if s < score_thr:
                        continue
                    x1,y1,x2,y2 = b
                    # Eğitim label'ını COCO GT'deki orijinal category_id'ye geri çevir
                    orig_cat = int(INV_CAT_MAP.get(int(lab), int(lab)))
                    rec = {
                        'image_id': img_id,
                        'category_id': orig_cat,
                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        'score': float(s)
                    }
                    fw.write(json.dumps(rec, ensure_ascii=False) + '\n')
            # Bellek temizliği (batch bazlı)
            del images, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            processed += len(targets)
    return processed

# Tüm seçilen split'i işle (RAM şişmeden)
processed = collect_detections_stream(score_thr=cfg.EVAL_SCORE_THR)
print(f'Processed images ({split}):', processed)

# JSONL -> JSON Array (stream ederek)
with open(DETS_JSONL, 'r', encoding='utf-8') as fr, open(DETS_JSON, 'w', encoding='utf-8') as fw:
    fw.write('[')
    first = True
    for line in fr:
        line = line.strip()
        if not line:
            continue
        if not first:
            fw.write(',')
        fw.write(line)
        first = False
    fw.write(']')

# Evaluate mAP
# COCO loadRes dosya yolunu kabul eder (JSON array formatında)
try:
    cocoDt = cocoGt.loadRes(DETS_JSON)
except Exception as e:
    print('Failed to load detections for COCOeval:', e)
    cocoDt = None

if cocoDt is None:
    print('No detections to evaluate or load failed.')
else:
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # Save metrics
    metrics = {
        'AP@[.5:.95]': float(cocoEval.stats[0]),
        'AP@0.5': float(cocoEval.stats[1]),
        'AP@0.75': float(cocoEval.stats[2])
    }
    with open(os.path.join(REPORT_DIR, f'metrics_{split}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f)
    print('Saved metrics to', os.path.join(REPORT_DIR, f'metrics_{split}.json'))
    # Bellek temizliği
    del cocoDt, cocoEval
    gc.collect()


# Çıktı Özeti (RUN_DIR, checkpoint ve raporlar) (Cell 19)
import os, glob, json

if 'RUN_DIR' not in globals():
    print('RUN_DIR tanımlı değil. Önce Config hücresini çalıştırın.')
else:
    print('RUN_DIR          :', RUN_DIR)
    ckpt_dir = os.path.join(RUN_DIR, 'checkpoints')
    rep_dir  = os.path.join(RUN_DIR, 'reports')

    print('\n[checkpoints]')
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
    print('adet:', len(ckpts))
    if ckpts:
        print('son:', ckpts[-1])

    print('\n[reports]')
    if os.path.isdir(rep_dir):
        reps = sorted(glob.glob(os.path.join(rep_dir, '*')))
        for p in reps[:20]:
            print('-', os.path.basename(p))
        # metrics_<split>.json varsa göster
        for split_name in ['train', 'test']:
            mpath = os.path.join(rep_dir, f'metrics_{split_name}.json')
            if os.path.exists(mpath):
                try:
                    with open(mpath, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    print(f"\nmetrics_{split_name}.json:", metrics)
                except Exception as e:
                    print(f"metrics_{split_name}.json okunamadı:", e)
    else:
        print('Rapor klasörü yok: önce değerlendirme hücresini çalıştırın.')

# Dataset Probe (Cell 20): JSON yapısı, anahtarlar ve top kategori özetleri
import os, json, glob
from collections import Counter, defaultdict

REPORT_DIR = os.path.join(RUN_DIR, 'reports') if 'RUN_DIR' in globals() else None
if REPORT_DIR:
    os.makedirs(REPORT_DIR, exist_ok=True)

probe = {
    'json_root': f"{cfg.OUT_ROOT}/jsonlar",
    'files_scanned': [],
    'per_file': [],
    'annotation_keys_union': [],
}

json_files = sorted(glob.glob(f"{cfg.OUT_ROOT}/jsonlar/*.json"))
print({'json_count': len(json_files)})
sample_files = json_files[:3]
print('sample_files:', [os.path.basename(p) for p in sample_files])
probe['files_scanned'] = [os.path.basename(p) for p in sample_files]

ann_keys_union = set()

for path in sample_files:
    entry = {'file': os.path.basename(path)}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print('failed to read', path, e)
        entry['error'] = str(e)
        probe['per_file'].append(entry)
        continue

    top_keys = list(data.keys())
    entry['top_keys'] = top_keys
    print(f"\n[{os.path.basename(path)}] top-level keys:", top_keys)

    cats = data.get('categories')
    if cats is None:
        print('categories: NONE')
        entry['categories'] = {'present': False}
    else:
        if isinstance(cats, dict):
            cats_list = list(cats.values())
        else:
            cats_list = cats
        print('categories: present, len =', len(cats_list))
        entry['categories'] = {'present': True, 'len': len(cats_list)}
        for c in cats_list[:3]:
            cid = c.get('id')
            name = c.get('name') or c.get('category_name')
            print('  sample category ->', {'id': cid, 'name': name})

    anns = data.get('annotations') or []
    if isinstance(anns, dict):
        ann_iter = list(anns.values())
    else:
        ann_iter = anns
    print('annotations len =', len(ann_iter))

    # Anahtar kümeleri ve isim adaylarını ara
    ann_keys = set()
    name_hits = {}
    counts = Counter()
    candidate_keys = [
        'category_name','cat_name','name','class_name','class',
        'symbol','label','semantic','semantic_name','cat','category'
    ]

    for a in ann_iter[:1000]:
        if isinstance(a, dict):
            ann_keys.update(a.keys())
            cid_val = a.get('cat_id') or a.get('category_id')
            try:
                cid = int(cid_val)
            except Exception:
                cid = None
            if cid is not None:
                counts[cid] += 1
                if cid not in name_hits:
                    for k in candidate_keys:
                        v = a.get(k)
                        if isinstance(v, str) and v:
                            name_hits[cid] = v
                            break
    ann_keys_union.update(ann_keys)

    entry['annotation_keys'] = sorted(list(ann_keys))
    entry['name_hits_count'] = len(name_hits)
    entry['top_categories'] = [
        {'category_id': int(cid), 'count': int(cnt), 'name': name_hits.get(cid)}
        for cid, cnt in counts.most_common(20)
    ]
    print('annotation keys (sample):', sorted(list(ann_keys))[:20])
    print('top 10 categories by count:')
    for x in entry['top_categories'][:10]:
        print(' ', x)

    probe['per_file'].append(entry)

probe['annotation_keys_union'] = sorted(list(ann_keys_union))

if REPORT_DIR:
    out_path = os.path.join(REPORT_DIR, 'dataset_probe.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(probe, f, ensure_ascii=False, indent=2)
    print('Dataset probe saved ->', out_path)
else:
    print('RUN_DIR tanımlı değil; JSON raporu kaydedilemedi.')


# Colab (Cell 21): categories.json'ı Drive'dan kopyala (opsiyonel ama tavsiye)
import os, shutil

# Colab kontrolü - lint hatalarını önlemek için
def _is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = _is_colab()

LOCAL_REF_DIR = os.path.join(os.getcwd(), 'understanding_dataset')
LOCAL_REF_PATH = os.path.join(LOCAL_REF_DIR, 'categories.json')
# Kullanıcının belirttiği konum + önceki varsayılanı dene (ilk bulunanı kopyala)
DRIVE_REF_CANDIDATES = [
    '/content/drive/MyDrive/omr_dataset/dataset/ds2/categories.json',
    '/content/drive/MyDrive/omr_copilot/understanding_dataset/categories.json',
]

os.makedirs(LOCAL_REF_DIR, exist_ok=True)

if IN_COLAB:
    src = None
    for p in DRIVE_REF_CANDIDATES:
        if os.path.exists(p):
            src = p
            break
    if src:
        shutil.copy2(src, LOCAL_REF_PATH)
        print({'copied': True, 'from': src, 'to': LOCAL_REF_PATH})
    else:
        print({'copied': False, 'reason': 'Drive path not found', 'tried': DRIVE_REF_CANDIDATES})
else:
    print('Not in Colab; skipping Drive copy.')

# Gradient accumulation training loop to reduce peak memory (Cell 22) — Opsiyonel
try:
    accum_steps = 2
    if 'train_loader' in globals() and 'model' in globals() and 'optimizer' in globals():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        from torch.amp import autocast, GradScaler
        scaler = GradScaler(enabled=torch.cuda.is_available())
        model.train()
        step = 0
        for epoch in range(1):  # set your desired number of epochs elsewhere
            optimizer.zero_grad(set_to_none=True)
            for images, targets in train_loader:
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
                with autocast('cuda', enabled=torch.cuda.is_available()):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                scaler.scale(losses / accum_steps).backward()
                step += 1
                if step % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                del images, targets, loss_dict, losses
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        print('Accumulation loop finished (for demo). Integrate into your main loop as needed.')
    else:
        print('train_loader/model/optimizer not ready; plug this into your training cell.')
except Exception as e:
    print('Accumulation loop skipped:', e)