# LOCAL TRAINING VERSION - CPU Optimized
# Windows local ortamı için optimize edilmiş MaskRCNN eğitim kodu

import sys, os
import torch, torchvision
from PIL import Image
from pathlib import Path
import json, glob, time, gc
import logging
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

print('='*60)
print('🖥️  LOCAL TRAINING MODE - CPU')
print('='*60)
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'Device: CPU (GPU disabled)')
print('='*60)

# ============================================================================
# CONFIG - Local Windows Paths
# ============================================================================

@dataclass
class Config:
    # 🔧 LOCAL PATHS - Otomatik algılama
    DATASET_NAME: str = 'ds2_dense_tmn3'
    BASE_DIR: str = r'C:\Users\mustafa.coban\Desktop\yar'
    
    # Dataset paths (otomatik oluşturulacak)
    OUT_ROOT: str = None
    IMG_ROOT: str = None
    TRAIN_ROOT: str = None
    MASKRCNN_ROOT: str = None
    
    # 💻 CPU Optimized Settings
    BATCH_SIZE: int = 1  # CPU için 1 yeterli
    EPOCHS: int = 2  # Test için kısa
    NUM_WORKERS: int = 0  # Windows + CPU için 0 önerilir
    
    # Model hyperparameters
    LR: float = 0.001  # CPU için daha düşük
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005
    
    # Memory settings
    MAX_INSTANCES: Optional[int] = 50  # CPU RAM için limit
    DATA_LOADER_PIN_MEMORY: bool = False
    
    # Image size (CPU için küçük)
    MIN_SIZE: int = 256
    MAX_SIZE: int = 512
    
    # Evaluation
    EVAL_SPLIT: str = 'test'
    EVAL_SCORE_THR: float = 0.05
    
    def __post_init__(self):
        """Otomatik path oluşturma"""
        dataset_path = Path(self.BASE_DIR) / self.DATASET_NAME
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset bulunamadı: {dataset_path}")
        
        self.OUT_ROOT = str(dataset_path)
        self.IMG_ROOT = str(dataset_path / 'images')
        self.TRAIN_ROOT = str(Path(self.BASE_DIR) / 'train' / self.DATASET_NAME)
        self.MASKRCNN_ROOT = str(Path(self.TRAIN_ROOT) / 'maskrcnn')
        
        print(f"\n✅ Dataset found: {dataset_path}")
        print(f"📁 Images: {len(list((dataset_path / 'images').glob('*')))} files")

cfg = Config()

# ============================================================================
# RUN DIRECTORY
# ============================================================================

os.makedirs(cfg.MASKRCNN_ROOT, exist_ok=True)
RUN_DIR = os.path.join(cfg.MASKRCNN_ROOT, time.strftime('%Y%m%d_%H%M%S') + '_local_cpu')
os.makedirs(RUN_DIR, exist_ok=True)
print(f'\n📂 RUN_DIR: {RUN_DIR}')

# ============================================================================
# LOGGING
# ============================================================================

LOG_DIR = os.path.join(RUN_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'train.log')),
        logging.StreamHandler()
    ]
)
logging.info('='*60)
logging.info('LOCAL TRAINING STARTED (CPU Mode)')
logging.info('='*60)

# ============================================================================
# CATEGORY MAPPING
# ============================================================================

train_jsons = sorted(glob.glob(f"{cfg.OUT_ROOT}/jsonlar/*train*.json"))
test_jsons = sorted(glob.glob(f"{cfg.OUT_ROOT}/jsonlar/*test*.json"))

logging.info(f"Train JSON files: {len(train_jsons)}")
logging.info(f"Test JSON files: {len(test_jsons)}")

def build_category_maps(json_paths):
    """Tüm kategorileri topla ve mapping oluştur"""
    cats = set()
    for jp in json_paths:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logging.warning(f"JSON okuma hatası {jp}: {e}")
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

logging.info(f"✅ Categories: {len(CAT_MAP)}, Total classes: {NUM_CLASSES}")

# ============================================================================
# LAZY LOADING DATASET
# ============================================================================

class DS2TMNDataset(Dataset):
    """Lazy loading dataset - RAM dostu"""
    
    def __init__(self, images_dir, json_paths, transform=None, max_instances=None, category_map=None):
        self.images_dir = images_dir
        self.transform = transform
        self.max_instances = max_instances
        self.category_map = category_map
        self.json_paths = json_paths
        self.items = []
        
        # Sadece görüntü listesini topla
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
                
                self.items.append({
                    'filename': fn,
                    'image_id': iid,
                    'json_idx': json_idx
                })
        
        self.items.sort(key=lambda x: x['filename'])
        self.to_tensor = torchvision.transforms.ToTensor()
        
        # ⚠️ Boş görüntü filtreleme ATLATILDI (çok yavaş)
        # Eğitim sırasında __getitem__ içinde zaten boş olanları handle ediyoruz
        logging.info(f"✅ Dataset ready: {len(self.items)} images (filtering skipped for speed)")
    
    def _check_has_annotations(self, image_id):
        """En az 1 valid annotation var mı?"""
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
                
                cats = a.get('cat_id') or a.get('category_id') or []
                if isinstance(cats, list) and len(cats) > 0:
                    orig_lab = int(cats[0])
                elif isinstance(cats, (int, str)):
                    orig_lab = int(cats)
                else:
                    continue
                
                if self.category_map is not None:
                    mapped = self.category_map.get(orig_lab)
                    if mapped is None:
                        continue
                
                return True
        
        return False
    
    def _load_annotations_for_image(self, filename, image_id):
        """Lazy loading: Sadece bu görüntünün annotation'larını yükle"""
        annotations = []
        
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
                cats = a.get('cat_id') or a.get('category_id') or []
                if not b or len(b) < 4:
                    continue
                
                if isinstance(cats, list) and len(cats) > 0:
                    orig_lab = int(cats[0])
                elif isinstance(cats, (int, str)):
                    orig_lab = int(cats)
                else:
                    orig_lab = 0
                
                if self.category_map is not None:
                    mapped = self.category_map.get(orig_lab)
                    if mapped is None:
                        continue
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
        except Exception as e:
            logging.warning(f"Image load error {fn}: {e}")
            W = H = 1
            img = Image.new('RGB', (W, H), (0, 0, 0))
            lst = []
        else:
            W, H = img.size
            lst = self._load_annotations_for_image(fn, img_id)
        
        boxes_labels = []
        for rec in lst:
            x1, y1, x2, y2, lab = rec
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            if x2 > x1 and y2 > y1:
                boxes_labels.append((x1, y1, x2, y2, lab))
        
        # Boş annotation güvenliği
        if len(boxes_labels) == 0:
            for attempt in range(10):
                next_idx = (idx + attempt + 1) % len(self.items)
                try:
                    return self.__getitem__(next_idx)
                except Exception:
                    continue
            boxes_labels = [(0, 0, min(10, W), min(10, H), 1)]
        
        # Instance sınırı
        if self.max_instances and len(boxes_labels) > self.max_instances:
            boxes_labels = boxes_labels[:self.max_instances]
        
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

# ============================================================================
# CREATE DATASETS
# ============================================================================

train_ds = DS2TMNDataset(
    images_dir=cfg.IMG_ROOT,
    json_paths=train_jsons,
    max_instances=cfg.MAX_INSTANCES,
    category_map=CAT_MAP
)

test_ds = DS2TMNDataset(
    images_dir=cfg.IMG_ROOT,
    json_paths=test_jsons,
    max_instances=cfg.MAX_INSTANCES,
    category_map=CAT_MAP
)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=cfg.DATA_LOADER_PIN_MEMORY,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_ds,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=cfg.DATA_LOADER_PIN_MEMORY,
    collate_fn=collate_fn
)

logging.info(f"✅ Train loader: {len(train_ds)} images, {len(train_loader)} batches")
logging.info(f"✅ Test loader: {len(test_ds)} images, {len(test_loader)} batches")

# ============================================================================
# MODEL - MobileNetV3 (Lightweight for CPU)
# ============================================================================

logging.info("🏗️  Building MobileNetV3 Mask R-CNN (CPU optimized)...")

device = torch.device('cpu')
torch.set_num_threads(4)  # CPU thread limiti

try:
    from torchvision.models.detection import maskrcnn_mobilenet_v3_large_fpn
    model = maskrcnn_mobilenet_v3_large_fpn(num_classes=NUM_CLASSES, weights=None)
    used_impl = 'factory'
except Exception as e:
    logging.warning(f"Factory method failed: {e}")
    from torchvision.models.detection.backbone_utils import mobilenet_backbone
    from torchvision.models.detection import MaskRCNN
    backbone = mobilenet_backbone('mobilenet_v3_large', pretrained=False, fpn=True, trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=NUM_CLASSES)
    used_impl = 'backbone_utils'

# CPU-friendly settings
if hasattr(model, 'transform'):
    if hasattr(model.transform, 'min_size'):
        model.transform.min_size = [cfg.MIN_SIZE]
    if hasattr(model.transform, 'max_size'):
        model.transform.max_size = cfg.MAX_SIZE

if hasattr(model, 'roi_heads'):
    if hasattr(model.roi_heads, 'detections_per_img'):
        model.roi_heads.detections_per_img = 20
    if hasattr(model.roi_heads, 'box_batch_size_per_image'):
        model.roi_heads.box_batch_size_per_image = 64  # CPU için düşük

model.to(device)
logging.info(f"✅ Model ready: MaskRCNN-MobileNetV3 (via {used_impl}) on CPU")

# ============================================================================
# TRAINING
# ============================================================================

from torch.optim import SGD

optimizer = SGD(
    model.parameters(),
    lr=cfg.LR,
    momentum=cfg.MOMENTUM,
    weight_decay=cfg.WEIGHT_DECAY
)

CKPT_DIR = os.path.join(RUN_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

logging.info("="*60)
logging.info(f"🚀 TRAINING START: {cfg.EPOCHS} epochs")
logging.info("="*60)

model.train()

for epoch in range(cfg.EPOCHS):
    epoch_start = time.time()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        batch_count += 1
        
        # Bellek temizliği
        del images, targets, loss_dict, losses
        gc.collect()
        
        # Progress log
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / batch_count
            logging.info(f"Epoch {epoch+1}/{cfg.EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
    
    # Epoch özeti
    epoch_duration = time.time() - epoch_start
    avg_loss = total_loss / max(1, len(train_loader))
    
    logging.info("="*60)
    logging.info(f"✅ Epoch {epoch+1}/{cfg.EPOCHS} Complete")
    logging.info(f"   Loss: {avg_loss:.4f}")
    logging.info(f"   Duration: {epoch_duration:.1f}s ({epoch_duration/60:.1f} min)")
    logging.info("="*60)
    
    # Checkpoint kaydet
    ckpt_path = os.path.join(CKPT_DIR, f'maskrcnn_epoch{epoch+1}.pt')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, ckpt_path)
    logging.info(f"💾 Checkpoint saved: {ckpt_path}")
    
    gc.collect()

logging.info("="*60)
logging.info("✅ TRAINING COMPLETE!")
logging.info("="*60)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("📊 TRAINING SUMMARY")
print("="*60)
print(f"RUN_DIR: {RUN_DIR}")
print(f"Checkpoints: {len(glob.glob(os.path.join(CKPT_DIR, '*.pt')))} files")
print(f"Logs: {LOG_DIR}")
print("="*60)
print("\n✅ Kod başarıyla tamamlandı!")
print("Checkpoint'leri test etmek için model.load_state_dict() kullanabilirsiniz.\n")
