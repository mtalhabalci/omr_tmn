# MaskRCNN Local Training - CPU Version

Windows local ortamında CPU ile Mask R-CNN eğitimi için optimize edilmiş kod.

## 📁 Klasör Yapısı

```
yar/
├── ds2_dense_tmn3/           # Dataset
│   ├── images/               # 1714 adet görüntü
│   ├── jsonlar/              # deepscores_train.json, deepscores_test.json
│   ├── instance/
│   └── segmentation/
├── train_local.py            # 🔥 Ana eğitim kodu (CPU optimize)
├── yarr.py                   # Colab versiyonu (GPU)
├── requirements.txt          # Gerekli paketler
├── setup_env.ps1            # Environment kurulum script'i
└── README.md                # Bu dosya
```

## 🚀 Kurulum (İlk Kez)

### 1. PowerShell'i Yönetici olarak açın

### 2. Script execution policy ayarlayın (ilk kez)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Environment kurun
```powershell
cd C:\Users\mustafa.coban\Desktop\yar
.\setup_env.ps1
```

Bu script:
- ✅ Virtual environment oluşturur
- ✅ PyTorch CPU versiyonunu yükler
- ✅ Tüm gerekli paketleri yükler

## 🏃 Eğitimi Başlatma

```powershell
# 1. Virtual environment'ı aktifleştir
.\venv\Scripts\Activate.ps1

# 2. Eğitimi başlat
python train_local.py
```

## ⚙️ Özellikler

### CPU Optimizasyonları
- ✅ **Batch size: 1** (CPU için optimal)
- ✅ **Görüntü boyutu: 256x512** (küçük, hızlı)
- ✅ **MobileNetV3 backbone** (hafif model)
- ✅ **Lazy loading** (düşük RAM kullanımı)
- ✅ **Boş annotation filtresi** (hata önleme)

### RAM Kullanımı
- **Beklenen:** 8-16 GB
- **Eski kod:** 172+ GB 😱

### Eğitim Süresi (tahmini)
- **1 epoch:** ~30-60 dakika (CPU'ya göre değişir)
- **2 epoch (varsayılan):** ~1-2 saat

## 📊 Çıktılar

Eğitim sonunda şu klasörler oluşur:

```
train/ds2_dense_tmn3/maskrcnn/20250209_143022_local_cpu/
├── logs/
│   └── train.log              # Detaylı log
├── checkpoints/
│   ├── maskrcnn_epoch1.pt     # Epoch 1 model
│   └── maskrcnn_epoch2.pt     # Epoch 2 model
└── reports/                   # (Evaluation sonrası)
```

## 🔧 Ayarlar

`train_local.py` içinde `Config` class'ını düzenleyebilirsiniz:

```python
@dataclass
class Config:
    BATCH_SIZE: int = 1        # 2'ye çıkarabilirsiniz (RAM yeterse)
    EPOCHS: int = 2            # 5-10 yapabilirsiniz
    MIN_SIZE: int = 256        # 320'ye çıkarabilirsiniz
    MAX_SIZE: int = 512        # 640'a çıkarabilirsiniz
    MAX_INSTANCES: int = 50    # Instance limiti
```

## ⚠️ Bilinen Sorunlar

### 1. Çok yavaş çalışıyor
- **Çözüm:** Normal! CPU ile eğitim GPU'ya göre 20-50x yavaştır
- **Öneri:** Colab'de GPU kullanın veya test için `EPOCHS=1` yapın

### 2. RAM doluyor
- **Çözüm:** `MAX_INSTANCES = 30` veya daha düşük yapın
- **Çözüm:** Diğer programları kapatın

### 3. "get_ipython is not defined" hatası
- **Çözüm:** Bu hata sadece `yarr.py` içinde (Colab versiyonu)
- **Kullanın:** `train_local.py` (Local versiyon)

## 📈 Model Kullanımı

Eğitilmiş modeli yüklemek için:

```python
import torch
from torchvision.models.detection import maskrcnn_mobilenet_v3_large_fpn

# Model oluştur
model = maskrcnn_mobilenet_v3_large_fpn(num_classes=YOUR_NUM_CLASSES)

# Checkpoint yükle
checkpoint = torch.load('train/ds2_dense_tmn3/maskrcnn/.../checkpoints/maskrcnn_epoch2.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    predictions = model([image_tensor])
```

## 🆘 Yardım

Sorun yaşarsanız:
1. Log dosyasını kontrol edin: `train/.../logs/train.log`
2. Python versiyonunu kontrol edin: `python --version` (3.8+ gerekli)
3. PyTorch'u test edin: `python -c "import torch; print(torch.__version__)"`

## 📝 Notlar

- Bu kod **sadece CPU** için optimize edilmiştir
- GPU kullanmak için `yarr.py` dosyasını Colab'de çalıştırın
- Lazy loading sayesinde RAM kullanımı minimal
- İlk epoch yavaş olabilir (annotation loading)
