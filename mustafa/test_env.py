# Test Script - Environment Kontrolü
import sys
print("Python version:", sys.version)
print("="*60)

try:
    import torch
    print("✅ PyTorch:", torch.__version__)
    print("   CPU available:", torch.cpu.is_available() if hasattr(torch, 'cpu') else True)
except ImportError as e:
    print("❌ PyTorch yüklenemedi:", e)

try:
    import torchvision
    print("✅ Torchvision:", torchvision.__version__)
except ImportError as e:
    print("❌ Torchvision yüklenemedi:", e)

try:
    import PIL
    print("✅ Pillow:", PIL.__version__)
except ImportError as e:
    print("❌ Pillow yüklenemedi:", e)

try:
    import numpy
    print("✅ NumPy:", numpy.__version__)
except ImportError as e:
    print("❌ NumPy yüklenemedi:", e)

try:
    import matplotlib
    print("✅ Matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("❌ Matplotlib yüklenemedi:", e)

try:
    import pycocotools
    print("✅ pycocotools: installed")
except ImportError as e:
    print("❌ pycocotools yüklenemedi:", e)

print("="*60)
print("\nEğer tüm paketler ✅ ise, train_local.py çalıştırabilirsiniz!")
