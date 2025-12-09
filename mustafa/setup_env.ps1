# Environment Kurulum Script'i - Windows PowerShell
# Bu dosyayı PowerShell'de çalıştırın: .\setup_env.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🚀 MaskRCNN Local Training - Environment Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 1. Python versiyonunu kontrol et
Write-Host "`n📌 Checking Python version..." -ForegroundColor Yellow
python --version

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python bulunamadı! Lütfen Python 3.8+ yükleyin." -ForegroundColor Red
    exit 1
}

# 2. Virtual environment oluştur
Write-Host "`n📦 Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠️  venv klasörü zaten var, siliniyor..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

python -m venv venv

# 3. Virtual environment'ı aktifleştir
Write-Host "`n✅ Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

# 4. Pip güncellemesi
Write-Host "`n📦 Updating pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 5. PyTorch CPU versiyonunu yükle
Write-Host "`n📦 Installing PyTorch (CPU version)..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. Diğer gerekli paketleri yükle
Write-Host "`n📦 Installing other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# 7. Yüklü paketleri listele
Write-Host "`n✅ Installed packages:" -ForegroundColor Green
pip list

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "✅ Environment hazır!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "`n📝 Eğitimi başlatmak için:" -ForegroundColor Yellow
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   python train_local.py" -ForegroundColor White
Write-Host ""
