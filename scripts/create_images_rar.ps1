# =============================================================================
# IMAGES KLASÖRÜNÜ 10 PARÇA RAR'A BÖLEN SCRIPT
# =============================================================================
# Kaynak: C:\projects\yl\oemer-main\dataset yedek\ds2_complete_tmn\images
# Hedef:  C:\projects\yl\oemer-main\dataset yedek\ds2_complete_tmn\zipler\images_rar\
# =============================================================================

$ErrorActionPreference = "Stop"

# Yollar
$imagesDir = "C:\projects\yl\oemer-main\dataset yedek\ds2_complete_tmn\images"
$outputDir = "C:\projects\yl\oemer-main\dataset yedek\ds2_complete_tmn\zipler\images_rar"
$winrarPath = "C:\Program Files\WinRAR\Rar.exe"

# WinRAR kontrolü
if (-not (Test-Path $winrarPath)) {
    Write-Host "HATA: WinRAR bulunamadı: $winrarPath" -ForegroundColor Red
    Write-Host "WinRAR yolunu kontrol et!" -ForegroundColor Red
    exit 1
}

# Hedef klasör oluştur
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Write-Host "Klasör oluşturuldu: $outputDir" -ForegroundColor Green
}

# Tüm dosyaları alfabetik sırala
Write-Host "`n[1/3] Dosyalar listeleniyor..." -ForegroundColor Cyan
$allFiles = Get-ChildItem -Path $imagesDir -File | Sort-Object Name
$totalFiles = $allFiles.Count
Write-Host "Toplam dosya: $totalFiles" -ForegroundColor Yellow

# 10 parçaya böl
$partsCount = 10
$filesPerPart = [math]::Ceiling($totalFiles / $partsCount)
Write-Host "Her RAR'da ~$filesPerPart dosya olacak`n" -ForegroundColor Yellow

# Her parça için RAR oluştur
Write-Host "[2/3] RAR dosyaları oluşturuluyor..." -ForegroundColor Cyan

for ($i = 0; $i -lt $partsCount; $i++) {
    $partNum = $i + 1
    $startIndex = $i * $filesPerPart
    $endIndex = [math]::Min(($startIndex + $filesPerPart - 1), ($totalFiles - 1))
    
    # Bu parçadaki dosyaları al
    $partFiles = $allFiles[$startIndex..$endIndex]
    $partFileCount = $partFiles.Count
    
    # RAR dosya adı
    $rarName = "images_{0:D2}.rar" -f $partNum
    $rarPath = Join-Path $outputDir $rarName
    
    # Geçici liste dosyası oluştur
    $listFile = Join-Path $env:TEMP "images_part_$partNum.txt"
    $partFiles | ForEach-Object { $_.FullName } | Out-File -FilePath $listFile -Encoding UTF8
    
    Write-Host "`n[$partNum/10] $rarName oluşturuluyor..." -ForegroundColor White
    Write-Host "       Dosya aralığı: $($startIndex + 1) - $($endIndex + 1) ($partFileCount dosya)" -ForegroundColor Gray
    
    # WinRAR komutu
    # -ep1 = Tam yolu değil, sadece dosya adını sakla
    # -m0  = Sıkıştırma yok (PNG zaten sıkışık, hızlı olur)
    # -o+  = Varsa üzerine yaz
    # @    = Liste dosyasından oku
    $rarArgs = @(
        "a",           # Add
        "-ep1",        # Exclude base path (sadece dosya adı)
        "-m0",         # Store (no compression)
        "-o+",         # Overwrite
        "`"$rarPath`"",
        "@`"$listFile`""
    )
    
    $startTime = Get-Date
    
    # RAR oluştur
    $process = Start-Process -FilePath $winrarPath -ArgumentList $rarArgs -NoNewWindow -Wait -PassThru
    
    $elapsed = (Get-Date) - $startTime
    
    if ($process.ExitCode -eq 0) {
        $rarSize = [math]::Round((Get-Item $rarPath).Length / 1GB, 2)
        Write-Host "       ✅ Tamamlandı: $rarSize GB ($([math]::Round($elapsed.TotalMinutes, 1)) dakika)" -ForegroundColor Green
    } else {
        Write-Host "       ❌ HATA! Exit code: $($process.ExitCode)" -ForegroundColor Red
    }
    
    # Geçici dosyayı sil
    Remove-Item $listFile -ErrorAction SilentlyContinue
}

# Özet
Write-Host "`n[3/3] ÖZET" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan
Get-ChildItem $outputDir -Filter "*.rar" | ForEach-Object {
    $sizeGB = [math]::Round($_.Length / 1GB, 2)
    Write-Host "$($_.Name): $sizeGB GB" -ForegroundColor White
}
$totalRarSize = [math]::Round((Get-ChildItem $outputDir -Filter "*.rar" | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "TOPLAM: $totalRarSize GB" -ForegroundColor Yellow
Write-Host "`n✅ İşlem tamamlandı!" -ForegroundColor Green
