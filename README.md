# OMR + TMN Augmentation Pipeline (Kısa)

## Amaç
DeepScoresV2 (Western) sayfalarına Türk Müziği mikrotonal accidentallerini (TMN) sentetik olarak ekleyip unified model eğitmek.

## Dizim
```
./ds2_dense/               # Orijinal yoğun DeepScoresV2 alt seti
  deepscores_train.json    # images + annotations (COCO benzeri)
./sample_annotations.json  # Seçilmiş notehead örnekleri (küçük alt küme)
./tmn_symbols_png/         # TMN sembol PNG dosyaları (sen manuel koyuyorsun)
augment_tmn_samples.py     # Notehead yanına TMN sembol ekler (multi-symbol)
inject_tmn_annotations.py  # Yeni TMN kategori + annotation + image kayıtlarını JSON'a yazar
```

## Adımlar
1. Notehead örnek çıkarımı (önceden yapıldı): `extract_sample_annotations.py`.
2. Augment üret: `python augment_tmn_samples.py`
   - Her annotation için rastgele bir TMN sembol seçer.
   - Sol tarafa dikey merkezli ekler.
   - `ds2_dense/deepscores_train.json` içinden image id -> filename map kullanır.
3. Birleşik JSON üret (gerekirse tekrar): `python inject_tmn_annotations.py`
   - TMN kategori id: 6000.
   - Yeni görüntü isimleri: `aug_*` prefix.
4. Eğitim hazirlama: Sınıf frekansı, kategori ekleme, config (harici).

## Önemli Noktalar
- Artık SVG dönüştürme yok; sadece `tmn_symbols_png` içindeki hazır PNG'ler kullanılıyor.
- Çoklu semboller bu klasördeki tüm *.png dosyalarından rastgele seçiliyor.
- Görüntü eşleştirme: JSON map yoksa heuristik pattern + son çare id.png.

## Genişletme Fikirleri
- Collision avoidance (çoklu sembol aynı nota veya staff arası çakışma kontrolü).
- TMN sembol yoğunluğu parametresi (p olasılık, max per page).
- Ek mikrotonal set / Hamparsum entegrasyonu.

## Hızlı Test
```
python augment_tmn_samples.py
# Çıktılar: ./augmented_samples/ içinde aug_<img_id>_*_*.png
```

## Sorun Giderme
- "[SKIP] Görüntü bulunamadı" -> img_id dataset'te yok veya alt set farklı; `deepscores_train.json` içindeki ilgili id'yi doğrula.
- Placeholder çok görünüyorsa: gerçek PNG export et (`cairosvg` kur veya manuel rasterize).

## Lisans / Not
DeepScoresV2 lisans koşullarına uyum + TMN SVG'leri proje içi sentez amaçlı.
