# Colab Setup for DS2 Dense TMN Training

This guide helps you run training on Google Colab with GPU.

## Steps

1. Open the notebook `colab/Train_MaskRCNN_DS2DenseTMN.ipynb` in Google Colab.
2. In Colab, enable GPU: Runtime → Change runtime type → GPU.
3. Mount Google Drive in the notebook.
4. If you want to run augmentation directly on Drive data (no upload), set:
	- `SRC_ROOT=/content/drive/MyDrive/omr_dataset/dataset/ds2/ds2_dense`
	- `OUT_ROOT=/content/drive/MyDrive/omr_dataset/dataset/ds2/ds2_dense_tmn`
	- The notebook includes a cell to run `place_tmn_batch.py` with `--from-fs-missing` and `--json-out-mode per-shard` so only missing outputs are generated and JSONs are written per shard.
5. Run the notebook cells top-to-bottom to install deps and execute placement.
6. For training, point your dataloader to `OUT_ROOT/images` and the `OUT_ROOT/jsonlar/*.json` files; checkpoints are saved under `BASE_DIR/outputs/mask_rcnn` (you can change this).

## Packing the dataset from Windows (PowerShell)

Run from your project root to create a zip you can upload to Drive:

```powershell
$src = "C:\projects\yl\omr_copilot\ds2_dense_tmn";
$zip = "C:\projects\yl\omr_copilot\ds2_dense_tmn.zip";
if (Test-Path $zip) { Remove-Item $zip -Force };
Add-Type -AssemblyName System.IO.Compression.FileSystem;
[System.IO.Compression.ZipFile]::CreateFromDirectory($src, $zip);
Write-Host "Zipped to $zip";
```

Upload `ds2_dense_tmn.zip` to your Google Drive under `MyDrive/omr_copilot/datasets/ds2_dense_tmn/`.

## Notes

- Colab preinstalls compatible CUDA; the notebook installs `torch/torchvision`, `pycocotools`, and common libs.
- Keep batch size small (2–4) to avoid OOM.
- Save checkpoints to Drive for persistence.

## Push code to GitHub (Windows PowerShell)

If you haven't pushed this repo yet and want Colab to clone from GitHub:

```powershell
cd C:\projects\yl\omr_copilot
git init
git add -A
git commit -m "Initial push with Colab notebook and TMN placement"
git remote add origin https://github.com/mtalhabalci/omr_copilot.git
git branch -M master
git push -u origin master
```

In Colab, the notebook includes a cell that clones `https://github.com/mtalhabalci/omr_tmn.git` into `/content/omr_tmn` and runs the placement script from there.
