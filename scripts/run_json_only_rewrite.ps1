param(
  [string]$PythonExe = "python",
  [string]$RepoRoot = "$PSScriptRoot\..",
  [Parameter(Mandatory=$true)] [string]$JsonGlob,
  [Parameter(Mandatory=$true)] [string]$ImagesDir,
  [string]$SymbolsDir = "",
  [Parameter(Mandatory=$true)] [string]$OutRoot,
  [int]$SlotW = 17,
  [int]$SlotH = 48,
  [int]$Checkpoint = 50,
  [switch]$Force
)

<#
Example usage (PowerShell 5.1):

  $repo = "C:\projects\yl\omr_copilot"
  $srcImages = "C:\projects\yl\oemer-main\dataset yedek\ds2_complete\images"
  $outRoot = "C:\projects\yl\oemer-main\dataset yedek\ds2_complete_tmn"
  $glob = "C:\projects\yl\oemer-main\dataset yedek\ds2_complete\**\deepscores-complete-*_train.json"

  PowerShell: .\scripts\run_json_only_rewrite.ps1 -RepoRoot $repo -ImagesDir $srcImages -OutRoot $outRoot -JsonGlob $glob -Checkpoint 50 -Force

Notes:
  - Uses place_tmn_batch.py with --json-only to update annotations only.
  - Writes per-shard JSONs into OutRoot\jsonlar with the same basenames as sources.
  - Set SymbolsDir explicitly if not under $RepoRoot\tmn_symbols_png.
#>

if (-not $SymbolsDir -or $SymbolsDir -eq "") {
  $SymbolsDir = Join-Path $RepoRoot "tmn_symbols_png"
}

$placeScript = Join-Path $RepoRoot "src\place_tmn_batch.py"
if (-not (Test-Path $placeScript)) {
  Write-Error "place_tmn_batch.py not found at $placeScript"
  exit 1
}

# Resolve glob to files (train shards). Support patterns with ** by splitting base and leaf pattern.
$jsonFiles = @()
try {
  # If the glob matches files directly, use it as-is
  $direct = Get-ChildItem -Path $JsonGlob -File -ErrorAction SilentlyContinue
  if ($direct) {
    $jsonFiles = $direct
  } else {
    $leafPattern = Split-Path -Path $JsonGlob -Leaf
    $parentRaw = Split-Path -Path $JsonGlob -Parent
    # Remove any ** segment from the base path and search recursively
    $parentBase = $parentRaw -replace "\*\*", ""
    if (-not (Test-Path -LiteralPath $parentBase)) {
      # Fallback: try one level up if needed
      $parentBase = Split-Path -Path $parentBase -Parent
    }
    if (-not (Test-Path -LiteralPath $parentBase)) {
      throw "Base path not found: $parentBase"
    }
    $jsonFiles = Get-ChildItem -Path $parentBase -Recurse -File -Filter $leafPattern -ErrorAction SilentlyContinue
  }
} catch {
  Write-Error $_
}
if (-not $jsonFiles -or $jsonFiles.Count -eq 0) {
  Write-Error "No JSON files matched: $JsonGlob"
  exit 2
}

Write-Host ("Found {0} shard JSONs" -f $jsonFiles.Count)

foreach ($f in $jsonFiles) {
  Write-Host ("Processing shard: {0}" -f $f.FullName)
  $argsList = @(
    $placeScript,
    "--json-path", $f.FullName,
    "--images-dir", $ImagesDir,
    "--symbols-dir", $SymbolsDir,
    "--out-root", $OutRoot,
    "--json-out-mode", "per-shard",
    "--slot-w", $SlotW,
    "--slot-h", $SlotH,
    "--checkpoint", $Checkpoint,
    "--json-only"
  )
  if ($Force.IsPresent) {
    $argsList += "--force"
  }

  & $PythonExe @argsList
  if ($LASTEXITCODE -ne 0) {
    Write-Warning ("Shard failed (exit {0}): {1}" -f $LASTEXITCODE, $f.FullName)
  }
}

Write-Host "JSON-only rewrite complete."
