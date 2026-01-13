param(
  [string]$PythonExe = "python",
  [Parameter(Mandatory=$true)] [string]$OutRoot,
  [int]$IntervalSec = 60,
  [int]$Tail = 500,
  [int]$TotalImages = 0
)

$progressPath = Join-Path $OutRoot "jsonlar\progress.jsonl"
$monitor = Join-Path $PSScriptRoot "..\understanding_dataset\monitor_progress.py"
if (-not (Test-Path $monitor)) {
  Write-Error "monitor_progress.py not found at $monitor"
  exit 1
}

Write-Host "Watching progress at: $progressPath"
while ($true) {
  Clear-Host
  if (-not (Test-Path $progressPath)) {
    Write-Host "Waiting for progress file... ($progressPath)"
  } else {
    & $PythonExe $monitor --progress $progressPath --tail $Tail --total-images $TotalImages
  }
  Start-Sleep -Seconds $IntervalSec
}
