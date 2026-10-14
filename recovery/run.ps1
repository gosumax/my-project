param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [Parameter(Mandatory=$true)][string]$Video,
    [Parameter(Mandatory=$true)][int]$Frames,
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$OcrWorkers = 2,
    [int]$VisualWorkers = 2,
    [int]$CardWorkers = 2
)
$ErrorActionPreference = 'Stop'
$dest = [IO.Path]::GetFullPath($Destination)
$python = Join-Path $dest 'runtime\python\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Restored base Python missing. Run restore.ps1 first.' }
& $python -I -S -B (Join-Path $PSScriptRoot 'run.py') --destination $dest --video $Video --frames $Frames --output $Output --ocr-workers $OcrWorkers --visual-workers $VisualWorkers --card-workers $CardWorkers
exit $LASTEXITCODE
