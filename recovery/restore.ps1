param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dest = [IO.Path]::GetFullPath($Destination)
if ($dest.TrimEnd('\') -eq $repo.TrimEnd('\')) { throw 'Destination must differ from clone.' }
function Assert-NoReparse([string]$Path) {
    $cursor = [IO.Path]::GetFullPath($Path)
    while ($cursor) {
        if ([IO.File]::Exists($cursor) -or [IO.Directory]::Exists($cursor)) {
            if (([IO.File]::GetAttributes($cursor) -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse point in recovery path rejected.' }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor.TrimEnd('\'))
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
}
function Assert-OwnedTree([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($dest.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Recovery path escapes destination.' }
    Assert-NoReparse $full
    if ([IO.Directory]::Exists($full)) {
        $pending = New-Object 'System.Collections.Generic.Stack[string]'
        $pending.Push($full)
        while ($pending.Count -gt 0) {
            foreach ($child in [IO.Directory]::EnumerateFileSystemEntries($pending.Pop())) {
                $attrs = [IO.File]::GetAttributes($child)
                if (($attrs -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse point inside recovery runtime/cache rejected.' }
                if (($attrs -band [IO.FileAttributes]::Directory) -ne 0) { $pending.Push($child) }
            }
        }
    }
}
Assert-NoReparse $dest
Assert-OwnedTree (Join-Path $dest 'runtime')
Assert-OwnedTree (Join-Path $dest '.recovery-cache')
$marker = Join-Path $dest '.poker-recovery-owned'
Assert-NoReparse $marker
if ($VerifyOnly) {
    $receipt = Join-Path $dest 'recovery_receipt.json'
    $existingPython = Join-Path $dest 'runtime\python\python.exe'
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf) -or
        (Get-Content -LiteralPath $marker -Raw).Trim() -ne 'poker-recovery-v1' -or
        -not (Test-Path -LiteralPath $receipt -PathType Leaf) -or
        -not (Test-Path -LiteralPath $existingPython -PathType Leaf)) {
        throw 'VerifyOnly requires a complete owned restore with its receipt and base Python.'
    }
    & $existingPython -I -S -B (Join-Path $PSScriptRoot 'restore.py') --repo $repo --destination $dest --verify-only
    exit $LASTEXITCODE
}
if (Test-Path -LiteralPath $dest) {
    if (-not (Test-Path -LiteralPath $marker)) {
        if (@(Get-ChildItem -LiteralPath $dest -Force).Count -ne 0) { throw 'Nonempty destination has no recovery ownership marker.' }
    } elseif ((Get-Content -LiteralPath $marker -Raw).Trim() -ne 'poker-recovery-v1') { throw 'Unknown destination ownership marker.' }
} else { New-Item -ItemType Directory -Path $dest | Out-Null }
if (-not (Test-Path -LiteralPath $marker)) { [IO.File]::WriteAllText($marker, 'poker-recovery-v1') }
$cache = Join-Path $dest '.recovery-cache'
New-Item -ItemType Directory -Force -Path $cache | Out-Null
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'recovery_manifest.json') -Raw | ConvertFrom-Json
$bootstrap = $manifest.bootstrap
function Get-Sha256([string]$Path) {
    $sha = [Security.Cryptography.SHA256]::Create()
    $inputStream = [IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($sha.ComputeHash($inputStream)).Replace('-', '').ToLowerInvariant() }
    finally { $inputStream.Dispose(); $sha.Dispose() }
}
function Test-Blob([string]$Path, [string]$Hash, [long]$Bytes) {
    return ((Test-Path -LiteralPath $Path -PathType Leaf) -and (Get-Item -LiteralPath $Path).Length -eq $Bytes -and (Get-Sha256 $Path) -eq $Hash)
}
function Fetch-Blob([string]$GitPath, [string]$Target) {
    if ($GitPath -notmatch '^recovery/bundles/[A-Za-z0-9._/-]+$' -or $GitPath.Contains('..')) { throw 'Unsafe bundle path.' }
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = 'git.exe'
    $psi.Arguments = '-c http.version=HTTP/1.1 cat-file blob HEAD:' + $GitPath
    $psi.WorkingDirectory = $repo
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $proc = New-Object Diagnostics.Process
    $proc.StartInfo = $psi
    $stream = $null
    try {
        if (-not $proc.Start()) { throw 'Cannot start Git.' }
        $stderr = $proc.StandardError.ReadToEndAsync()
        $stream = [IO.File]::Create($Target)
        $copy = $proc.StandardOutput.BaseStream.CopyToAsync($stream)
        if (-not $proc.WaitForExit(120000)) {
            # Kill only the process tree owned by this invocation.
            $kill = New-Object Diagnostics.ProcessStartInfo
            $kill.FileName = 'taskkill.exe'
            $kill.Arguments = '/PID ' + $proc.Id + ' /T /F'
            $kill.UseShellExecute = $false
            $kill.CreateNoWindow = $true
            $kill.RedirectStandardOutput = $true
            $kill.RedirectStandardError = $true
            $killer = [Diagnostics.Process]::Start($kill)
            $ko = $killer.StandardOutput.ReadToEndAsync()
            $ke = $killer.StandardError.ReadToEndAsync()
            $null = $killer.WaitForExit(10000)
            $killer.Dispose()
            throw 'Git bundle fetch timed out.'
        }
        $copy.GetAwaiter().GetResult()
        $null = $stderr.GetAwaiter().GetResult()
        if ($proc.ExitCode -ne 0) { throw 'Git bundle fetch failed; check remote access.' }
    } finally { if ($null -ne $stream) { $stream.Dispose() }; $proc.Dispose() }
}
$partFiles = @()
foreach ($part in $bootstrap.parts) {
    $target = Join-Path $cache ([IO.Path]::GetFileName([string]$part.path))
    if (-not (Test-Blob $target $part.sha256 $part.bytes)) {
        $local = Join-Path $repo ([string]$part.path)
        if (Test-Blob $local $part.sha256 $part.bytes) { [IO.File]::Copy($local, $target, $true) }
        else {
            $ok = $false
            for ($attempt=1; $attempt -le 3; $attempt++) {
                try {
                    Fetch-Blob $part.path ($target + '.tmp')
                    if (-not (Test-Blob ($target + '.tmp') $part.sha256 $part.bytes)) { throw 'Bundle SHA256/size mismatch.' }
                    [IO.File]::Copy(($target + '.tmp'), $target, $true)
                    $ok = $true
                    break
                } catch { if ($attempt -eq 3) { throw }; Write-Host "Retry bootstrap part $attempt/3" }
            }
            if (-not $ok) { throw 'Bootstrap fetch incomplete.' }
        }
    }
    $partFiles += $target
}
$archive = Join-Path $cache 'python-base.zip'
$out = [IO.File]::Create($archive)
try { foreach ($file in $partFiles) { $src = [IO.File]::OpenRead($file); try { $src.CopyTo($out) } finally { $src.Dispose() } } } finally { $out.Dispose() }
if ((Get-Sha256 $archive) -ne $bootstrap.archive_sha256) { throw 'Bootstrap archive SHA256 mismatch.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$runtime = Join-Path $dest 'runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    foreach ($entry in $zip.Entries) {
        $name = $entry.FullName.Replace('/', '\')
        if ($name -notmatch '^python\\' -or $name.Contains(':') -or $name -match '(^|\\)\.\.(\\|$)' -or [IO.Path]::IsPathRooted($name)) { throw 'Unsafe bootstrap ZIP entry.' }
        $path = [IO.Path]::GetFullPath((Join-Path $runtime $name))
        if (-not $path.StartsWith($runtime.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'ZIP traversal rejected.' }
        if ($entry.Name -eq '') { New-Item -ItemType Directory -Force -Path $path | Out-Null }
        else {
            New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($path)) | Out-Null
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $path, $true)
        }
    }
} finally { $zip.Dispose() }
$python = Join-Path $runtime 'python\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Bootstrap python.exe missing.' }
$restoreArgs = @('-I', '-S', '-B', (Join-Path $PSScriptRoot 'restore.py'), '--repo', $repo, '--destination', $dest)
if ($VerifyOnly) { $restoreArgs += '--verify-only' }
& $python @restoreArgs
exit $LASTEXITCODE
