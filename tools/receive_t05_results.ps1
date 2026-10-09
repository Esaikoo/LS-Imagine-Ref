<# Receive one verified result bundle through existing OpenSSH configuration.
No password or key is saved. Run this on Windows, not inside the server terminal.
The JSON embeds all result tables/reports/images and requires no renaming.
#>
[CmdletBinding()]
param(
    [string]$Server = 'T2-3090',
    [Parameter(Mandatory = $true)][string]$RemoteIndex,
    [string]$Destination = (Join-Path $env:USERPROFILE 'Downloads\T05Results')
)
$ErrorActionPreference = 'Stop'
if ($Server -notmatch '^[A-Za-z0-9_-]+$' -or $RemoteIndex -notmatch '^/[A-Za-z0-9_./-]+/bundle_index\.json$') {
    throw 'Use a configured SSH alias and an absolute bundle_index.json path without spaces.'
}
$taskDestination = [IO.Path]::GetFullPath($Destination)
[IO.Directory]::CreateDirectory($taskDestination) | Out-Null
$taskFolder = Join-Path $taskDestination ('received_' + (Get-Date -Format 'yyyyMMddTHHmmss_fffffff'))
if (Test-Path -LiteralPath $taskFolder) { throw 'Refusing to overwrite a receipt directory.' }
[IO.Directory]::CreateDirectory($taskFolder) | Out-Null
$taskIndexPath = Join-Path $taskFolder 'bundle_index.json'
& scp.exe -o StrictHostKeyChecking=yes -o ConnectTimeout=15 "${Server}:$RemoteIndex" $taskIndexPath
if ($LASTEXITCODE -ne 0) { throw 'SSH login/transfer failed. Existing source files were not changed.' }
$taskIndex = Get-Content -LiteralPath $taskIndexPath -Raw | ConvertFrom-Json
if ($taskIndex.format -ne 'ls_imagine_result_bundle_v1' -or
    $taskIndex.json_path -notmatch '^/[A-Za-z0-9_./-]+/T05_[A-Za-z0-9_-]+_results\.json$' -or
    $taskIndex.json_sha256 -notmatch '^[a-f0-9]{64}$') { throw 'Invalid bundle index.' }
$taskName = [IO.Path]::GetFileName($taskIndex.json_path)
$taskPartial = Join-Path $taskFolder ($taskName + '.partial')
& scp.exe -o StrictHostKeyChecking=yes -o ConnectTimeout=15 "${Server}:$($taskIndex.json_path)" $taskPartial
if ($LASTEXITCODE -ne 0) { throw 'Incomplete transfer retained as .partial; do not submit it.' }
$taskHash = (Get-FileHash -LiteralPath $taskPartial -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskHash -ne $taskIndex.json_sha256) { throw 'Downloaded bundle SHA256 mismatch; partial file retained.' }
$taskBundle = Get-Content -LiteralPath $taskPartial -Raw -Encoding UTF8 | ConvertFrom-Json
if ($taskBundle.format -ne $taskIndex.format -or $taskBundle.file_count -ne $taskIndex.file_count -or
    $taskBundle.files.Count -ne $taskIndex.file_count) { throw 'Downloaded bundle inventory mismatch.' }
$taskSeen = @{}
$taskHasher = [Security.Cryptography.SHA256]::Create()
try {
    foreach ($taskEntry in $taskBundle.files) {
        $taskKey = $taskEntry.source + '/' + $taskEntry.name
        if ($taskSeen.ContainsKey($taskKey)) { throw "Duplicate bundle entry: $taskKey" }
        $taskSeen[$taskKey] = $true
        if ($taskEntry.encoding -eq 'base64') { $taskBytes = [Convert]::FromBase64String($taskEntry.content) }
        elseif ($taskEntry.encoding -eq 'utf-8') { $taskBytes = [Text.Encoding]::UTF8.GetBytes($taskEntry.content) }
        else { throw 'Unsupported bundle encoding.' }
        $taskActual = ([BitConverter]::ToString($taskHasher.ComputeHash($taskBytes))).Replace('-', '').ToLowerInvariant()
        if ($taskBytes.Length -ne $taskEntry.bytes -or $taskActual -ne $taskEntry.sha256) {
            throw "Source entry SHA256 mismatch: $taskKey"
        }
    }
} finally { $taskHasher.Dispose() }
$taskFinal = Join-Path $taskFolder $taskName
Move-Item -LiteralPath $taskPartial -Destination $taskFinal
Write-Output "SAVED_RESULT=$taskFinal"
Write-Output "Verified $($taskIndex.file_count) files; attach this one JSON or tell Codex this local path."
