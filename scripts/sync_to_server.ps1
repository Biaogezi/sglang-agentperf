param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [Parameter(Mandatory = $true)][string]$KeyPath,
    [string]$UserName = "ecs-user",
    [string]$RemotePath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SnapshotCommit = (git -C $ProjectRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "Project must be a Git checkout" }
if (-not $RemotePath) { $RemotePath = "/data/sglang-agentperf-$($SnapshotCommit.Substring(0, 12))" }
if ($RemotePath -notmatch '^/(data|home)/[A-Za-z0-9_./-]+$' -or $RemotePath.Contains('..')) {
    throw "Use an explicit fresh directory under /data or /home"
}
$SnapshotName = "sglang-agentperf-$([guid]::NewGuid().ToString('N')).tar.gz"
$ArchivePath = Join-Path ([System.IO.Path]::GetTempPath()) $SnapshotName
$RemoteArchive = "/tmp/$SnapshotName"
$TrackedFiles = git -C $ProjectRoot ls-tree -r --name-only HEAD
if ($TrackedFiles | Where-Object { $_ -match '(^|/)(\.env|[^/]+\.(pem|key))$' }) {
    throw "Snapshot includes a potential credential file; inspect the committed files before upload"
}

try {
    # Only committed project files; never .git, local secrets, raw results or model caches.
    git -C $ProjectRoot -c core.autocrlf=false archive --format=tar.gz --output=$ArchivePath HEAD
    if ($LASTEXITCODE -ne 0) { throw "Failed to create sync archive" }
    ssh -i $KeyPath "$UserName@$HostName" "mkdir '$RemotePath'"
    if ($LASTEXITCODE -ne 0) { throw "Failed to create remote project directory" }
    scp -i $KeyPath $ArchivePath "$UserName@${HostName}:$RemoteArchive"
    if ($LASTEXITCODE -ne 0) { throw "Failed to upload sync archive" }
    ssh -i $KeyPath "$UserName@$HostName" `
        "tar -xzf '$RemoteArchive' -C '$RemotePath' && rm '$RemoteArchive'"
    if ($LASTEXITCODE -ne 0) { throw "Failed to extract sync archive" }
}
finally {
    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }
}

Write-Host "Uploaded committed snapshot $SnapshotCommit to $UserName@$HostName`:$RemotePath"
Write-Host "Existing deployments are not overwritten. Configure .env and upstream checkout separately."
