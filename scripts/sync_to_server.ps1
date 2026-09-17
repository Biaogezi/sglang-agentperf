param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [Parameter(Mandatory = $true)][string]$KeyPath,
    [string]$UserName = "ecs-user",
    [string]$RemotePath = "/data/sglang-agentperf"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ArchivePath = Join-Path ([System.IO.Path]::GetTempPath()) "sglang-agentperf-sync.tar.gz"

try {
    tar --format=ustar --exclude=.venv --exclude=results --exclude=profiles --exclude=quality `
        --exclude='upstream/*.bundle' `
        --exclude=upstream/sglang -czf $ArchivePath -C $ProjectRoot .
    if ($LASTEXITCODE -ne 0) { throw "Failed to create sync archive" }
    ssh -i $KeyPath "$UserName@$HostName" "mkdir -p '$RemotePath'"
    if ($LASTEXITCODE -ne 0) { throw "Failed to create remote project directory" }
    scp -i $KeyPath $ArchivePath "$UserName@${HostName}:/tmp/sglang-agentperf-sync.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw "Failed to upload sync archive" }
    ssh -i $KeyPath "$UserName@$HostName" `
        "tar -xzf /tmp/sglang-agentperf-sync.tar.gz -C '$RemotePath' && rm /tmp/sglang-agentperf-sync.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw "Failed to extract sync archive" }
}
finally {
    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }
}

Write-Host "Synced to $UserName@$HostName`:$RemotePath"
