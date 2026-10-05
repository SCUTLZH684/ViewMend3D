param(
    [string]$SshHost = 'JM-New-Outside',
    [string]$SshConfig = '',
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$uiUrl = "http://127.0.0.1:$Port"
function Get-ViewerStatus {
    try { return Invoke-RestMethod "$uiUrl/api/status" -TimeoutSec 10 }
    catch { return $null }
}
$viewerStatus = Get-ViewerStatus
if (-not $viewerStatus) {
    $sshArgs = @('-N', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ExitOnForwardFailure=yes',
                 '-o', 'ServerAliveInterval=30', '-L', "127.0.0.1:${Port}:127.0.0.1:${Port}")
    if ($SshConfig) {
        $resolvedConfig = (Resolve-Path -LiteralPath $SshConfig).Path
        $sshArgs += @('-F', ('"' + $resolvedConfig + '"'))
    }
    $sshArgs += $SshHost
    $tunnelProcess = Start-Process -FilePath 'ssh' -ArgumentList $sshArgs -WindowStyle Hidden -PassThru
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        Start-Sleep -Milliseconds 500
        $viewerStatus = Get-ViewerStatus
        if ($viewerStatus -or $tunnelProcess.HasExited) { break }
    }
    if (-not $viewerStatus) { throw '界面未连接。请检查 SSH 密钥，并确认服务器 scripts/web/server.py 已启动。' }
}
if ($viewerStatus.method -ne 'ActiveGS confidence' -or $viewerStatus.scene -ne 'Replica office0') {
    throw '此端口不是预期的 ViewMend3D 服务，请选择其他端口。'
}
Write-Output "ViewMend3D: $uiUrl"
Start-Process $uiUrl
