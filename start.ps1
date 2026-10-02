# Starts Asset Forge (if not already running) and opens the UI.
$port = 8790
Set-Location $PSScriptRoot
# Short venv path avoids Windows MAX_PATH errors when loading torch DLLs.
$env:UV_PROJECT_ENVIRONMENT = "C:\af\venv"
# Folder Latent Library watches; new images are copied here with embedded generation metadata.
if (-not $env:LIBRARY_DIR) { $env:LIBRARY_DIR = "$env:USERPROFILE\Documents\Sissies Sweets Assets" }

if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
    if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    }
    Start-Process -FilePath uv -ArgumentList "run", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", $port -WindowStyle Minimized
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep 1
        if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { break }
    }
}
Start-Process "http://127.0.0.1:$port"
