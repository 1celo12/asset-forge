# Short venv path avoids Windows MAX_PATH errors when loading torch DLLs from deep folders.
$env:UV_PROJECT_ENVIRONMENT = "C:\af\venv"
if (-not $env:LIBRARY_DIR) { $env:LIBRARY_DIR = "$env:USERPROFILE\Documents\Sissies Sweets Assets" }
uv run uvicorn server:app --host 127.0.0.1 --port 8790
