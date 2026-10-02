# Asset Forge

Local web-asset generator. Text, SVG and HTML/CSS come from Ollama models; images come from SDXL-Turbo via `diffusers`.

## Run

```powershell
uv sync        # first time (Python 3.12, CUDA 12.8 torch)
./start.ps1    # starts the server and opens http://127.0.0.1:8790
```

Env vars: `IMAGE_MODEL` (default `stabilityai/sdxl-turbo`, non-commercial license), `OLLAMA_URL`, `DEFAULT_MODEL` (used by MCP tools when no model is given).

`start.ps1` and `run.ps1` set `UV_PROJECT_ENVIRONMENT=C:\af\venv`: a short venv path avoids Windows path-length errors when loading torch DLLs.

## MCP

Streamable HTTP endpoint on the same port: `http://127.0.0.1:8790/mcp/`

Tools: `generate_image`, `generate_svg`, `generate_html`, `generate_copy`, `list_models`. Files are saved to `outputs/`.

```bash
claude mcp add --transport http asset-forge http://127.0.0.1:8790/mcp/
```
