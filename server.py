"""Asset Forge: local web-asset generator (Ollama for text/SVG/HTML, diffusers for images)."""
import contextlib, os, re, shutil, time, uuid, threading
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel

OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "stabilityai/sdxl-turbo")
# Folder watched by Latent Library; generated images are copied here (unset = disabled).
LIBRARY_DIR = Path(os.environ["LIBRARY_DIR"]) if os.getenv("LIBRARY_DIR") else None
ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("asset-forge", streamable_http_path="/", stateless_http=True)


@contextlib.asynccontextmanager
async def lifespan(_app):
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="Asset Forge", lifespan=lifespan)

SYSTEM = {
    "svg": "You design clean, minimal SVG graphics for websites. Reply with ONE complete <svg> element only: "
           "include viewBox, no width/height, no external resources, no scripts. Prefer currentColor for icons.",
    "html": "You write modern, accessible, responsive HTML+CSS components. Reply with ONE self-contained HTML "
            "snippet in a single ```html block: inline <style>, no external assets or frameworks, no JavaScript "
            "unless requested.",
    "copy": "You are a concise web copywriter. Give the requested copy (headlines, meta description, alt text, etc.) "
            "with no preamble. Use short labelled lines when multiple items are requested.",
}


class TextReq(BaseModel):
    kind: str  # svg | html | copy
    prompt: str
    model: str
    temperature: float = 0.7


class ImageReq(BaseModel):
    prompt: str
    negative: str = ""
    width: int = 512
    height: int = 512
    steps: int = 4
    guidance: float = 0.0
    seed: int | None = None


def strip_think(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()


def extract(kind: str, s: str) -> str:
    s = strip_think(s)
    if kind == "svg":
        m = re.search(r"<svg\b.*?</svg>", s, re.S | re.I)
        return m.group(0) if m else s
    if kind == "html":
        m = re.search(r"```(?:html)?\s*(.*?)```", s, re.S)
        return (m.group(1) if m else s).strip()
    return s


@app.get("/api/models")
def models():
    try:
        r = httpx.get(f"{OLLAMA}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"Ollama unreachable at {OLLAMA}: {e}")
    return {"models": [m["name"] for m in r.json()["models"]], "image_model": IMAGE_MODEL}


@app.post("/api/text")
def text(req: TextReq):
    if req.kind not in SYSTEM:
        raise HTTPException(400, "kind must be svg, html or copy")
    body = {
        "model": req.model,
        "stream": False,
        "think": False,
        "options": {"temperature": req.temperature},
        "messages": [
            {"role": "system", "content": SYSTEM[req.kind]},
            {"role": "user", "content": req.prompt},
        ],
    }
    try:
        r = httpx.post(f"{OLLAMA}/api/chat", json=body, timeout=600)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"Ollama error: {e}")
    result = extract(req.kind, r.json()["message"]["content"])
    ext = {"svg": "svg", "html": "html", "copy": "txt"}[req.kind]
    name = f"{req.kind}-{int(time.time())}-{uuid.uuid4().hex[:6]}.{ext}"
    (OUT / name).write_text(result, encoding="utf-8")
    return {"result": result, "file": f"/outputs/{name}"}


_pipe = None
_lock = threading.Lock()


def free_ollama_vram():
    """Unload any resident Ollama models so the diffusion model gets the GPU."""
    try:
        for m in httpx.get(f"{OLLAMA}/api/ps", timeout=5).json().get("models", []):
            httpx.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def get_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image

        _pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL, torch_dtype=torch.float16, variant="fp16"
        )
        _pipe.enable_model_cpu_offload()  # fits 12 GB VRAM comfortably; 63 GB RAM available
    return _pipe


@app.post("/api/image")
def image(req: ImageReq):
    import torch

    with _lock:
        free_ollama_vram()
        pipe = get_pipe()
        seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "little") % 2**31
        gen = torch.Generator("cpu").manual_seed(seed)
        kwargs = dict(
            prompt=req.prompt,
            width=req.width - req.width % 8,
            height=req.height - req.height % 8,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            generator=gen,
        )
        if req.negative and req.guidance > 1:
            kwargs["negative_prompt"] = req.negative
        img = pipe(**kwargs).images[0]
    name = f"img-{int(time.time())}-{seed}.png"
    # A1111-style "parameters" PNG text chunk: the format Latent Library's metadata parser reads.
    params = f"{req.prompt}\n"
    if "negative_prompt" in kwargs:
        params += f"Negative prompt: {req.negative}\n"
    params += (
        f"Steps: {req.steps}, Sampler: {SAMPLERS.get(type(pipe.scheduler).__name__, type(pipe.scheduler).__name__)}, "
        f"CFG scale: {req.guidance}, Seed: {seed}, Size: {kwargs['width']}x{kwargs['height']}, "
        f"Model: {IMAGE_MODEL.split('/')[-1]}, Version: Asset Forge"
    )
    meta = PngInfo()
    meta.add_text("parameters", params)
    img.save(OUT / name, pnginfo=meta)
    in_library = publish_to_library(OUT / name)
    return {"file": f"/outputs/{name}", "seed": seed, "library": in_library}


SAMPLERS = {"EulerAncestralDiscreteScheduler": "Euler a", "EulerDiscreteScheduler": "Euler",
            "DPMSolverMultistepScheduler": "DPM++ 2M", "DDIMScheduler": "DDIM"}


def publish_to_library(path: Path) -> str | None:
    """Copy a finished image into the Latent Library watched folder (atomic: temp name, then rename)."""
    if not LIBRARY_DIR:
        return None
    try:
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        dest = LIBRARY_DIR / f"assetforge-{path.name}"
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copyfile(path, tmp)
        os.replace(tmp, dest)
        return str(dest)
    except OSError as e:
        print(f"Latent Library publish failed: {e}")
        return None


def default_model() -> str:
    return os.getenv("DEFAULT_MODEL") or models()["models"][0]


def _text_tool(kind: str, prompt: str, model: str | None) -> str:
    r = text(TextReq(kind=kind, prompt=prompt, model=model or default_model()))
    return f"{r['result']}\n\n(saved: {OUT / Path(r['file']).name})"


@mcp.tool()
def generate_image(prompt: str, width: int = 512, height: int = 512, steps: int = 4,
                   guidance: float = 0.0, seed: int | None = None) -> str:
    """Generate a raster image (hero art, backgrounds) locally. Returns the saved PNG path."""
    r = image(ImageReq(prompt=prompt, width=width, height=height, steps=steps, guidance=guidance, seed=seed))
    lib = f", Latent Library: {r['library']}" if r["library"] else ""
    return f"{OUT / Path(r['file']).name} (seed {r['seed']}{lib})"


@mcp.tool()
def generate_svg(prompt: str, model: str | None = None) -> str:
    """Generate an SVG icon/logo/graphic with a local LLM. Returns the SVG markup and saved path."""
    return _text_tool("svg", prompt, model)


@mcp.tool()
def generate_html(prompt: str, model: str | None = None) -> str:
    """Generate a self-contained HTML/CSS component with a local LLM."""
    return _text_tool("html", prompt, model)


@mcp.tool()
def generate_copy(prompt: str, model: str | None = None) -> str:
    """Generate web copy (headlines, meta descriptions, alt text) with a local LLM."""
    return _text_tool("copy", prompt, model)


@mcp.tool()
def list_models() -> list[str]:
    """List the local Ollama models available for text/SVG/HTML generation."""
    return models()["models"]


@app.get("/api/outputs")
def outputs():
    files = sorted(OUT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:60]
    return {"files": [f"/outputs/{p.name}" for p in files]}


app.mount("/mcp", mcp.streamable_http_app())
app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
