"""Smoke tests against a running Asset Forge server. Usage: uv run python smoke_test.py [base_url]"""
import asyncio, sys, time
import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8790"
results = []


def check(name, fn):
    t = time.time()
    try:
        detail = fn() or ""
        results.append((name, True, f"{time.time() - t:.1f}s {detail}"))
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))


c = httpx.Client(base_url=BASE, timeout=600)
state = {}


def t_index():
    r = c.get("/")
    assert r.status_code == 200 and "Asset Forge" in r.text


def t_models():
    r = c.get("/api/models")
    r.raise_for_status()
    state["models"] = r.json()["models"]
    assert state["models"], "no ollama models"
    state["model"] = state["models"][0]
    return f"{len(state['models'])} models"


def text_test(kind, validator):
    def run():
        r = c.post("/api/text", json={"kind": kind, "prompt": "a tiny coffee shop landing section" if kind != "svg" else "a simple star icon", "model": state["model"]})
        r.raise_for_status()
        j = r.json()
        validator(j["result"])
        assert c.get(j["file"]).status_code == 200, "saved file not served"
        return f"{len(j['result'])} chars"
    return run


def v_svg(s):
    assert s.lstrip().startswith("<svg") and s.rstrip().endswith("</svg>"), "not a bare <svg>"


def v_html(s):
    assert "<" in s and "```" not in s, "html not extracted cleanly"


def v_copy(s):
    assert len(s.strip()) > 10 and "<think>" not in s


def t_bad_kind():
    assert c.post("/api/text", json={"kind": "nope", "prompt": "x", "model": "x"}).status_code == 400


def t_ollama_down_error():
    r = c.post("/api/text", json={"kind": "copy", "prompt": "x", "model": "no-such-model-xyz"})
    assert r.status_code == 502, f"expected 502, got {r.status_code}"


def t_image():
    r = c.post("/api/image", json={"prompt": "flat vector style mountain logo, white background", "seed": 7})
    r.raise_for_status()
    j = r.json()
    f = c.get(j["file"])
    assert f.status_code == 200 and f.content[:4] == b"\x89PNG", "not a PNG"
    state["img"] = j
    from io import BytesIO
    from PIL import Image
    params = Image.open(BytesIO(f.content)).text.get("parameters", "")
    assert "Seed: 7" in params and "Steps:" in params and "Sampler:" in params, f"bad metadata: {params!r}"
    if j.get("library"):
        from pathlib import Path
        lib = Path(j["library"])
        assert lib.exists() and Image.open(lib).text.get("parameters") == params, "library copy missing/incorrect"
        assert not list(lib.parent.glob("*.tmp")), "stray .tmp left in watched folder"
    return f"{len(f.content) // 1024} KB"


def t_image_seed_repro():
    r = c.post("/api/image", json={"prompt": "flat vector style mountain logo, white background", "seed": 7})
    r.raise_for_status()
    assert r.json()["seed"] == 7


def t_outputs():
    files = c.get("/api/outputs").json()["files"]
    assert state["img"]["file"] in files


def t_traversal():
    r = c.get("/outputs/..%2Fserver.py")
    assert r.status_code in (400, 404), f"traversal returned {r.status_code}"
    assert "FastMCP" not in r.text


def t_mcp():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def go():
        async with streamablehttp_client(f"{BASE}/mcp/") as (rd, wr, _):
            async with ClientSession(rd, wr) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                want = {"generate_image", "generate_svg", "generate_html", "generate_copy", "list_models"}
                assert want <= names, f"missing {want - names}"
                out = await s.call_tool("list_models", {})
                assert not out.isError
                out = await s.call_tool("generate_copy", {"prompt": "a 6-word tagline for a bakery"})
                assert not out.isError and "saved:" in out.content[0].text
                return len(names)
    return f"{asyncio.run(go())} tools"


check("GET / serves UI", t_index)
check("GET /api/models", t_models)
if state.get("model"):
    check("POST /api/text svg", text_test("svg", v_svg))
    check("POST /api/text html", text_test("html", v_html))
    check("POST /api/text copy", text_test("copy", v_copy))
    check("unknown model -> 502", t_ollama_down_error)
check("unknown kind -> 400", t_bad_kind)
check("POST /api/image", t_image)
if "img" in state:
    check("image seed echoed", t_image_seed_repro)
    check("image in /api/outputs", t_outputs)
check("path traversal blocked", t_traversal)
check("MCP tools + calls", t_mcp)

for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:28} {detail}")
failed = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - failed}/{len(results)} passed")
sys.exit(1 if failed else 0)
