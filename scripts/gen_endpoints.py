"""Export the HTTP contract of the Python backend for the Go port.

Two artifacts, both generated from the LIVE app so they cannot drift:

  docs/openapi.contract.json — the OpenAPI schema. The Go server must
      serve a schema-compatible response for every operation in it.
      This is the acceptance fixture for the Go test suite.

  docs/ENDPOINTS.md — a human-readable inventory (path, method, auth,
      source file) for the port checklist.

Run:  python scripts/gen_endpoints.py
"""
import ast
import json
import os
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PUBLIC_URL", "http://localhost:8080")

DOCS = pathlib.Path(r"D:\Agents AI Go\docs")
HTTP_METHODS = {"get", "post", "put", "delete", "patch"}


def re_path(decorator_src: str):
    """Pull (verb, path) out of a decorator like api_router.post('/x')."""
    return re.search(r"\.(\w+)\(\s*[\"']([^\"']+)[\"']", decorator_src)


from STT_server.STT_Server import app  # noqa: E402

# ── 1. the OpenAPI schema: the authoritative contract ───────────────────
schema = app.openapi()
(DOCS / "openapi.contract.json").write_text(
    json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)

# ── 2. router prefixes, read from the source ────────────────────────────
prefix_of = {}
for mod, var in (("STT_server/routes/api.py", "api_router"),
                 ("STT_server/routes/auth.py", "router")):
    src = (ROOT / mod).read_text(encoding="utf-8-sig")
    pm = re.search(rf"{var}\s*=\s*APIRouter\(([^)]*)\)", src)
    pref = ""
    if pm:
        am = re.search(r"prefix\s*=\s*[\"']([^\"']+)[\"']", pm.group(1))
        if am:
            pref = am.group(1)
    prefix_of[mod] = pref

# ── 3. per-operation metadata from the source ───────────────────────────
meta = {}
for f in sorted((ROOT / "STT_server").rglob("*.py")):
    rel = str(f.relative_to(ROOT)).replace("\\", "/")
    try:
        tree = ast.parse(f.read_text(encoding="utf-8-sig"))
    except SyntaxError:
        continue
    prefix = prefix_of.get(rel, "")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            s = ast.unparse(d)
            m = re_path(s)
            if not m:
                continue
            verb = m.group(1).lower()
            if verb == "websocket":
                meta[("WS", m.group(2))] = {
                    "handler": node.name, "file": rel, "auth": "—", "ws": True,
                }
            elif verb in HTTP_METHODS:
                meta[(verb.upper(), prefix + m.group(2))] = {
                    "handler": node.name,
                    "file": rel,
                    "auth": "Bearer" if "require_auth" in s else "—",
                    "ws": False,
                }

# ── 4. write the markdown inventory ─────────────────────────────────────
rows = []
for path, ops in schema.get("paths", {}).items():
    for verb in ops:
        m = meta.get((verb.upper(), path))
        rows.append((
            verb.upper(), path,
            (m or {}).get("auth", "—"),
            (m or {}).get("handler", "?"),
            (m or {}).get("file", "?"),
        ))
for (verb, path), m in meta.items():
    if m.get("ws"):
        rows.append(("WS", path, "—", m["handler"], m["file"]))

rows.sort(key=lambda r: (r[1], r[0]))

fams = {}
for row in rows:
    seg = row[1].strip("/").split("/")[0] if row[1].strip("/") else "(root)"
    fams.setdefault(seg, []).append(row)

lines = [
    "# Inventario de endpoints — contrato a preservar",
    "",
    "Generado con `python scripts/gen_endpoints.py` desde el backend Python.",
    "",
    "El backend Go debe servir **exactamente** esta lista: mismo path, mismos",
    "métodos, misma dependencia de auth, y respuestas equivalentes (cuerpos,",
    "códigos de estado, forma de los errores). Ver `DESIGN.md` §6.",
    "",
    "El schema OpenAPI completo esta en `openapi.contract.json` — ese JSON es",
    "el fixture de aceptacion de las pruebas de contrato en Go.",
    "",
    f"**Total: {len(rows)} operaciones** en {len(schema.get('paths', {}))} paths "
    f"({sum(1 for r in rows if r[0] == 'WS')} WebSocket, "
    f"{sum(1 for r in rows if r[2] == 'Bearer')} con Bearer).",
    "",
]
for fam in sorted(fams):
    items = fams[fam]
    lines.append(f"## `/{fam}` ({len(items)})")
    lines.append("")
    lines.append("| Metodo | Path | Auth | Handler | Archivo |")
    lines.append("|---|---|---|---|---|")
    for verb, path, auth, handler, src in items:
        lines.append(f"| `{verb}` | `{path}` | {auth} | `{handler}` | `{src}` |")
    lines.append("")

DOCS.mkdir(parents=True, exist_ok=True)
(DOCS / "ENDPOINTS.md").write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {DOCS / 'openapi.contract.json'} ({len(schema.get('paths', {}))} paths)")
print(f"wrote {DOCS / 'ENDPOINTS.md'} ({len(rows)} operations)")
