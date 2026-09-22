from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path
from urllib.parse import urlencode

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Route
import uvicorn

try:
    from .langchain_mcp_agent import ask_question
except Exception:
    from langchain_mcp_agent import ask_question

INDEX_HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GeoLifeCLEF MCP Chat</title>
  <style>
    :root { --bg:#0e1116; --card:#171b22; --text:#e6edf3; --muted:#8b949e; --accent:#2f81f7; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: radial-gradient(1200px 600px at 20% -10%, #1f2a44 0%, var(--bg) 55%); color:var(--text); }
    .wrap { max-width: 980px; margin: 28px auto; padding: 0 16px; }
    .card { background: var(--card); border: 1px solid #2d333b; border-radius: 14px; padding: 16px; }
    h1 { margin: 0 0 14px; font-size: 22px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; }
    input, textarea, button, select { border-radius:10px; border:1px solid #30363d; background:#0d1117; color:var(--text); padding:10px 12px; }
    textarea { width:100%; min-height:90px; resize:vertical; }
    input { min-width: 220px; }
    input[type="checkbox"] { min-width: auto; width: auto; padding: 0; accent-color: var(--accent); }
    button { background: var(--accent); border:none; font-weight:600; cursor:pointer; }
    button:disabled { opacity:.6; cursor:not-allowed; }
    .muted { color: var(--muted); font-size: 13px; }
    .chat { margin-top: 16px; display:grid; gap:12px; }
    .msg { border:1px solid #30363d; border-radius:10px; padding:10px; background:#0d1117; }
    .msg .label { font-size:12px; color:var(--muted); margin-bottom:6px; }
    .artifacts { margin-top:10px; display:grid; gap:8px; }
    .artifacts a { color:#2f81f7; text-decoration:none; word-break:break-all; font-size:12px; }
    .artifacts img { max-width:100%; height:auto; border:1px solid #30363d; border-radius:8px; background:#0b0f14; }
    pre { white-space: pre-wrap; word-break: break-word; margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>GeoLifeCLEF MCP Chat</h1>
      <p class="muted">UI locale gratuite: Ollama + MCP server + LangChain/LangSmith (optionnel).</p>
      <p id="jsStatus" class="muted">JS: chargement...</p>
      <form id="askForm" method="post" action="/ask_sync">
        <div class="row" style="margin-bottom:10px;">
        <select id="provider" name="provider">
          <option value="ollama" __SEL_OLLAMA__>ollama</option>
          <option value="openrouter" __SEL_OPENROUTER__>openrouter</option>
        </select>
        <input id="model" name="model" value="__INIT_MODEL__" />
        <input id="baseUrl" name="ollama_base_url" value="__INIT_OLLAMA_BASE_URL__" />
        <input id="openrouterBaseUrl" name="openrouter_base_url" value="__INIT_OPENROUTER_BASE_URL__" />
      </div>
      <textarea id="question" name="question" placeholder="Pose une question sur les données ou le modèle...">__INIT_QUESTION__</textarea>
      <div class="row" style="margin-top:10px;">
        <label style="display:flex; align-items:center; gap:8px; color:#8b949e;">
          <input id="strictTools" name="strict_tools" type="checkbox" __INIT_STRICT_CHECKED__ />
          Strict tools
        </label>
        <button id="askBtn" type="submit" onclick="if (window.__mcpAsk) { window.__mcpAsk(); return false; } return true;">Envoyer</button>
      </div>
      </form>
      <div id="chat" class="chat"></div>
    </div>
  </div>

  <script>
    window.__mcpAsk = function () {
      const chatEl = document.getElementById('chat');
      if (!chatEl) return;
      const div = document.createElement('div');
      div.className = 'msg';
      div.innerHTML = `<div class="label">Erreur UI</div><pre>Le handler JS n'est pas initialisé. Recharge la page (Cmd+Shift+R).</pre>`;
      chatEl.prepend(div);
    };

    const askForm = document.getElementById('askForm');
    const askBtn = document.getElementById('askBtn');
    const chat = document.getElementById('chat');
    const providerEl = document.getElementById('provider');
    const modelEl = document.getElementById('model');
    const jsStatus = document.getElementById('jsStatus');
    if (jsStatus) jsStatus.textContent = 'JS: actif (chat inline)';
    const INIT_PROVIDER = "__INIT_PROVIDER__";
    const INIT_STRICT_TOOLS = "__INIT_STRICT_TOOLS__";
    if (INIT_PROVIDER === "openrouter" || INIT_PROVIDER === "ollama") {
      providerEl.value = INIT_PROVIDER;
    }
    if (INIT_STRICT_TOOLS === "0" || INIT_STRICT_TOOLS === "1") {
      document.getElementById('strictTools').checked = INIT_STRICT_TOOLS === "1";
    }
    const DEFAULT_MODELS = {
      ollama: 'qwen3:8b',
      openrouter: 'google/gemma-4-31b-it:free',
    };
    let currentProvider = providerEl.value;
    const lastModelByProvider = {
      ollama: DEFAULT_MODELS.ollama,
      openrouter: DEFAULT_MODELS.openrouter,
    };
    const userEditedModelByProvider = {
      ollama: false,
      openrouter: false,
    };

    function normalizeModelForProvider(provider) {
      const v = (modelEl.value || '').trim();
      if (provider === 'openrouter') {
        if (!v || v === DEFAULT_MODELS.ollama) {
          modelEl.value = DEFAULT_MODELS.openrouter;
        }
      } else if (provider === 'ollama') {
        if (!v || v === DEFAULT_MODELS.openrouter) {
          modelEl.value = DEFAULT_MODELS.ollama;
        }
      }
    }

    function syncProviderModelDefaults() {
      const p = (providerEl.value || '').trim();
      if (p === 'openrouter') {
        const v = (modelEl.value || '').trim();
        // Cas Safari/autofill: provider restauré à openrouter mais modèle laissé à qwen.
        if (!v || v === DEFAULT_MODELS.ollama) {
          modelEl.value = DEFAULT_MODELS.openrouter;
        }
      } else if (p === 'ollama') {
        const v = (modelEl.value || '').trim();
        if (!v || v === DEFAULT_MODELS.openrouter) {
          modelEl.value = DEFAULT_MODELS.ollama;
        }
      }
    }

    // Corrige les restaurations navigateur incohérentes au chargement.
    normalizeModelForProvider(currentProvider);
    syncProviderModelDefaults();
    // Double passe pour les restaurations tardives du navigateur.
    setTimeout(syncProviderModelDefaults, 100);
    setTimeout(syncProviderModelDefaults, 600);
    if ((modelEl.value || '').trim()) {
      lastModelByProvider[currentProvider] = modelEl.value.trim();
    }

    modelEl.addEventListener('input', () => {
      const value = modelEl.value.trim();
      if (value) {
        lastModelByProvider[currentProvider] = value;
        userEditedModelByProvider[currentProvider] = true;
      }
    });

    providerEl.addEventListener('change', () => {
      const previousValue = (modelEl.value || '').trim();
      if (previousValue) {
        lastModelByProvider[currentProvider] = previousValue;
      }
      const nextProvider = providerEl.value;
      if (userEditedModelByProvider[nextProvider]) {
        modelEl.value = lastModelByProvider[nextProvider] || DEFAULT_MODELS[nextProvider] || '';
      } else {
        modelEl.value = DEFAULT_MODELS[nextProvider] || '';
      }
      normalizeModelForProvider(nextProvider);
      syncProviderModelDefaults();
      currentProvider = nextProvider;
    });

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function isImagePath(path) {
      return /\\.(png|jpg|jpeg|gif|webp|svg)$/i.test(String(path || ''));
    }

    function addArtifactBlock(container, artifactPaths) {
      const paths = Array.isArray(artifactPaths) ? artifactPaths.filter(Boolean) : [];
      if (!paths.length) return;

      const artifactsDiv = document.createElement('div');
      artifactsDiv.className = 'artifacts';
      const title = document.createElement('div');
      title.className = 'label';
      title.textContent = 'Artifacts';
      artifactsDiv.appendChild(title);

      const seen = new Set();
      for (const path of paths) {
        const key = String(path);
        if (seen.has(key)) continue;
        seen.add(key);

        const link = document.createElement('a');
        link.href = `/artifact?${new URLSearchParams({ path: key }).toString()}`;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = key;
        artifactsDiv.appendChild(link);

        if (isImagePath(key)) {
          const img = document.createElement('img');
          img.loading = 'lazy';
          img.alt = key;
          img.src = `/artifact?${new URLSearchParams({ path: key }).toString()}`;
          artifactsDiv.appendChild(img);
        }
      }
      container.appendChild(artifactsDiv);
    }

    function addMsg(label, text, artifactPaths) {
      const div = document.createElement('div');
      div.className = 'msg';
      div.innerHTML = `<div class="label">${escapeHtml(label)}</div><pre>${escapeHtml(text)}</pre>`;
      addArtifactBlock(div, artifactPaths || []);
      chat.prepend(div);
    }

    window.__mcpAsk = async () => {
      const question = document.getElementById('question').value.trim();
      const provider = providerEl.value.trim();
      const model = modelEl.value.trim();
      const ollamaBaseUrl = document.getElementById('baseUrl').value.trim();
      const openrouterBaseUrl = document.getElementById('openrouterBaseUrl').value.trim();
      const strictTools = !!document.getElementById('strictTools').checked;
      if (!question) return;
      // Garantit le bon modèle même si le navigateur a restauré des champs incohérents.
      syncProviderModelDefaults();
      const finalModel = modelEl.value.trim();

      addMsg('Question', question);
      askBtn.disabled = true;
      try {
        const r = await fetch('/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            question,
            provider,
            model: finalModel,
            ollama_base_url: ollamaBaseUrl,
            openrouter_base_url: openrouterBaseUrl,
            strict_tools: strictTools
          })
        });
        const data = await r.json();
        const answer = data.ok
          ? data.answer
          : `${data.error}\n${data.hint || ''}${data.details ? `\n\nDetails: ${data.details}` : ''}${(data.tool_failures || []).length ? `\n\nTool failures: ${JSON.stringify(data.tool_failures)}` : ''}`;
        const artifacts = Array.isArray(data.artifact_paths) ? data.artifact_paths : [];
        const okTool = (data.ok_tool === undefined || data.ok_tool === null) ? 'n/a' : data.ok_tool;
        const strictOut = (data.strict_tools === undefined || data.strict_tools === null) ? strictTools : data.strict_tools;
        const latency = (data.latency_ms === undefined || data.latency_ms === null) ? 'n/a' : data.latency_ms;
        const meta = `ok=${data.ok} | ok_tool=${okTool} | strict_tools=${strictOut} | provider=${data.provider || provider} | model=${data.model || finalModel} | latency=${latency}ms | tools=${(data.tools_called || []).join(', ')}`;
        addMsg('Réponse', `${answer}\n\n---\n${meta}`, artifacts);
      } catch (e) {
        addMsg('Erreur', String(e));
      } finally {
        askBtn.disabled = false;
      }
    };

    // Mode dynamique JS: on intercepte le submit du formulaire.
    askForm.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      await window.__mcpAsk();
      return false;
    });

  </script>
</body>
</html>
"""


def create_app(server_path: Path, python_cmd: str) -> Starlette:
    no_cache_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    repo_root = server_path.parent.parent.resolve()

    def _as_bool(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
        return default

    def _normalize_provider(value: str) -> str:
        token = value.strip().lower()
        return token if token in {"ollama", "openrouter"} else "ollama"

    def _model_default_for_provider(provider: str) -> str:
        if provider == "openrouter":
            return os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
        return os.getenv("OLLAMA_MODEL", "qwen3:8b")

    def _render_index_html(
        *,
        provider: str,
        model: str,
        ollama_base_url: str,
        openrouter_base_url: str,
        question: str,
        strict_tools: bool,
    ) -> str:
        provider = _normalize_provider(provider)
        model = (model or "").strip() or _model_default_for_provider(provider)
        strict_value = "1" if strict_tools else "0"
        strict_checked = "checked" if strict_tools else ""

        return (
            INDEX_HTML.replace("__INIT_PROVIDER__", html.escape(provider, quote=True))
            .replace("__INIT_MODEL__", html.escape(model, quote=True))
            .replace("__INIT_OLLAMA_BASE_URL__", html.escape(ollama_base_url, quote=True))
            .replace("__INIT_OPENROUTER_BASE_URL__", html.escape(openrouter_base_url, quote=True))
            .replace("__INIT_QUESTION__", html.escape(question))
            .replace("__INIT_STRICT_TOOLS__", strict_value)
            .replace("__INIT_STRICT_CHECKED__", strict_checked)
            .replace("__SEL_OLLAMA__", "selected" if provider == "ollama" else "")
            .replace("__SEL_OPENROUTER__", "selected" if provider == "openrouter" else "")
        )

    async def homepage(request: Request) -> HTMLResponse:
        params = request.query_params
        provider_default = _normalize_provider(os.getenv("AGENT_LLM_PROVIDER", "ollama"))
        provider = _normalize_provider(str(params.get("provider") or provider_default))
        strict_default = _as_bool(os.getenv("UI_STRICT_TOOLS", "true"), True)
        strict_tools = _as_bool(params.get("strict_tools"), strict_default)

        html_doc = _render_index_html(
            provider=provider,
            model=str(params.get("model") or os.getenv("AGENT_MODEL") or _model_default_for_provider(provider)),
            ollama_base_url=str(params.get("ollama_base_url") or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
            openrouter_base_url=str(params.get("openrouter_base_url") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")),
            question=str(params.get("question") or ""),
            strict_tools=strict_tools,
        )
        return HTMLResponse(html_doc, headers=no_cache_headers)

    async def ask(request: Request) -> JSONResponse:
        payload = await request.json()
        question = str(payload.get("question", "")).strip()
        provider = str(payload.get("provider") or os.getenv("AGENT_LLM_PROVIDER", "ollama")).strip().lower()
        model_default = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free") if provider == "openrouter" else os.getenv("OLLAMA_MODEL", "qwen3:8b")
        model = str(payload.get("model") or os.getenv("AGENT_MODEL") or model_default).strip()
        ollama_base_url = str(payload.get("ollama_base_url") or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).strip()
        openrouter_base_url = str(payload.get("openrouter_base_url") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).strip()
        strict_default = _as_bool(os.getenv("UI_STRICT_TOOLS", "true"), True)
        strict_tools = _as_bool(payload.get("strict_tools"), strict_default)

        result = await ask_question(
            question,
            provider=provider,
            model=model,
            ollama_base_url=ollama_base_url,
            openrouter_base_url=openrouter_base_url,
            server_path=server_path,
            python_cmd=python_cmd,
        )
        if strict_tools and result.get("ok") and result.get("ok_tool") is False:
            result = dict(result)
            result["ok"] = False
            result["error"] = "Tool call failed in strict mode"
            result["hint"] = "At least one MCP tool returned ok=false. Check tool_failures."
        result["strict_tools"] = strict_tools
        return JSONResponse(result)

    async def ask_sync(request: Request) -> HTMLResponse:
        form = await request.form()
        question = str(form.get("question", "")).strip()
        provider = str(form.get("provider") or os.getenv("AGENT_LLM_PROVIDER", "ollama")).strip().lower()
        model_default = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free") if provider == "openrouter" else os.getenv("OLLAMA_MODEL", "qwen3:8b")
        model = str(form.get("model") or os.getenv("AGENT_MODEL") or model_default).strip()
        ollama_base_url = str(form.get("ollama_base_url") or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).strip()
        openrouter_base_url = str(form.get("openrouter_base_url") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).strip()
        strict_default = _as_bool(os.getenv("UI_STRICT_TOOLS", "true"), True)
        strict_tools = _as_bool(form.get("strict_tools"), strict_default)

        result = await ask_question(
            question,
            provider=provider,
            model=model,
            ollama_base_url=ollama_base_url,
            openrouter_base_url=openrouter_base_url,
            server_path=server_path,
            python_cmd=python_cmd,
        )
        if strict_tools and result.get("ok") and result.get("ok_tool") is False:
            result = dict(result)
            result["ok"] = False
            result["error"] = "Tool call failed in strict mode"
            result["hint"] = "At least one MCP tool returned ok=false. Check tool_failures."

        answer = str(result.get("answer") or result.get("error") or "No response")
        details = str(result.get("hint") or "")
        meta = (
            f"ok={result.get('ok')} | ok_tool={result.get('ok_tool', 'n/a')} | "
            f"provider={result.get('provider', provider)} | model={result.get('model', model)} | "
            f"latency={result.get('latency_ms', 'n/a')}ms | tools={','.join(result.get('tools_called') or [])}"
        )
        back_provider = _normalize_provider(str(result.get("provider") or provider))
        back_model = str(result.get("model") or model)
        back_href = "/?" + urlencode(
            {
                "provider": back_provider,
                "model": back_model,
                "ollama_base_url": ollama_base_url,
                "openrouter_base_url": openrouter_base_url,
                "question": question,
                "strict_tools": "1" if strict_tools else "0",
            }
        )
        artifact_paths = [str(p) for p in (result.get("artifact_paths") or []) if isinstance(p, str) and p.strip()]
        artifact_html_parts: list[str] = []
        for p in artifact_paths:
            href = "/artifact?" + urlencode({"path": p})
            artifact_html_parts.append(
                f'<a href="{html.escape(href, quote=True)}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#2f81f7;word-break:break-all;">{html.escape(p)}</a>'
            )
            if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
                artifact_html_parts.append(
                    f'<img src="{html.escape(href, quote=True)}" alt="{html.escape(p, quote=True)}" '
                    f'style="max-width:100%;height:auto;border:1px solid #30363d;border-radius:8px;background:#0b0f14;" />'
                )
        artifacts_block = "<br/>".join(artifact_html_parts)

        doc = f"""
<!doctype html>
<html lang="fr">
<head><meta charset="utf-8" /><title>GeoLifeCLEF MCP Chat (fallback)</title></head>
<body style="background:#0e1116;color:#e6edf3;font-family:ui-monospace,Menlo,monospace;padding:16px;">
  <p><a href="{html.escape(back_href, quote=True)}" style="color:#2f81f7">← Retour au chat</a></p>
  <h3>Réponse (fallback sans JS)</h3>
  <pre style="white-space:pre-wrap">{html.escape(answer)}</pre>
  <pre style="white-space:pre-wrap">{html.escape(details)}</pre>
  {"<hr/><h4>Artifacts</h4>" + artifacts_block if artifacts_block else ""}
  <hr />
  <pre>{html.escape(meta)}</pre>
</body>
</html>
"""
        return HTMLResponse(doc, headers=no_cache_headers)

    async def artifact_file(request: Request) -> JSONResponse | FileResponse:
        raw = str(request.query_params.get("path") or "").strip()
        if not raw:
            return JSONResponse({"ok": False, "error": "Missing path query parameter."}, status_code=400)

        target = Path(raw).expanduser()
        try:
            target = target.resolve()
        except Exception:
            return JSONResponse({"ok": False, "error": "Invalid path."}, status_code=400)

        if not target.exists() or not target.is_file():
            return JSONResponse({"ok": False, "error": "Artifact file not found."}, status_code=404)

        try:
            target.relative_to(repo_root)
        except ValueError:
            return JSONResponse({"ok": False, "error": "Path outside workspace is not allowed."}, status_code=403)

        if target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
            return JSONResponse({"ok": False, "error": "Only image artifacts are served."}, status_code=415)

        return FileResponse(str(target), headers=no_cache_headers)

    return Starlette(
        routes=[
            Route("/", homepage),
            Route("/ask", ask, methods=["POST"]),
            Route("/ask_sync", ask_sync, methods=["POST"]),
            Route("/artifact", artifact_file, methods=["GET"]),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Web UI locale pour l'agent MCP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--server-path",
        default=str(Path(__file__).resolve().parent / "server.py"),
    )
    parser.add_argument("--python", default=os.getenv("MCP_SERVER_PYTHON", sys.executable))
    args = parser.parse_args()

    app = create_app(Path(args.server_path).expanduser().resolve(), args.python)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
