from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are an ecological assistant for GeoLifeCLEF. "
    "Always use MCP tools when the question needs dataset, statistics, or model-explainability facts. "
    "For questions about the most important variables in a region, use get_top_drivers and report driver names/signs/scores from tool output. "
    "When data is uncertain, explicitly say what is missing. "
    "Respond in human, pedagogical language. "
    "Structure explanations with clear arguments and evidence: claim -> numeric proof from tools -> interpretation. "
    "Avoid generic textbook answers if tools provide concrete values. "
    "End with an explicit confidence level and one limitation."
)
DEFAULT_RESPONSE_STYLE_POLICY = (
    "Style policy (always apply): "
    "Write for a non-expert user in clear pedagogical language. "
    "Provide 3-5 concise arguments; for each argument: claim, numeric evidence from tools, interpretation. "
    "Avoid unexplained jargon and avoid generic statements not grounded in tool outputs. "
    "Finish with: confidence level (high/medium/low) and one main limitation."
)
DEFAULT_FAST_MODEL = "qwen3:8b"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_PROVIDER = "ollama"
_ARTIFACT_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")


class MissingDependencyError(RuntimeError):
    pass


def _load_langchain_dependencies(provider: str):
    try:
        from langchain.agents import create_agent  # type: ignore
        from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime guard
        raise MissingDependencyError(
            "Missing dependencies for local agent. Install with: "
            "pip install langchain langchain-mcp-adapters langchain-ollama langchain-openai langsmith"
        ) from exc

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime guard
            raise MissingDependencyError(
                "Missing ChatOllama dependency. Install with: pip install langchain-ollama"
            ) from exc
        return create_agent, MultiServerMCPClient, ChatOllama

    if provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime guard
            raise MissingDependencyError(
                "Missing ChatOpenAI dependency. Install with: pip install langchain-openai"
            ) from exc
        return create_agent, MultiServerMCPClient, ChatOpenAI

    raise MissingDependencyError(f"Unsupported provider: {provider}")


def _resolve_provider(provider: str) -> str:
    value = (provider or DEFAULT_PROVIDER).strip().lower()
    if value not in {"ollama", "openrouter"}:
        raise ValueError(f"Invalid provider '{provider}'. Use ollama or openrouter.")
    return value


def _default_model_for_provider(provider: str) -> str:
    if provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    return os.getenv("OLLAMA_MODEL", DEFAULT_FAST_MODEL)


def _default_server_path() -> Path:
    return Path(__file__).resolve().parent / "server.py"


def _build_mcp_env(provider: str, model: str) -> dict[str, str]:
    # Pass full environment so stdio MCP subprocess sees DATA_* and model vars.
    env = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Keep tools_llm aligned with agent provider/model by default.
    env.setdefault("LLM_BACKEND", provider)
    if provider == "openrouter":
        if model.strip():
            env.setdefault("OPENROUTER_MODEL", model.strip())
    elif provider == "ollama":
        if model.strip():
            env.setdefault("OLLAMA_MODEL", model.strip())
    return env


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return ""


def _extract_answer(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        text = _extract_text(content)
        if text:
            msg_type = (getattr(msg, "type", "") or "").lower()
            cls_name = msg.__class__.__name__.lower()
            if "ai" in cls_name or msg_type == "ai":
                return text

    # Fallback: premier message textuel disponible
    for msg in reversed(messages):
        text = _extract_text(getattr(msg, "content", None))
        if text:
            return text

    return ""


def _extract_tools_called(result: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for msg in result.get("messages", []):
        tool_calls = getattr(msg, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                name = None
                if isinstance(call, dict):
                    name = call.get("name")
                    if name is None and isinstance(call.get("function"), dict):
                        name = call["function"].get("name")
                if isinstance(name, str) and name and name not in seen:
                    seen.add(name)
                    names.append(name)

        msg_name = getattr(msg, "name", None)
        cls_name = msg.__class__.__name__.lower()
        if isinstance(msg_name, str) and "tool" in cls_name and msg_name not in seen:
            seen.add(msg_name)
            names.append(msg_name)

    return names


def _parse_tool_payload(content_text: str) -> Any:
    text = (content_text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: essaie d'extraire un bloc JSON principal.
    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right > left:
        snippet = text[left : right + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def _extract_tool_failures(result: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for msg in result.get("messages", []):
        cls_name = msg.__class__.__name__.lower()
        msg_type = (getattr(msg, "type", "") or "").lower()
        if "tool" not in cls_name and msg_type != "tool":
            continue

        tool_name = getattr(msg, "name", None) or "unknown_tool"
        content_text = _extract_text(getattr(msg, "content", None))
        if not content_text:
            continue

        payload = _parse_tool_payload(content_text)
        if isinstance(payload, dict):
            if payload.get("ok") is False:
                failures.append(
                    {
                        "tool": tool_name,
                        "error": payload.get("error"),
                        "hint": payload.get("hint"),
                        "raw": content_text[:400],
                    }
                )
            continue

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("ok") is False:
                    failures.append(
                        {
                            "tool": tool_name,
                            "error": item.get("error"),
                            "hint": item.get("hint"),
                            "raw": content_text[:400],
                        }
                    )
            continue

        if '"ok": false' in content_text.lower() or "'ok': false" in content_text.lower():
            failures.append({"tool": tool_name, "error": "Tool returned ok=false", "raw": content_text[:400]})

    return failures


def _apply_default_response_style(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return q
    return f"{q}\n\n{DEFAULT_RESPONSE_STYLE_POLICY}"


def _max_artifacts_to_return() -> int:
    raw = (os.getenv("AGENT_MAX_ARTIFACTS", "4") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 4
    return max(1, min(value, 12))


def _is_image_artifact_path(path: str) -> bool:
    value = (path or "").strip().lower()
    return bool(value) and value.endswith(_ARTIFACT_IMAGE_EXTS)


def _collect_artifact_candidates(
    value: Any,
    out: list[dict[str, str]],
    seen: set[str],
    *,
    tool_name: str,
) -> None:
    if isinstance(value, str):
        v = value.strip()
        if v and _is_image_artifact_path(v):
            if v not in seen:
                seen.add(v)
                out.append({"path": v, "tool": tool_name})
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_artifact_candidates(item, out, seen, tool_name=tool_name)
        return
    if isinstance(value, list):
        for item in value:
            _collect_artifact_candidates(item, out, seen, tool_name=tool_name)


def _extract_artifact_candidates(result: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for msg in result.get("messages", []):
        cls_name = msg.__class__.__name__.lower()
        msg_type = (getattr(msg, "type", "") or "").lower()
        if "tool" not in cls_name and msg_type != "tool":
            continue
        tool_name = str(getattr(msg, "name", "") or "unknown_tool")
        content_text = _extract_text(getattr(msg, "content", None))
        if not content_text:
            continue
        payload = _parse_tool_payload(content_text)
        if payload is None:
            continue
        _collect_artifact_candidates(payload, candidates, seen, tool_name=tool_name)
    return candidates


def _artifact_category(path: str) -> str:
    p = (path or "").lower()
    if "branch_importance" in p:
        return "branch"
    if "landsat_temporal_importance" in p:
        return "temporal_landsat"
    if "bioclim_monthly_importance" in p:
        return "temporal_bioclim"
    if "missingness_timeline_landsat" in p:
        return "quality_ts_landsat"
    if "missingness_timeline_bioclim" in p:
        return "quality_ts_bioclim"
    if "tabular_quality_overview" in p:
        return "quality_tabular"
    if "species_" in p and "env_attribution" in p:
        return "species_env"
    if "attention" in p:
        return "attention"
    if "gradcam" in p:
        return "gradcam"
    if "shap" in p:
        return "shap"
    return "other"


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def _artifact_relevance_score(
    *,
    path: str,
    tool_name: str,
    question_l: str,
    tools_called_set: set[str],
) -> int:
    p = path.lower()
    score = 0

    # Baseline by artifact type.
    category = _artifact_category(path)
    base_by_cat = {
        "branch": 60,
        "temporal_landsat": 55,
        "temporal_bioclim": 55,
        "quality_ts_landsat": 55,
        "quality_ts_bioclim": 55,
        "quality_tabular": 55,
        "species_env": 50,
        "attention": 40,
        "gradcam": 30,
        "shap": 25,
        "other": 10,
    }
    score += base_by_cat.get(category, 10)

    # Tool intent alignment.
    if tool_name == "get_branch_importance":
        score += 35 if category == "branch" else 8
    elif tool_name == "get_attention_weights":
        score += 35 if category in {"temporal_landsat", "temporal_bioclim", "quality_ts_landsat", "quality_ts_bioclim"} else 8
    elif tool_name == "explain_model_prediction":
        score += 20

    # Question intent alignment.
    branch_q = _contains_any(question_l, ("branche", "branch", "contribution"))
    temporal_q = _contains_any(question_l, ("saison", "mois", "temporel", "temporal", "attention", "landsat", "bioclim"))
    quality_q = _contains_any(question_l, ("qualité", "quality", "incertitude", "uncertainty", "missing", "manquant", "imput", "fiabilité"))
    species_q = _contains_any(question_l, ("espèce", "species", "prédiction", "prediction", "top-k", "top k", "influencé"))
    image_q = _contains_any(question_l, ("nuage", "cloud", "image", "tiff", "haze", "gradcam", "shap"))

    if branch_q and category == "branch":
        score += 140
    if temporal_q and category in {"temporal_landsat", "temporal_bioclim", "attention"}:
        score += 120
    if quality_q and category in {"quality_ts_landsat", "quality_ts_bioclim", "quality_tabular"}:
        score += 130
    if species_q and category == "species_env":
        score += 115
    if image_q and category in {"attention", "gradcam", "shap"}:
        score += 70

    # Slight preference for standard explainability plots over advanced optional plots.
    if "malala_" in p:
        score -= 10

    # If explainability tools were called, prioritize explainability artifacts over others.
    if {"get_branch_importance", "get_attention_weights", "explain_model_prediction"} & tools_called_set:
        if category in {"branch", "temporal_landsat", "temporal_bioclim", "quality_ts_landsat", "quality_ts_bioclim", "quality_tabular", "species_env"}:
            score += 15

    return score


def _select_relevant_artifact_paths(
    *,
    question: str,
    tools_called: list[str],
    candidates: list[dict[str, str]],
) -> list[str]:
    if not candidates:
        return []

    max_items = _max_artifacts_to_return()
    question_l = (question or "").strip().lower()
    tools_called_set = {t for t in tools_called if isinstance(t, str)}

    scored: list[dict[str, Any]] = []
    for c in candidates:
        path = str(c.get("path") or "").strip()
        if not path:
            continue
        tool_name = str(c.get("tool") or "unknown_tool")
        score = _artifact_relevance_score(
            path=path,
            tool_name=tool_name,
            question_l=question_l,
            tools_called_set=tools_called_set,
        )
        scored.append(
            {
                "path": path,
                "tool": tool_name,
                "category": _artifact_category(path),
                "score": score,
            }
        )

    if not scored:
        return []

    scored.sort(key=lambda x: (-int(x["score"]), str(x["path"])))

    # Pass 1: keep diversity (one best artifact per category).
    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    used_categories: set[str] = set()
    for item in scored:
        if item["path"] in selected_paths:
            continue
        category = str(item["category"])
        if category in used_categories:
            continue
        selected.append(item)
        selected_paths.add(item["path"])
        used_categories.add(category)
        if len(selected) >= max_items:
            break

    # Pass 2: fill remaining slots with best scored leftovers.
    if len(selected) < max_items:
        for item in scored:
            if item["path"] in selected_paths:
                continue
            selected.append(item)
            selected_paths.add(item["path"])
            if len(selected) >= max_items:
                break

    return [str(item["path"]) for item in selected[:max_items]]


def _build_chat_model(
    provider: str,
    model: str,
    llm_cls: Any,
    *,
    ollama_base_url: str,
    openrouter_base_url: str,
) -> Any:
    temperature = float(os.getenv("AGENT_TEMPERATURE", "0.1"))

    if provider == "ollama":
        return llm_cls(
            model=model,
            base_url=ollama_base_url,
            temperature=temperature,
        )

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise MissingDependencyError("Missing OPENROUTER_API_KEY for provider=openrouter.")

    timeout_raw = os.getenv("OPENROUTER_API_TIMEOUT", "45").strip()
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise MissingDependencyError("OPENROUTER_API_TIMEOUT must be numeric.") from exc

    headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "GeoLifeCLEF-MCP"),
    }
    normalized_base_url = openrouter_base_url.rstrip("/")
    if normalized_base_url.endswith("/chat/completions"):
        normalized_base_url = normalized_base_url[: -len("/chat/completions")]

    return llm_cls(
        model=model,
        api_key=api_key,
        base_url=normalized_base_url,
        temperature=temperature,
        timeout=timeout,
        default_headers=headers,
    )


async def ask_question(
    question: str,
    *,
    provider: str,
    model: str,
    ollama_base_url: str,
    openrouter_base_url: str,
    server_path: Path,
    python_cmd: str,
) -> dict[str, Any]:
    try:
        provider = _resolve_provider(provider)
    except ValueError as exc:
        return {
            "ok": False,
            "error": "Invalid provider",
            "hint": str(exc),
        }

    try:
        create_agent, MultiServerMCPClient, llm_cls = _load_langchain_dependencies(provider)
    except MissingDependencyError as exc:
        return {
            "ok": False,
            "error": "Missing dependencies",
            "hint": str(exc),
        }

    if not question.strip():
        return {
            "ok": False,
            "error": "Empty question",
            "hint": "Provide a non-empty question.",
        }

    if not server_path.exists():
        return {
            "ok": False,
            "error": "MCP server not found",
            "hint": f"Expected server.py at {server_path}",
        }

    client = MultiServerMCPClient(
        {
            "geolifeclef": {
                "transport": "stdio",
                "command": python_cmd,
                "args": [str(server_path)],
                "cwd": str(server_path.parent),
                "env": _build_mcp_env(provider=provider, model=model),
            }
        }
    )

    start = time.perf_counter()
    try:
        tools = await client.get_tools()
        try:
            llm = _build_chat_model(
                provider=provider,
                model=model,
                llm_cls=llm_cls,
                ollama_base_url=ollama_base_url,
                openrouter_base_url=openrouter_base_url,
            )
        except MissingDependencyError as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            return {
                "ok": False,
                "error": "LLM configuration error",
                "hint": str(exc),
                "latency_ms": elapsed_ms,
            }
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

        trace_tags = ["geolifeclef", "mcp", provider, "langchain"]
        config = {
            "tags": trace_tags,
            "metadata": {
                "question": question,
                "mcp_server": str(server_path),
                "llm_provider": provider,
                "llm_model": model,
                "ollama_base_url": ollama_base_url,
                "openrouter_base_url": openrouter_base_url,
            },
        }
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": _apply_default_response_style(question)}]},
            config=config,
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "ok": False,
            "error": "Agent invocation failed",
            "hint": "Check MCP server startup, LLM provider availability, and LangChain dependencies.",
            "details": str(exc),
            "latency_ms": elapsed_ms,
        }

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    answer = _extract_answer(result)
    tools_called = _extract_tools_called(result)
    tool_failures = _extract_tool_failures(result)
    artifact_candidates = _extract_artifact_candidates(result)
    artifact_paths = _select_relevant_artifact_paths(
        question=question,
        tools_called=tools_called,
        candidates=artifact_candidates,
    )
    ok_tool = len(tool_failures) == 0

    return {
        "ok": True,
        "question": question,
        "answer": answer,
        "tools_called": tools_called,
        "ok_tool": ok_tool,
        "tool_failures": tool_failures,
        "artifact_paths": artifact_paths,
        "artifact_count_total": len(artifact_candidates),
        "artifact_count_selected": len(artifact_paths),
        "latency_ms": elapsed_ms,
        "provider": provider,
        "model": model,
        "trace_tags": ["geolifeclef", "mcp", provider, "langchain"],
        "langsmith_tracing": os.getenv("LANGSMITH_TRACING", "false"),
    }


async def _run_once(args: argparse.Namespace) -> int:
    response = await ask_question(
        args.question,
        provider=args.provider,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        openrouter_base_url=args.openrouter_base_url,
        server_path=Path(args.server_path).expanduser().resolve(),
        python_cmd=args.python,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if response.get("ok") else 1


async def _run_interactive(args: argparse.Namespace) -> int:
    print("Interactive mode. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if question.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            return 0

        response = await ask_question(
            question,
            provider=args.provider,
            model=args.model,
            ollama_base_url=args.ollama_base_url,
            openrouter_base_url=args.openrouter_base_url,
            server_path=Path(args.server_path).expanduser().resolve(),
            python_cmd=args.python,
        )
        print(json.dumps(response, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local LangChain agent using MCP server tools + Ollama/OpenRouter.",
    )
    parser.add_argument("--question", type=str, default="")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive REPL mode.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("AGENT_LLM_PROVIDER", DEFAULT_PROVIDER),
        choices=["ollama", "openrouter"],
        help="LLM provider used by the agent.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("AGENT_MODEL", "").strip(),
        help="Model name for selected provider.",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--openrouter-base-url",
        type=str,
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter base URL.",
    )
    parser.add_argument(
        "--server-path",
        type=str,
        default=str(_default_server_path()),
        help="Path to MCP server.py.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=os.getenv("MCP_SERVER_PYTHON", sys.executable),
        help="Python executable used to spawn MCP server subprocess.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        args.provider = _resolve_provider(args.provider)
    except ValueError as exc:
        parser.error(str(exc))
    if not str(args.model or "").strip():
        args.model = _default_model_for_provider(args.provider)

    if args.interactive:
        return asyncio.run(_run_interactive(args))

    if not args.question.strip():
        parser.error("Provide --question or use --interactive.")

    return asyncio.run(_run_once(args))


if __name__ == "__main__":
    raise SystemExit(main())
