from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

from langchain_mcp_agent import ask_question

DEFAULT_QUESTIONS = [
    "Donne-moi les sites en France.",
    "Quels sont les 10 espèces les plus fréquentes ?",
    "Quelles sont les espèces les plus rares ?",
    "Filtre les observations entre 43 et 46 de latitude.",
    "Décris le survey 212.",
    "Pourquoi le modèle prédit cette plante sur le survey 212 ?",
    "Quelle branche du modèle contribue le plus pour 212 ?",
    "Quelles saisons Landsat comptent le plus pour 212 ?",
    "Explique le terme BioClim en français.",
    "Quel est le statut du serveur de données ?",
]


async def run_smoke(
    model: str,
    provider: str,
    ollama_base_url: str,
    openrouter_base_url: str,
    server_path: Path,
    python_cmd: str,
) -> dict:
    runs = []
    for question in DEFAULT_QUESTIONS:
        result = await ask_question(
            question,
            provider=provider,
            model=model,
            ollama_base_url=ollama_base_url,
            openrouter_base_url=openrouter_base_url,
            server_path=server_path,
            python_cmd=python_cmd,
        )
        runs.append(result)

    latencies = [r["latency_ms"] for r in runs if r.get("ok") and isinstance(r.get("latency_ms"), (int, float))]
    median_latency = round(float(statistics.median(latencies)), 1) if latencies else None

    return {
        "ok": all(r.get("ok") for r in runs),
        "total_questions": len(DEFAULT_QUESTIONS),
        "successful_runs": sum(1 for r in runs if r.get("ok")),
        "median_latency_ms": median_latency,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke + latency benchmark for MCP LangChain agent.")
    parser.add_argument("--provider", default=os.getenv("AGENT_LLM_PROVIDER", "ollama"), choices=["ollama", "openrouter"])
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_MODEL", "").strip(),
    )
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--openrouter-base-url", default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument(
        "--server-path",
        default=str(Path(__file__).resolve().parent / "server.py"),
    )
    parser.add_argument("--python", default=os.getenv("MCP_SERVER_PYTHON", sys.executable))
    args = parser.parse_args()

    result = asyncio.run(
        run_smoke(
            model=args.model
            or (
                os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
                if args.provider == "openrouter"
                else os.getenv("OLLAMA_MODEL", "qwen3:8b")
            ),
            provider=args.provider,
            ollama_base_url=args.ollama_base_url,
            openrouter_base_url=args.openrouter_base_url,
            server_path=Path(args.server_path).expanduser().resolve(),
            python_cmd=args.python,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
