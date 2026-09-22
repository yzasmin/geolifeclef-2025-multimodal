from __future__ import annotations

import os
from typing import Any, Optional

import requests

try:
    from . import tools_data, tools_stats
except ImportError:
    import tools_data  # type: ignore
    import tools_stats  # type: ignore

GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_MODEL_ENV = "GITHUB_MODEL"
GITHUB_ENDPOINT_ENV = "GITHUB_MODELS_ENDPOINT"
GITHUB_TIMEOUT_ENV = "GITHUB_API_TIMEOUT"
GITHUB_API_VERSION_ENV = "GITHUB_API_VERSION"
GITHUB_MODEL_DEFAULT = "openai/gpt-4.1"
GITHUB_ENDPOINT_DEFAULT = "https://models.github.ai/inference/chat/completions"

LLM_BACKEND_ENV = "LLM_BACKEND"  # ollama | openrouter | github_models | auto
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"
OLLAMA_TIMEOUT_ENV = "OLLAMA_API_TIMEOUT"
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
OLLAMA_MODEL_DEFAULT = "qwen3:8b"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "OPENROUTER_MODEL"
OPENROUTER_BASE_URL_ENV = "OPENROUTER_BASE_URL"
OPENROUTER_TIMEOUT_ENV = "OPENROUTER_API_TIMEOUT"
OPENROUTER_HTTP_REFERER_ENV = "OPENROUTER_HTTP_REFERER"
OPENROUTER_APP_TITLE_ENV = "OPENROUTER_APP_TITLE"
OPENROUTER_BASE_URL_DEFAULT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_DEFAULT = "google/gemma-4-31b-it:free"


def _error(error: str, hint: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": error, "hint": hint}
    payload.update(extra)
    return payload


def _language_name(language: str) -> str:
    return "French" if language.lower().startswith("fr") else "English"


def _pedagogical_style_requirements(language: str) -> str:
    """
    Returns a compact style contract so answers stay human, pedagogical,
    argument-based, and explicit about uncertainty.
    """
    if language.lower().startswith("fr"):
        return (
            "Rédige en langage humain et pédagogique. "
            "Évite le jargon non expliqué. "
            "Structure la réponse en 3 à 5 points avec, pour chaque point: argument -> preuve -> interprétation. "
            "Utilise des chiffres issus des tools quand disponibles. "
            "Termine par un niveau de confiance et une limite principale."
        )
    return (
        "Write in clear, pedagogical human language. "
        "Avoid unexplained jargon. "
        "Use 3 to 5 bullet points where each point follows: claim -> evidence -> interpretation. "
        "Use numeric evidence from tools when available. "
        "End with a confidence level and one main limitation."
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _fallback_explain_prediction_text(
    *,
    language: str,
    survey_summary: dict[str, Any],
    env: dict[str, Any],
    obs_quality_ctx: dict[str, Any],
) -> str:
    location = env.get("location", {}) if isinstance(env.get("location"), dict) else {}
    country = location.get("country") or "n/a"
    region = location.get("region") or "n/a"
    top_species = survey_summary.get("top_species", [])
    top_species_line = ", ".join(
        f"{item.get('species_id')} ({item.get('count')})"
        for item in top_species[:3]
        if isinstance(item, dict)
    ) or "n/a"

    quality = obs_quality_ctx.get("observation_quality", {}) if isinstance(obs_quality_ctx, dict) else {}
    conf_level = str(quality.get("confidence_level") or "medium")
    conf_score = _safe_float(quality.get("quality_confidence"))
    quality_summary = str(quality.get("quality_explanation_summary") or "").strip()
    ts_quality = quality.get("time_series_quality", {}) if isinstance(quality.get("time_series_quality"), dict) else {}
    tab_quality = quality.get("tabular_quality", {}) if isinstance(quality.get("tabular_quality"), dict) else {}
    ts_missing = _safe_float(ts_quality.get("overall_ts_missingness"))
    tab_missing = _safe_float(tab_quality.get("tabular_missing_ratio"))

    conf_score_txt = f"{conf_score:.2f}" if conf_score is not None else "n/a"
    ts_missing_txt = f"{ts_missing:.2f}" if ts_missing is not None else "n/a"
    tab_missing_txt = f"{tab_missing:.2f}" if tab_missing is not None else "n/a"

    if language.lower().startswith("fr"):
        return (
            "Je n’ai pas pu utiliser le backend LLM distant, mais je fournis une explication pédagogique basée sur les tools.\n"
            f"- Argument 1: Le site est situé en {country} ({region}). Preuve: survey={survey_summary.get('survey_id')}. "
            "Interprétation: le contexte biogéographique contraint fortement les espèces possibles.\n"
            f"- Argument 2: Les espèces dominantes observées localement sont {top_species_line}. "
            "Preuve: comptages issus du dataset sur ce survey. Interprétation: ces espèces décrivent la niche locale.\n"
            f"- Argument 3: Qualité des entrées. Preuve: quality_confidence={conf_score_txt}, "
            f"missingness_TS={ts_missing_txt}, missingness_tabulaire={tab_missing_txt}. "
            "Interprétation: la fiabilité de l’explication dépend directement de ces indicateurs.\n"
            f"Niveau de confiance: {conf_level}. Limite principale: {quality_summary or 'LLM indisponible, synthèse basée uniquement sur les tools structurés.'}"
        )
    return (
        "I could not use the remote LLM backend, so I provide a pedagogical explanation from MCP tools only.\n"
        f"- Claim 1: The site is in {country} ({region}). Evidence: survey={survey_summary.get('survey_id')}. "
        "Interpretation: biogeographic context strongly constrains plausible species.\n"
        f"- Claim 2: Dominant local species are {top_species_line}. "
        "Evidence: per-survey counts from dataset tools. Interpretation: these species characterize local niche conditions.\n"
        f"- Claim 3: Input quality affects reliability. Evidence: quality_confidence={conf_score_txt}, "
        f"TS_missingness={ts_missing_txt}, tabular_missingness={tab_missing_txt}. "
        "Interpretation: uncertainty rises when key inputs are degraded.\n"
        f"Confidence level: {conf_level}. Main limitation: {quality_summary or 'LLM unavailable; answer is based only on structured tool outputs.'}"
    )


def _fallback_describe_survey_text(
    *,
    language: str,
    survey_summary: dict[str, Any],
    env: dict[str, Any],
    coocc: list[dict[str, Any]],
    obs_quality_ctx: dict[str, Any],
) -> str:
    location = env.get("location", {}) if isinstance(env.get("location"), dict) else {}
    country = location.get("country") or "n/a"
    region = location.get("region") or "n/a"
    top_species = survey_summary.get("top_species", [])
    top_species_line = ", ".join(
        f"{item.get('species_id')} ({item.get('count')})"
        for item in top_species[:3]
        if isinstance(item, dict)
    ) or "n/a"
    coocc_line = ", ".join(
        f"{item.get('species_id')} ({item.get('shared_surveys')})"
        for item in coocc[:3]
        if isinstance(item, dict)
    ) or "n/a"
    quality = obs_quality_ctx.get("observation_quality", {}) if isinstance(obs_quality_ctx, dict) else {}
    conf_level = str(quality.get("confidence_level") or "medium")
    conf_score = _safe_float(quality.get("quality_confidence"))
    conf_score_txt = f"{conf_score:.2f}" if conf_score is not None else "n/a"
    quality_summary = str(quality.get("quality_explanation_summary") or "").strip()

    if language.lower().startswith("fr"):
        return (
            f"Portrait du site {survey_summary.get('survey_id')} ({country}, {region}).\n"
            f"- Espèces dominantes: {top_species_line}.\n"
            f"- Cooccurrences typiques: {coocc_line}.\n"
            f"- Qualité des données: score={conf_score_txt}, niveau={conf_level}.\n"
            f"Lecture pédagogique: ce site combine un signal écologique local cohérent avec une fiabilité {conf_level}. "
            f"Limite principale: {quality_summary or 'absence de génération LLM détaillée, synthèse issue des tools.'}"
        )
    return (
        f"Survey profile {survey_summary.get('survey_id')} ({country}, {region}).\n"
        f"- Dominant species: {top_species_line}.\n"
        f"- Typical co-occurrences: {coocc_line}.\n"
        f"- Data quality: score={conf_score_txt}, level={conf_level}.\n"
        f"Pedagogical interpretation: local ecological signal is coherent with {conf_level} reliability. "
        f"Main limitation: {quality_summary or 'no detailed LLM generation, summary built from tools only.'}"
    )


def _fallback_explain_term_text(term: str, language: str) -> str:
    if language.lower().startswith("fr"):
        return (
            f"Définition (fallback) de '{term}': c’est un concept utilisé pour décrire ou expliquer la distribution des espèces.\n"
            "Pourquoi c’est utile: il aide à relier des variables mesurées (climat, sol, saisonnalité) à la présence probable d’espèces.\n"
            "Exemple concret: on compare sa valeur entre sites pour voir si elle augmente ou réduit la probabilité d’occurrence."
        )
    return (
        f"Fallback definition of '{term}': it is a concept used to describe or explain species distribution patterns.\n"
        "Why it matters: it links measured variables (climate, soil, seasonality) to species occurrence likelihood.\n"
        "Concrete example: compare its value across sites to see whether it increases or decreases predicted occurrence."
    )


def _required_env(name: str) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None, _error(
            f"Missing {name}",
            f"Set {name} in your .env before calling tools_llm.",
        )
    return value.strip(), None


def _parse_timeout(env_name: str, default_value: str) -> tuple[Optional[float], Optional[dict[str, Any]]]:
    timeout_raw = os.getenv(env_name, default_value)
    try:
        return float(timeout_raw), None
    except ValueError:
        return None, _error(
            f"Invalid {env_name}",
            f"Set {env_name} to a numeric value in seconds.",
            current_value=timeout_raw,
        )


def _extract_text_from_response(payload: dict[str, Any]) -> Optional[str]:
    # Format Ollama: {"message": {"content": "..."}}
    message_obj = payload.get("message")
    if isinstance(message_obj, dict):
        content = message_obj.get("content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        text = content.strip()
        return text or None

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        text = "\n".join(part.strip() for part in parts if part and part.strip()).strip()
        return text or None

    return None


def _call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    chosen_model = (model or os.getenv(OLLAMA_MODEL_ENV, OLLAMA_MODEL_DEFAULT)).strip()
    base_url = os.getenv(OLLAMA_BASE_URL_ENV, OLLAMA_BASE_URL_DEFAULT).rstrip("/")
    endpoint = f"{base_url}/api/chat"

    timeout, timeout_err = _parse_timeout(OLLAMA_TIMEOUT_ENV, "45")
    if timeout_err:
        return timeout_err
    assert timeout is not None

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
    except requests.Timeout:
        return _error("Ollama timeout", f"Increase {OLLAMA_TIMEOUT_ENV} or retry later.")
    except requests.RequestException as exc:
        return _error(
            "Ollama request failed",
            "Local Ollama endpoint is unreachable.",
            details=str(exc),
            endpoint=endpoint,
        )

    if not response.ok:
        snippet = response.text[:400] if response.text else ""
        return _error(
            "Ollama API error",
            "Ollama returned a non-success status code.",
            status_code=response.status_code,
            response_snippet=snippet,
            model=chosen_model,
            endpoint=endpoint,
        )

    try:
        data = response.json()
    except ValueError:
        return _error("Invalid Ollama response", "Response was not valid JSON.", raw_text=response.text[:400])

    text = _extract_text_from_response(data)
    if text is None:
        return _error(
            "Empty Ollama answer",
            "No textual content was returned by the model.",
            model=chosen_model,
        )

    return {
        "ok": True,
        "provider": "ollama",
        "model": chosen_model,
        "text": text,
        "usage": {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "total_duration": data.get("total_duration"),
        },
    }


def _call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    api_key, key_err = _required_env(OPENROUTER_API_KEY_ENV)
    if key_err:
        return key_err
    assert api_key is not None

    if model is None:
        chosen_model = os.getenv(OPENROUTER_MODEL_ENV, OPENROUTER_MODEL_DEFAULT).strip()
    else:
        chosen_model = model.strip()

    endpoint = os.getenv(OPENROUTER_BASE_URL_ENV, OPENROUTER_BASE_URL_DEFAULT).strip()
    if not endpoint:
        endpoint = OPENROUTER_BASE_URL_DEFAULT

    timeout, timeout_err = _parse_timeout(OPENROUTER_TIMEOUT_ENV, "30")
    if timeout_err:
        return timeout_err
    assert timeout is not None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv(OPENROUTER_HTTP_REFERER_ENV, "http://localhost"),
        "X-Title": os.getenv(OPENROUTER_APP_TITLE_ENV, "GeoLifeCLEF-MCP"),
    }
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    except requests.Timeout:
        return _error("OpenRouter timeout", f"Increase {OPENROUTER_TIMEOUT_ENV} or retry later.")
    except requests.RequestException as exc:
        return _error(
            "OpenRouter request failed",
            "Network or endpoint error while calling OpenRouter API.",
            details=str(exc),
            endpoint=endpoint,
        )

    if not response.ok:
        snippet = response.text[:400] if response.text else ""
        return _error(
            "OpenRouter API error",
            "OpenRouter returned a non-success status code.",
            status_code=response.status_code,
            response_snippet=snippet,
            model=chosen_model,
            endpoint=endpoint,
        )

    try:
        data = response.json()
    except ValueError:
        return _error("Invalid OpenRouter response", "Response was not valid JSON.", raw_text=response.text[:400])

    text = _extract_text_from_response(data)
    if text is None:
        return _error(
            "Empty OpenRouter answer",
            "No textual content was returned by the model.",
            model=chosen_model,
        )

    return {
        "ok": True,
        "provider": "openrouter",
        "model": chosen_model,
        "text": text,
        "usage": data.get("usage"),
    }


def _call_github_models(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    token, token_err = _required_env(GITHUB_TOKEN_ENV)
    if token_err:
        return token_err
    assert token is not None

    if model is None:
        chosen_model = os.getenv(GITHUB_MODEL_ENV, GITHUB_MODEL_DEFAULT).strip()
    else:
        chosen_model = model.strip()

    endpoint = os.getenv(GITHUB_ENDPOINT_ENV, GITHUB_ENDPOINT_DEFAULT).strip()
    if not endpoint:
        endpoint = GITHUB_ENDPOINT_DEFAULT

    timeout, timeout_err = _parse_timeout(GITHUB_TIMEOUT_ENV, "30")
    if timeout_err:
        return timeout_err
    assert timeout is not None

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": os.getenv(GITHUB_API_VERSION_ENV, "2026-03-10"),
        "Content-Type": "application/json",
    }
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    except requests.Timeout:
        return _error("LLM request timeout", f"Increase {GITHUB_TIMEOUT_ENV} or retry later.")
    except requests.RequestException as exc:
        return _error(
            "LLM request failed",
            "Network or endpoint error while calling GitHub Models API.",
            details=str(exc),
        )

    if not response.ok:
        snippet = response.text[:400] if response.text else ""
        return _error(
            "LLM API error",
            "GitHub Models API returned a non-success status code.",
            status_code=response.status_code,
            response_snippet=snippet,
            model=chosen_model,
        )

    try:
        data = response.json()
    except ValueError:
        return _error("Invalid LLM response", "Response was not valid JSON.", raw_text=response.text[:400])

    text = _extract_text_from_response(data)
    if text is None:
        return _error(
            "Empty LLM answer",
            "No textual content was returned by the model.",
            model=chosen_model,
        )

    return {
        "ok": True,
        "provider": "github_models",
        "model": chosen_model,
        "text": text,
        "usage": data.get("usage"),
    }


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> dict[str, Any]:
    if not system_prompt or not system_prompt.strip():
        return _error("Empty system_prompt", "Provide a non-empty system prompt.")
    if not user_prompt or not user_prompt.strip():
        return _error("Empty user_prompt", "Provide a non-empty user prompt.")

    backend = os.getenv(LLM_BACKEND_ENV, "ollama").strip().lower()
    if backend not in {"ollama", "openrouter", "github_models", "auto"}:
        return _error(
            f"Invalid {LLM_BACKEND_ENV}",
            "Use one of: ollama, openrouter, github_models, auto.",
            current_value=backend,
        )

    attempts: list[str] = []
    errors: list[dict[str, Any]] = []

    def _attempt(provider: str) -> Optional[dict[str, Any]]:
        if provider == "ollama":
            resp = _call_ollama(system_prompt, user_prompt, model, temperature, max_tokens)
        elif provider == "openrouter":
            resp = _call_openrouter(system_prompt, user_prompt, model, temperature, max_tokens)
        else:
            resp = _call_github_models(system_prompt, user_prompt, model, temperature, max_tokens)
        attempts.append(provider)
        if resp.get("ok"):
            return resp
        errors.append({"provider": provider, "error": resp})
        return None

    if backend in {"openrouter", "auto"}:
        ok_resp = _attempt("openrouter")
        if ok_resp is not None:
            return ok_resp

    if backend in {"ollama", "auto"}:
        ok_resp = _attempt("ollama")
        if ok_resp is not None:
            return ok_resp

    if backend in {"github_models", "ollama", "openrouter", "auto"}:
        ok_resp = _attempt("github_models")
        if ok_resp is not None:
            return ok_resp

    if len(errors) == 1 and isinstance(errors[0].get("error"), dict):
        single = dict(errors[0]["error"])
        single["attempted_backends"] = attempts
        return single

    # Si tout échoue, on renvoie un résumé clair des tentatives.
    return _error(
        "All LLM backends failed",
        "Configure OpenRouter/Ollama locally or set GitHub Models credentials as fallback.",
        attempted_backends=attempts,
        backend_errors=errors,
    )


def _get_survey_species_summary(survey_id: Any, top_k_species: int) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    df, err = tools_data._load_dataframe()
    if err:
        return None, err
    assert df is not None

    columns = tools_data._resolve_standard_columns(df)
    survey_col = columns["survey_id"]
    species_col = columns["species_id"]
    if survey_col is None or species_col is None:
        return None, tools_data._error(
            "Missing survey/species columns",
            "Expected surveyId and speciesId columns to build LLM context.",
            detected_columns=columns,
        )

    sid = tools_data._normalize_survey_id(survey_id)
    if sid is None:
        return None, tools_data._error("Invalid survey_id", "Provide a valid survey identifier.")

    rows = df[df[survey_col].astype(str).str.replace(r"\.0$", "", regex=True) == sid]
    if rows.empty:
        return None, tools_data._error("Survey not found", "No row matches this survey_id.", survey_id=sid)

    species_counts = rows[species_col].dropna().value_counts().head(max(1, int(top_k_species)))
    top_species = [
        {"species_id": tools_data._safe_value(sp), "count": int(count)}
        for sp, count in species_counts.items()
    ]

    return {"survey_id": sid, "top_species": top_species, "row_count": int(len(rows))}, None


def _get_observation_quality_context(survey_id: Any) -> dict[str, Any]:
    """
    Fetches explainability quality diagnostics when available.
    Never raises: returns a fallback context on failure.
    """
    try:
        sid = int(str(survey_id))
    except Exception:
        return {
            "available": False,
            "reason": "invalid_survey_id",
        }

    try:
        try:
            from . import tools_explain  # type: ignore
        except Exception:
            import tools_explain  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "reason": "tools_explain_import_failed",
            "details": str(exc),
        }

    try:
        quality_resp = tools_explain.get_attention_weights(sid)
    except Exception as exc:
        return {
            "available": False,
            "reason": "tools_explain_call_failed",
            "details": str(exc),
        }

    if not isinstance(quality_resp, dict) or not quality_resp.get("ok"):
        return {
            "available": False,
            "reason": "observation_quality_unavailable",
            "error": quality_resp.get("error") if isinstance(quality_resp, dict) else "invalid_response",
            "hint": quality_resp.get("hint") if isinstance(quality_resp, dict) else None,
        }

    quality = quality_resp.get("observation_quality")
    if not isinstance(quality, dict):
        return {
            "available": False,
            "reason": "observation_quality_missing",
        }

    return {
        "available": True,
        "observation_quality": quality,
        "temporal_importance": {
            "landsat_seasonal_importance": quality_resp.get("landsat_seasonal_importance", []),
            "bioclim_monthly_importance": quality_resp.get("bioclim_monthly_importance", []),
        },
    }


def explain_prediction(survey_id: Any, top_k_species: int = 5, language: str = "fr") -> dict[str, Any]:
    survey_summary, err = _get_survey_species_summary(survey_id, top_k_species=top_k_species)
    if err:
        return err
    assert survey_summary is not None

    env = tools_stats.get_env_features(survey_id)
    if not env.get("ok"):
        return env
    obs_quality_ctx = _get_observation_quality_context(survey_summary["survey_id"])

    lang_name = _language_name(language)
    style_req = _pedagogical_style_requirements(language)
    system_prompt = (
        f"You are an ecology assistant for species distribution modelling. "
        f"Answer in {lang_name} with concise, evidence-based explanations. "
        "Use cautious wording (probable/possible), never categorical claims without evidence. "
        f"{style_req}"
    )
    user_prompt = (
        "Use the site context below to explain ecological drivers of predicted species presence.\n"
        f"- Survey ID: {survey_summary['survey_id']}\n"
        f"- Top species in this survey: {survey_summary['top_species']}\n"
        f"- Location/context: {env.get('location', {})}\n"
        f"- Environmental features: {env.get('env_features', {})}\n\n"
        f"- Observation quality diagnostics: {obs_quality_ctx}\n\n"
        "Produce:\n"
        "1) Main ecological factors\n"
        "2) Data-quality causes that may affect prediction reliability (time series and tabular)\n"
        "3) Why species may co-occur here\n"
        "4) One uncertainty/limitation of the explanation with at least one numeric signal\n"
        "5) Keep a pedagogical tone for non-expert readers"
    )

    llm = _call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
    if not llm.get("ok"):
        return {
            "ok": True,
            "survey_id": survey_summary["survey_id"],
            "language": language,
            "context": {
                "top_species": survey_summary["top_species"],
                "location": env.get("location", {}),
                "env_features": env.get("env_features", {}),
                "observation_quality": obs_quality_ctx,
            },
            "prompt": {"system": system_prompt, "user": user_prompt},
            "explanation": _fallback_explain_prediction_text(
                language=language,
                survey_summary=survey_summary,
                env=env,
                obs_quality_ctx=obs_quality_ctx,
            ),
            "provider": "fallback_no_llm",
            "model": "deterministic_template",
            "fallback_used": True,
            "llm_error": llm,
        }

    return {
        "ok": True,
        "survey_id": survey_summary["survey_id"],
        "language": language,
        "context": {
            "top_species": survey_summary["top_species"],
            "location": env.get("location", {}),
            "env_features": env.get("env_features", {}),
            "observation_quality": obs_quality_ctx,
        },
        "prompt": {"system": system_prompt, "user": user_prompt},
        "explanation": llm.get("text"),
        "provider": llm.get("provider"),
        "model": llm.get("model"),
    }


def explain_term(term: str, language: str = "fr") -> dict[str, Any]:
    if not term or not str(term).strip():
        return _error("Empty term", "Provide a non-empty technical term to explain.")

    lang_name = _language_name(language)
    style_req = _pedagogical_style_requirements(language)
    system_prompt = (
        f"You are a pedagogy-oriented ecology tutor for species distribution modelling. "
        f"Answer in {lang_name}. "
        f"{style_req}"
    )
    user_prompt = (
        f"Explain the technical term '{term}' in the context of biodiversity prediction.\n"
        "Give:\n"
        "1) A simple definition\n"
        "2) Why it matters for prediction\n"
        "3) One concrete example\n"
        "4) One common misunderstanding and how to avoid it"
    )

    llm = _call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
    if not llm.get("ok"):
        return {
            "ok": True,
            "term": term,
            "language": language,
            "prompt": {"system": system_prompt, "user": user_prompt},
            "explanation": _fallback_explain_term_text(term=term, language=language),
            "provider": "fallback_no_llm",
            "model": "deterministic_template",
            "fallback_used": True,
            "llm_error": llm,
        }

    return {
        "ok": True,
        "term": term,
        "language": language,
        "prompt": {"system": system_prompt, "user": user_prompt},
        "explanation": llm.get("text"),
        "provider": llm.get("provider"),
        "model": llm.get("model"),
    }


def describe_survey(survey_id: Any, language: str = "fr") -> dict[str, Any]:
    survey_summary, err = _get_survey_species_summary(survey_id, top_k_species=8)
    if err:
        return err
    assert survey_summary is not None

    env = tools_stats.get_env_features(survey_id)
    if not env.get("ok"):
        return env
    obs_quality_ctx = _get_observation_quality_context(survey_summary["survey_id"])

    coocc = []
    top_species = survey_summary.get("top_species", [])
    if top_species:
        first_species = top_species[0].get("species_id")
        coocc_resp = tools_stats.get_cooccurrences(first_species, top_k=5)
        if coocc_resp.get("ok"):
            coocc = coocc_resp.get("cooccurrences", [])

    lang_name = _language_name(language)
    style_req = _pedagogical_style_requirements(language)
    system_prompt = (
        f"You are an ecological field-report assistant. "
        f"Write in {lang_name} with clear structure and scientific caution. "
        "State confidence explicitly and tie claims to numeric evidence when available. "
        f"{style_req}"
    )
    user_prompt = (
        "Generate a narrative profile of this survey site.\n"
        f"- Survey ID: {survey_summary['survey_id']}\n"
        f"- Location/context: {env.get('location', {})}\n"
        f"- Environmental features: {env.get('env_features', {})}\n"
        f"- Frequent species on this survey: {top_species}\n"
        f"- Typical co-occurrences (dataset-wide): {coocc}\n\n"
        f"- Observation quality diagnostics: {obs_quality_ctx}\n\n"
        "Output sections:\n"
        "1) Site portrait\n"
        "2) Ecological interpretation\n"
        "3) Data-quality caveats (time series + tabular) and confidence level\n"
        "4) Practical takeaway in plain language"
    )

    llm = _call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
    if not llm.get("ok"):
        return {
            "ok": True,
            "survey_id": survey_summary["survey_id"],
            "language": language,
            "context": {
                "location": env.get("location", {}),
                "env_features": env.get("env_features", {}),
                "top_species": top_species,
                "cooccurrences": coocc,
                "observation_quality": obs_quality_ctx,
            },
            "prompt": {"system": system_prompt, "user": user_prompt},
            "description": _fallback_describe_survey_text(
                language=language,
                survey_summary=survey_summary,
                env=env,
                coocc=coocc,
                obs_quality_ctx=obs_quality_ctx,
            ),
            "provider": "fallback_no_llm",
            "model": "deterministic_template",
            "fallback_used": True,
            "llm_error": llm,
        }

    return {
        "ok": True,
        "survey_id": survey_summary["survey_id"],
        "language": language,
        "context": {
            "location": env.get("location", {}),
            "env_features": env.get("env_features", {}),
            "top_species": top_species,
            "cooccurrences": coocc,
            "observation_quality": obs_quality_ctx,
        },
        "prompt": {"system": system_prompt, "user": user_prompt},
        "description": llm.get("text"),
        "provider": llm.get("provider"),
        "model": llm.get("model"),
    }
