from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing MCP Python SDK. Install it in the venv with: pip install mcp"
    ) from exc

import tools_data
import tools_llm
import tools_stats


mcp = FastMCP("GeoLifeCLEF MCP Server")


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(value)


@mcp.tool(name="filter_by_region")
def filter_by_region(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    limit: int = 1000,
) -> dict[str, Any]:
    """Return observations inside a GPS bounding box."""
    return tools_data.filter_by_region(
        min_lat=_as_float(min_lat),
        max_lat=_as_float(max_lat),
        min_lon=_as_float(min_lon),
        max_lon=_as_float(max_lon),
        limit=_as_int(limit),
    )


@mcp.tool(name="filter_by_region_name")
def filter_by_region_name(
    region_name: str, limit: int = 1000, case_sensitive: bool = False
) -> dict[str, Any]:
    """Return observations filtered by region or country name."""
    return tools_data.filter_by_region_name(
        region_name=region_name,
        limit=_as_int(limit),
        case_sensitive=bool(case_sensitive),
    )


@mcp.tool(name="get_random_images")
def get_random_images(
    n: int = 10, seed: int | None = None, region_name: str | None = None
) -> dict[str, Any]:
    """Return N random satellite TIFF image paths, optionally filtered by region."""
    return tools_data.get_random_images(
        n=_as_int(n), seed=None if seed is None else _as_int(seed), region_name=region_name
    )


@mcp.tool(name="select_model")
def select_model(model_name: str) -> dict[str, Any]:
    """Set the active ecological prediction model tag (e.g., v1, v6)."""
    return tools_data.select_model(model_name=model_name)


@mcp.tool(name="get_server_status")
def get_server_status() -> dict[str, Any]:
    """Return MCP data server status: CSV/TIFF availability and active model."""
    return tools_data.get_server_status()


@mcp.tool(name="get_species_stats")
def get_species_stats(top_k: int = 20, rare_threshold: int = 5) -> dict[str, Any]:
    """Compute species frequency ranking and rare/common summary statistics."""
    return tools_stats.get_species_stats(top_k=_as_int(top_k), rare_threshold=_as_int(rare_threshold))


@mcp.tool(name="get_env_features")
def get_env_features(survey_id: int | str) -> dict[str, Any]:
    """Return environmental feature values for one survey identifier."""
    return tools_stats.get_env_features(survey_id=survey_id)


@mcp.tool(name="filter_by_features")
def filter_by_features(
    min_altitude: float | None = None,
    max_altitude: float | None = None,
    min_temperature: float | None = None,
    max_temperature: float | None = None,
    country: str | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Filter observations by altitude, temperature, and country constraints."""
    return tools_stats.filter_by_features(
        min_altitude=None if min_altitude is None else _as_float(min_altitude),
        max_altitude=None if max_altitude is None else _as_float(max_altitude),
        min_temperature=None if min_temperature is None else _as_float(min_temperature),
        max_temperature=None if max_temperature is None else _as_float(max_temperature),
        country=country,
        limit=_as_int(limit),
    )


@mcp.tool(name="get_cooccurrences")
def get_cooccurrences(species_id: int | str, top_k: int = 10) -> dict[str, Any]:
    """Return top co-occurring species for a target species."""
    return tools_stats.get_cooccurrences(species_id=species_id, top_k=_as_int(top_k))


@mcp.tool(name="get_top_drivers")
def get_top_drivers(
    region: str | None = None,
    country: str | None = None,
    elevation_min: float = 0.0,
    elevation_max: float = 9000.0,
    bioclim_var: str | None = None,
    bioclim_min: float | None = None,
    bioclim_max: float | None = None,
    n_species_min: int | None = None,
    top_k_species: int = 10,
    top_k_vars: int = 8,
) -> dict[str, Any]:
    """Return top environmental drivers for a filtered region/country subset."""
    return tools_stats.get_top_drivers(
        region=region,
        country=country,
        elevation_min=_as_float(elevation_min),
        elevation_max=_as_float(elevation_max),
        bioclim_var=bioclim_var,
        bioclim_min=None if bioclim_min is None else _as_float(bioclim_min),
        bioclim_max=None if bioclim_max is None else _as_float(bioclim_max),
        n_species_min=None if n_species_min is None else _as_int(n_species_min),
        top_k_species=_as_int(top_k_species),
        top_k_vars=_as_int(top_k_vars),
    )


@mcp.tool(name="explain_prediction")
def explain_prediction(
    survey_id: int | str, top_k_species: int = 5, language: str = "fr"
) -> dict[str, Any]:
    """Explain why predicted species can occur at a given survey site."""
    return tools_llm.explain_prediction(
        survey_id=survey_id, top_k_species=_as_int(top_k_species), language=language
    )


@mcp.tool(name="explain_term")
def explain_term(term: str, language: str = "fr") -> dict[str, Any]:
    """Explain a technical ecology/SDM term using the LLM."""
    return tools_llm.explain_term(term=term, language=language)


@mcp.tool(name="describe_survey")
def describe_survey(survey_id: int | str, language: str = "fr") -> dict[str, Any]:
    """Generate a narrative ecological profile for a survey site."""
    return tools_llm.describe_survey(survey_id=survey_id, language=language)


@mcp.tool(name="ping")
def ping() -> dict[str, Any]:
    """Health check endpoint for MCP connectivity."""
    return {"ok": True, "message": "GeoLifeCLEF MCP server is running."}

import tools_explain

@mcp.tool(name="explain_model_prediction")
def explain_model_prediction_tool(survey_id: int, top_k: int = 5) -> dict:
    """Explique la prédiction du modèle : variables influentes, branches, saisons."""
    return tools_explain.explain_model_prediction(survey_id, top_k)

@mcp.tool(name="get_branch_importance")
def get_branch_importance_tool(survey_id: int) -> dict:
    """Quelle branche du modèle a le plus contribué à la prédiction."""
    return tools_explain.get_branch_importance(survey_id)

@mcp.tool(name="get_attention_weights")
def get_attention_weights_tool(survey_id: int) -> dict:
    """Quelle saison Landsat ou quel mois bioclim a le plus pesé."""
    return tools_explain.get_attention_weights(survey_id)

@mcp.tool(name="list_available_checkpoints")
def list_checkpoints_tool() -> dict:
    """Liste les modèles disponibles avec leurs scores F1."""
    return tools_explain.list_available_checkpoints()


if __name__ == "__main__":
    mcp.run(transport="stdio")
