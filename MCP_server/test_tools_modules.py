from __future__ import annotations

import importlib
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import requests
import torch


def _build_tiff_path(root: Path, survey_id: int) -> Path:
    sid = str(survey_id)
    if len(sid) >= 4:
        return root / sid[-2:] / sid[-4:-2] / f"{sid}.tiff"
    return root / sid / f"{sid}.tiff"


class ToolsModulesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        server_dir = Path(__file__).resolve().parent
        if str(server_dir) not in sys.path:
            sys.path.insert(0, str(server_dir))

        self.tmp = tempfile.TemporaryDirectory()
        tmp_root = Path(self.tmp.name)
        self.csv_path = tmp_root / "observations.csv"
        self.tiff_root = tmp_root / "tiffs"
        self.tiff_root.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "surveyId": 1,
                "spId": 101,
                "lat": 43.5,
                "lon": 3.2,
                "region": "sud de la france",
                "country": "France",
                "Bio1": 12.0,
                "Bio12": 550.0,
                "Elevation": 120.0,
                "temperature": 15.0,
            },
            {
                "surveyId": 1,
                "spId": 102,
                "lat": 43.5,
                "lon": 3.2,
                "region": "sud de la france",
                "country": "France",
                "Bio1": 12.0,
                "Bio12": 550.0,
                "Elevation": 120.0,
                "temperature": 15.0,
            },
            {
                "surveyId": 2,
                "spId": 101,
                "lat": 48.8,
                "lon": 2.3,
                "region": "ile de france",
                "country": "France",
                "Bio1": 10.0,
                "Bio12": 700.0,
                "Elevation": 80.0,
                "temperature": 11.0,
            },
            {
                "surveyId": 3,
                "spId": 103,
                "lat": 41.9,
                "lon": 12.5,
                "region": "lazio",
                "country": "Italy",
                "Bio1": 14.0,
                "Bio12": 800.0,
                "Elevation": 60.0,
                "temperature": 17.0,
            },
            {
                "surveyId": 3018575,
                "spId": 104,
                "lat": 43.7,
                "lon": 5.1,
                "region": "provence",
                "country": "France",
                "Bio1": 13.5,
                "Bio12": 500.0,
                "Elevation": 230.0,
                "temperature": 16.0,
            },
        ]
        pd.DataFrame(rows).to_csv(self.csv_path, index=False)

        for sid in (1, 2, 3018575):
            path = _build_tiff_path(self.tiff_root, sid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"dummy")

        os.environ["DATA_CSV_PATH"] = str(self.csv_path)
        os.environ["TIFF_ROOT"] = str(self.tiff_root)
        os.environ.pop("GITHUB_TOKEN", None)

        import tools_data
        import tools_explain
        import langchain_mcp_agent
        import tools_llm
        import tools_stats

        self.tools_data = importlib.reload(tools_data)
        self.tools_explain = importlib.reload(tools_explain)
        self.agent = importlib.reload(langchain_mcp_agent)
        self.tools_stats = importlib.reload(tools_stats)
        self.tools_llm = importlib.reload(tools_llm)
        self.tools_data._reset_cache_for_tests()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_data_filters_and_server_status(self) -> None:
        region = self.tools_data.filter_by_region(43.0, 44.0, 3.0, 6.0, limit=10)
        self.assertTrue(region["ok"])
        self.assertGreaterEqual(region["matched_count"], 3)

        region_name = self.tools_data.filter_by_region_name("sud de la france")
        self.assertTrue(region_name["ok"])
        self.assertGreaterEqual(region_name["matched_count"], 2)

        status = self.tools_data.get_server_status()
        self.assertTrue(status["ok"])
        self.assertTrue(status["csv_available"])
        self.assertTrue(status["tiff_root_available"])
        self.assertEqual(status["row_count"], 5)

    def test_random_images_is_seeded(self) -> None:
        first = self.tools_data.get_random_images(n=2, seed=42)
        second = self.tools_data.get_random_images(n=2, seed=42)
        self.assertTrue(first["ok"])
        self.assertEqual(first["images"], second["images"])
        self.assertEqual(first["returned"], 2)

    def test_stats_and_cooccurrences(self) -> None:
        species = self.tools_stats.get_species_stats(top_k=3, rare_threshold=1)
        self.assertTrue(species["ok"])
        self.assertGreaterEqual(species["total_species"], 3)

        env = self.tools_stats.get_env_features(1)
        self.assertTrue(env["ok"])
        self.assertIn("env_features", env)
        self.assertIn("Bio1", env["env_features"])

        filtered = self.tools_stats.filter_by_features(min_altitude=100, country="France")
        self.assertTrue(filtered["ok"])
        self.assertGreaterEqual(filtered["matched_count"], 2)

        co = self.tools_stats.get_cooccurrences(101, top_k=5)
        self.assertTrue(co["ok"])
        self.assertGreaterEqual(co["target_survey_count"], 1)

    def test_get_top_drivers_bridge(self) -> None:
        fake_filter_output = {
            "df_combined": pd.DataFrame(
                [
                    {"surveyId": 1, "speciesId": 11, "bio_17": 120.0, "bio_6": 3.0, "elev": 80.0},
                    {"surveyId": 1, "speciesId": 12, "bio_17": 120.0, "bio_6": 3.0, "elev": 80.0},
                    {"surveyId": 2, "speciesId": 11, "bio_17": 90.0, "bio_6": 1.0, "elev": 150.0},
                    {"surveyId": 3, "speciesId": 13, "bio_17": 60.0, "bio_6": -1.0, "elev": 220.0},
                ]
            ),
            "filter_context": {"region": "MEDITERRANEAN"},
            "file_registry": {1: None, 2: None, 3: None},
            "execution_status": {"count": 4, "is_empty": False},
        }

        with mock.patch.object(self.tools_stats.tools_data, "apply_filter", return_value=fake_filter_output):
            result = self.tools_stats.get_top_drivers(region="MEDITERRANEAN", top_k_vars=3)

        self.assertTrue(result["ok"])
        self.assertEqual(result["filter_context"]["region"], "MEDITERRANEAN")
        self.assertEqual(result["analysis_source"], "tools_stats.top_drivers")
        self.assertIn("drivers", result)

    def test_get_top_drivers_region_alias_resolution(self) -> None:
        fake_filter_output = {
            "df_combined": pd.DataFrame(
                [
                    {"surveyId": 1, "speciesId": 11, "bio_17": 120.0, "bio_6": 3.0, "elev": 80.0},
                    {"surveyId": 2, "speciesId": 12, "bio_17": 90.0, "bio_6": 1.0, "elev": 150.0},
                ]
            ),
            "filter_context": {"region": "MEDITERRANEAN"},
            "file_registry": {1: None, 2: None},
            "execution_status": {"count": 2, "is_empty": False},
        }

        with mock.patch.object(self.tools_stats.tools_data, "apply_filter", return_value=fake_filter_output) as mocked:
            result = self.tools_stats.get_top_drivers(region="Méditerranée", top_k_vars=2)

        self.assertTrue(result["ok"])
        self.assertEqual(result["region_input"], "Méditerranée")
        self.assertEqual(result["region_resolved"], "MEDITERRANEAN")
        self.assertEqual(mocked.call_args.kwargs["region"], "MEDITERRANEAN")

    def test_filter_by_features_temperature_compat(self) -> None:
        # Compat server.py -> tools_stats : min_temperature / max_temperature
        filtered_temp = self.tools_stats.filter_by_features(
            min_temperature=14.0,
            max_temperature=16.0,
            country="France",
            limit=10,
        )
        self.assertTrue(filtered_temp["ok"])
        self.assertGreaterEqual(filtered_temp["matched_count"], 2)
        self.assertIn("temperature_source", filtered_temp["filters_applied"])

        # Ancienne signature conservée (min_bio1 / max_bio1)
        filtered_bio = self.tools_stats.filter_by_features(
            min_bio1=11.0,
            max_bio1=13.0,
            limit=10,
        )
        self.assertTrue(filtered_bio["ok"])
        self.assertGreaterEqual(filtered_bio["matched_count"], 1)

    def test_soilgrid_contract_mapping(self) -> None:
        sample = pd.DataFrame(
            [
                {
                    "surveyId": 1,
                    "Soilgrid-bdod": 12.0,
                    "Soilgrid-cec": 8.0,
                    "Soilgrid-phh2o": 65.0,
                }
            ]
        )
        renamed = sample.rename(columns=self.tools_data._SOIL_RENAME)
        self.assertIn("soil_bdod", renamed.columns)
        self.assertIn("soil_cec", renamed.columns)
        self.assertIn("soil_pH", renamed.columns)

    def test_explain_checkpoint_default_is_configurable(self) -> None:
        with mock.patch.dict(os.environ, {"DEFAULT_EXPLAIN_CHECKPOINT": "fold4_best.pt"}, clear=False):
            self.assertEqual(self.tools_explain._default_checkpoint_name(), "fold4_best.pt")
        sig = inspect.signature(self.tools_explain.explain_model_prediction)
        self.assertIsNone(sig.parameters["checkpoint_name"].default)

    def test_call_llm_handles_missing_token(self) -> None:
        with mock.patch.dict(os.environ, {"LLM_BACKEND": "github_models"}, clear=False):
            response = self.tools_llm._call_llm("system", "user")
        self.assertFalse(response["ok"])
        self.assertIn("GITHUB_TOKEN", response["error"] + response["hint"])

    def test_call_llm_with_mocked_api(self) -> None:
        class _FakeResponse:
            ok = True
            status_code = 200
            text = ""

            @staticmethod
            def json() -> dict:
                return {
                    "choices": [{"message": {"content": "Réponse LLM mockée"}}],
                    "usage": {"total_tokens": 42},
                }

        with mock.patch.dict(
            os.environ,
            {
                "LLM_BACKEND": "github_models",
                "GITHUB_TOKEN": "dummy-token",
            },
            clear=False,
        ):
            with mock.patch("tools_llm.requests.post", return_value=_FakeResponse()):
                response = self.tools_llm._call_llm("system prompt", "user prompt")
                self.assertTrue(response["ok"])
                self.assertEqual(response["text"], "Réponse LLM mockée")

    def test_call_llm_with_mocked_ollama(self) -> None:
        class _FakeOllamaResponse:
            ok = True
            status_code = 200
            text = ""

            @staticmethod
            def json() -> dict:
                return {
                    "message": {"content": "Réponse locale Ollama"},
                    "prompt_eval_count": 12,
                    "eval_count": 18,
                    "total_duration": 123456,
                }

        with mock.patch.dict(
            os.environ,
            {
                "LLM_BACKEND": "ollama",
                "OLLAMA_MODEL": "qwen3:14b",
            },
            clear=False,
        ):
            with mock.patch("tools_llm.requests.post", return_value=_FakeOllamaResponse()):
                response = self.tools_llm._call_llm("system prompt", "user prompt")
                self.assertTrue(response["ok"])
                self.assertEqual(response["provider"], "ollama")
                self.assertEqual(response["text"], "Réponse locale Ollama")

    def test_call_llm_with_mocked_openrouter(self) -> None:
        class _FakeOpenRouterResponse:
            ok = True
            status_code = 200
            text = ""

            @staticmethod
            def json() -> dict:
                return {
                    "choices": [{"message": {"content": "Réponse OpenRouter"}}],
                    "usage": {"total_tokens": 13},
                }

        with mock.patch.dict(
            os.environ,
            {
                "LLM_BACKEND": "openrouter",
                "OPENROUTER_API_KEY": "dummy-key",
                "OPENROUTER_MODEL": "google/gemma-4-31b-it:free",
            },
            clear=False,
        ):
            with mock.patch("tools_llm.requests.post", return_value=_FakeOpenRouterResponse()):
                response = self.tools_llm._call_llm("system prompt", "user prompt")
                self.assertTrue(response["ok"])
                self.assertEqual(response["provider"], "openrouter")
                self.assertEqual(response["text"], "Réponse OpenRouter")

    def test_call_llm_fallback_ollama_to_github(self) -> None:
        class _FakeGitHubResponse:
            ok = True
            status_code = 200
            text = ""

            @staticmethod
            def json() -> dict:
                return {
                    "choices": [{"message": {"content": "Fallback GitHub OK"}}],
                    "usage": {"total_tokens": 21},
                }

        with mock.patch.dict(
            os.environ,
            {
                "LLM_BACKEND": "ollama",
                "GITHUB_TOKEN": "dummy-token",
            },
            clear=False,
        ):
            side_effects = [
                requests.RequestException("ollama down"),
                _FakeGitHubResponse(),
            ]
            with mock.patch("tools_llm.requests.post", side_effect=side_effects):
                response = self.tools_llm._call_llm("system prompt", "user prompt")
                self.assertTrue(response["ok"])
                self.assertEqual(response["provider"], "github_models")
                self.assertEqual(response["text"], "Fallback GitHub OK")

    def test_explain_prediction_with_stubbed_llm(self) -> None:
        fake_llm = {
            "ok": True,
            "provider": "github_models",
            "model": "mock-model",
            "text": "Explication écologique synthétique.",
        }
        fake_quality = {
            "available": True,
            "observation_quality": {
                "confidence_level": "medium",
                "quality_confidence": 0.61,
                "quality_explanation_summary": "Interprétation possible avec réserve.",
            },
        }
        with (
            mock.patch.object(self.tools_llm, "_call_llm", return_value=fake_llm),
            mock.patch.object(self.tools_llm, "_get_observation_quality_context", return_value=fake_quality),
        ):
            response = self.tools_llm.explain_prediction(1, top_k_species=3, language="fr")
            self.assertTrue(response["ok"])
            self.assertEqual(response["survey_id"], "1")
            self.assertIn("top_species", response["context"])
            self.assertIn("prompt", response)
            self.assertEqual(response["context"]["observation_quality"], fake_quality)
            user_prompt = response["prompt"]["user"]
            self.assertIn("Data-quality causes", user_prompt)
            self.assertIn("numeric signal", user_prompt)
            self.assertIn("Observation quality diagnostics", user_prompt)

    def test_explain_prediction_fallback_when_llm_fails(self) -> None:
        llm_error = {
            "ok": False,
            "error": "All LLM backends failed",
            "hint": "simulated",
        }
        fake_quality = {
            "available": True,
            "observation_quality": {
                "confidence_level": "low",
                "quality_confidence": 0.42,
                "quality_explanation_summary": "Hypothèse fragile.",
                "time_series_quality": {"overall_ts_missingness": 0.3},
                "tabular_quality": {"tabular_missing_ratio": 0.2},
            },
        }
        with (
            mock.patch.object(self.tools_llm, "_call_llm", return_value=llm_error),
            mock.patch.object(self.tools_llm, "_get_observation_quality_context", return_value=fake_quality),
        ):
            response = self.tools_llm.explain_prediction(1, top_k_species=3, language="fr")
        self.assertTrue(response["ok"])
        self.assertTrue(response["fallback_used"])
        self.assertEqual(response["provider"], "fallback_no_llm")
        self.assertIsInstance(response.get("llm_error"), dict)
        self.assertIn("Niveau de confiance", response.get("explanation", ""))

    def test_time_series_quality_flags(self) -> None:
        landsat = np.ones((6, 4, 21), dtype=np.float32)
        bioclim = np.ones((4, 19, 12), dtype=np.float32)
        landsat[:, :, 0] = np.nan
        landsat[:, :, 1] = np.nan
        bioclim[:, :, 0] = np.nan

        temporal = {
            "landsat_seasonal_importance_full": [
                {"season": "2000-Hiver", "importance_pct": 55.0},
                {"season": "2000-Print.", "importance_pct": 20.0},
                {"season": "2000-Été", "importance_pct": 10.0},
                {"season": "2000-Automne", "importance_pct": 5.0},
            ] + [
                {"season": f"s{i}", "importance_pct": 0.5} for i in range(17)
            ],
            "bioclim_monthly_importance_full": [
                {"month": "Jan", "importance_pct": 30.0},
                {"month": "Fév", "importance_pct": 10.0},
                {"month": "Mar", "importance_pct": 10.0},
                {"month": "Avr", "importance_pct": 10.0},
                {"month": "Mai", "importance_pct": 5.0},
                {"month": "Jun", "importance_pct": 5.0},
                {"month": "Jul", "importance_pct": 5.0},
                {"month": "Aoû", "importance_pct": 5.0},
                {"month": "Sep", "importance_pct": 5.0},
                {"month": "Oct", "importance_pct": 5.0},
                {"month": "Nov", "importance_pct": 5.0},
                {"month": "Déc", "importance_pct": 5.0},
            ],
        }
        quality = self.tools_explain._compute_time_series_quality(
            {"x_landsat_raw": landsat, "x_climate_raw": bioclim},
            temporal,
        )

        self.assertGreater(quality["landsat_missing_ratio_total"], 0.05)
        self.assertGreater(quality["salient_timesteps_missing_ratio"], 0.20)
        self.assertTrue(quality["flags"]["salient_periods_degraded"])
        self.assertTrue(quality["flags"]["winter_dominant_signal"])

    def test_tabular_quality_metrics(self) -> None:
        fake_ref = {
            "feature_stats": {
                "Bio1": {"median": 10.0, "iqr": 2.0, "mean": 10.0, "std": 1.0, "family": "bioclim"},
                "soil_bdod": {"median": 10.0, "iqr": 2.0, "mean": 10.0, "std": 1.0, "family": "soil"},
                "aux_lon": {"median": 1.0, "iqr": 1.0, "mean": 1.0, "std": 1.0, "family": "aux"},
            }
        }
        tensors = {
            "feature_names": ["Bio1", "soil_bdod"],
            "x_env_raw": np.array([np.nan, 100.0], dtype=np.float32),
            "aux_feature_names": ["aux_lon"],
            "x_aux_raw": np.array([1.0], dtype=np.float32),
        }
        with mock.patch.object(self.tools_explain, "_build_tabular_reference_stats", return_value=fake_ref):
            quality = self.tools_explain._compute_tabular_quality(tensors, salient_feature_names=["soil_bdod"])

        self.assertAlmostEqual(quality["tabular_missing_ratio"], 1 / 3, places=2)
        self.assertEqual(quality["imputed_feature_count"], 1)
        self.assertGreaterEqual(len(quality["extreme_features"]), 1)
        self.assertEqual(quality["salient_tabular_features_low_quality_count"], 1)
        self.assertTrue(quality["flags"]["tabular_many_missing"])

    def test_quality_confidence_mapping(self) -> None:
        low = self.tools_explain._quality_confidence_from_components(
            {
                "salient_timesteps_missing_ratio": 0.6,
                "salient_months_missing_ratio": 0.5,
                "overall_ts_missingness": 0.4,
                "coverage_penalty": 0.6,
            },
            {"tabular_penalty": 0.7},
        )
        self.assertEqual(low["confidence_level"], "low")

        high = self.tools_explain._quality_confidence_from_components(
            {
                "salient_timesteps_missing_ratio": 0.02,
                "salient_months_missing_ratio": 0.03,
                "overall_ts_missingness": 0.02,
                "coverage_penalty": 0.0,
            },
            {"tabular_penalty": 0.05},
        )
        self.assertEqual(high["confidence_level"], "high")

    def test_get_attention_weights_exposes_observation_quality_block(self) -> None:
        fake_tensors = {
            "feature_names": ["Bio1", "soil_bdod"],
            "x_env_raw": np.array([12.0, 8.0], dtype=np.float32),
            "aux_feature_names": ["aux_lon"],
            "x_aux_raw": np.array([3.2], dtype=np.float32),
            "x_landsat_raw": np.zeros((6, 4, 21), dtype=np.float32),
            "x_climate_raw": np.zeros((4, 19, 12), dtype=np.float32),
        }
        fake_attn = {
            "landsat_seasonal_importance": [{"season": "2000-Hiver", "importance_pct": 40.0}],
            "bioclim_monthly_importance": [{"month": "Jan", "importance_pct": 22.0}],
            "landsat_seasonal_importance_full": [
                {"season": f"s{i}", "importance_pct": 100.0 / 21.0} for i in range(21)
            ],
            "bioclim_monthly_importance_full": [
                {"month": m, "importance_pct": 100.0 / 12.0}
                for m in ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
            ],
        }
        fake_ref = {
            "feature_stats": {
                "Bio1": {"median": 12.0, "iqr": 2.0, "mean": 12.0, "std": 1.0, "family": "bioclim"},
                "soil_bdod": {"median": 8.0, "iqr": 2.0, "mean": 8.0, "std": 1.0, "family": "soil"},
                "aux_lon": {"median": 3.0, "iqr": 1.0, "mean": 3.0, "std": 1.0, "family": "aux"},
            }
        }

        with (
            mock.patch.object(
                self.tools_explain,
                "_load_checkpoint",
                return_value=(object(), {"env_dim": 2, "aux_dim": 1}),
            ),
            mock.patch.object(self.tools_explain, "_get_survey_tensors", return_value=fake_tensors),
            mock.patch.object(self.tools_explain, "_extract_attention_weights", return_value=fake_attn),
            mock.patch.object(
                self.tools_explain,
                "_init_artifacts_payload",
                return_value={"enabled": False, "plots": {}, "warnings": []},
            ),
            mock.patch.object(self.tools_explain, "_build_tabular_reference_stats", return_value=fake_ref),
        ):
            response = self.tools_explain.get_attention_weights(212)

        self.assertTrue(response["ok"])
        self.assertIn("observation_quality", response)
        obs = response["observation_quality"]
        self.assertIn("time_series_quality", obs)
        self.assertIn("tabular_quality", obs)
        self.assertIn("confidence_level", obs)
        self.assertIn("quality_confidence", obs)
        self.assertIn("quality_explanation_summary", obs)

    def test_explain_model_prediction_exposes_observation_quality_block(self) -> None:
        class _DummyModel:
            def __call__(self, x_env, x_aux, x_landsat, x_climate):  # noqa: ANN001
                return torch.tensor([[0.8, 0.3, 0.1]], dtype=torch.float32)

            def zero_grad(self) -> None:
                return None

        fake_tensors = {
            "x_env": torch.zeros((1, 2), dtype=torch.float32),
            "x_aux": torch.zeros((1, 1), dtype=torch.float32),
            "x_landsat": torch.zeros((1, 21, 24), dtype=torch.float32),
            "x_climate": torch.zeros((1, 12, 76), dtype=torch.float32),
            "feature_names": ["Bio1", "soil_bdod"],
            "x_env_raw": np.array([12.0, 8.0], dtype=np.float32),
            "aux_feature_names": ["aux_lon"],
            "x_aux_raw": np.array([3.2], dtype=np.float32),
            "x_landsat_raw": np.zeros((6, 4, 21), dtype=np.float32),
            "x_climate_raw": np.zeros((4, 19, 12), dtype=np.float32),
        }
        fake_attn = {
            "landsat_seasonal_importance": [{"season": "2000-Hiver", "importance_pct": 40.0}],
            "bioclim_monthly_importance": [{"month": "Jan", "importance_pct": 22.0}],
            "landsat_seasonal_importance_full": [
                {"season": f"s{i}", "importance_pct": 100.0 / 21.0} for i in range(21)
            ],
            "bioclim_monthly_importance_full": [
                {"month": m, "importance_pct": 100.0 / 12.0}
                for m in ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
            ],
        }
        fake_explanations = [
            {
                "top_env_features": [
                    {"feature": "Bio1", "attribution": 0.3, "abs_score": 0.3},
                    {"feature": "soil_bdod", "attribution": -0.1, "abs_score": 0.1},
                ]
            },
            {
                "top_env_features": [
                    {"feature": "Bio1", "attribution": 0.2, "abs_score": 0.2},
                ]
            },
        ]
        fake_ref = {
            "feature_stats": {
                "Bio1": {"median": 12.0, "iqr": 2.0, "mean": 12.0, "std": 1.0, "family": "bioclim"},
                "soil_bdod": {"median": 8.0, "iqr": 2.0, "mean": 8.0, "std": 1.0, "family": "soil"},
                "aux_lon": {"median": 3.0, "iqr": 1.0, "mean": 3.0, "std": 1.0, "family": "aux"},
            }
        }

        with (
            mock.patch.object(
                self.tools_explain,
                "_load_checkpoint",
                return_value=(
                    _DummyModel(),
                    {
                        "env_dim": 2,
                        "aux_dim": 1,
                        "checkpoint_path": "/tmp/fold4_best.pt",
                        "fold": 4,
                        "score": 0.34,
                        "n_classes": 3,
                    },
                ),
            ),
            mock.patch.object(self.tools_explain, "_get_survey_tensors", return_value=fake_tensors),
            mock.patch.object(self.tools_explain, "_gradient_x_input_attribution", return_value=fake_explanations),
            mock.patch.object(self.tools_explain, "_branch_importance", return_value={"Landsat": 35.0}),
            mock.patch.object(self.tools_explain, "_extract_attention_weights", return_value=fake_attn),
            mock.patch.object(self.tools_explain, "_get_species_ids_from_meta", return_value=np.array([10, 11, 12])),
            mock.patch.object(
                self.tools_explain,
                "_init_artifacts_payload",
                return_value={"enabled": False, "plots": {}, "warnings": []},
            ),
            mock.patch.object(self.tools_explain, "_build_tabular_reference_stats", return_value=fake_ref),
        ):
            response = self.tools_explain.explain_model_prediction(212, top_k=2)

        self.assertTrue(response["ok"])
        self.assertEqual(len(response["predicted_species"]), 2)
        self.assertIn("observation_quality", response)
        obs = response["observation_quality"]
        self.assertIn("time_series_quality", obs)
        self.assertIn("tabular_quality", obs)
        self.assertIn("confidence_level", obs)
        self.assertIn("quality_confidence", obs)
        self.assertIn("quality_explanation_summary", obs)

    def test_artifact_selection_limits_to_four(self) -> None:
        candidates = [
            {"path": "/tmp/branch_importance.png", "tool": "get_branch_importance"},
            {"path": "/tmp/landsat_temporal_importance.png", "tool": "get_attention_weights"},
            {"path": "/tmp/bioclim_monthly_importance.png", "tool": "get_attention_weights"},
            {"path": "/tmp/missingness_timeline_landsat.png", "tool": "get_attention_weights"},
            {"path": "/tmp/missingness_timeline_bioclim.png", "tool": "get_attention_weights"},
            {"path": "/tmp/tabular_quality_overview.png", "tool": "explain_model_prediction"},
            {"path": "/tmp/species_6874_env_attribution.png", "tool": "explain_model_prediction"},
            {"path": "/tmp/malala_shap_importance.png", "tool": "explain_model_prediction"},
        ]
        selected = self.agent._select_relevant_artifact_paths(
            question="Fais un diagnostic complet avec variables importantes, qualité des données et branche dominante.",
            tools_called=["explain_model_prediction", "get_attention_weights", "get_branch_importance"],
            candidates=candidates,
        )
        self.assertLessEqual(len(selected), 4)
        self.assertGreaterEqual(len(selected), 1)
        self.assertTrue(any("branch_importance" in p for p in selected))
        self.assertTrue(
            any(
                "missingness_timeline" in p or "tabular_quality_overview" in p
                for p in selected
            )
        )

    def test_artifact_selection_branch_question_prioritizes_branch_plot(self) -> None:
        candidates = [
            {"path": "/tmp/landsat_temporal_importance.png", "tool": "get_attention_weights"},
            {"path": "/tmp/bioclim_monthly_importance.png", "tool": "get_attention_weights"},
            {"path": "/tmp/branch_importance.png", "tool": "get_branch_importance"},
            {"path": "/tmp/species_6874_env_attribution.png", "tool": "explain_model_prediction"},
        ]
        selected = self.agent._select_relevant_artifact_paths(
            question="Quelle branche du modèle domine ici ?",
            tools_called=["get_branch_importance", "get_attention_weights"],
            candidates=candidates,
        )
        self.assertGreaterEqual(len(selected), 1)
        self.assertIn("branch_importance", selected[0])


if __name__ == "__main__":
    unittest.main()
