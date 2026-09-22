# MCP Server - GeoLifeCLEF 2026

Ce dossier contient les outils MCP du projet:

- `server.py`: serveur MCP (fichier réservé, ne pas modifier ici).
- `tools_data.py`: filtres données et accès images (Raihan).
- `tools_stats.py`: statistiques écologiques (Malala).
- `tools_llm.py`: explications LLM (Guilhem).

## 1) Prérequis

- Python 3.9+.
- Environnement virtuel actif (ex: `venv_challenge`).
- SDK MCP Python installé dans le venv: `pip install mcp`.
- Accès aux données sur serveur (`CSV` + `SatelitePatches`).
- Token GitHub Models (fine-grained PAT avec permission `Models: Read`).

## 2) Configuration `.env`

Créer un `.env` à la racine du repo (ou adapter celui existant):

```dotenv
DATA_DIR=./data
DATA_CSV_PATH=./data/GLC25_PA_metadata_train.csv
TIFF_ROOT=./data/SatelitePatches/PA-train

GITHUB_TOKEN=REPLACE_WITH_YOUR_GITHUB_TOKEN
GITHUB_MODEL=openai/gpt-4.1
GITHUB_MODELS_ENDPOINT=https://models.github.ai/inference/chat/completions
GITHUB_API_VERSION=2026-03-10
GITHUB_API_TIMEOUT=30
```

Variables LLM obligatoires pour `tools_llm.py`:

- `GITHUB_TOKEN`
- `GITHUB_MODEL`
- `GITHUB_MODELS_ENDPOINT`

Variables LLM local recommandées (Ollama, backend par défaut):

- `LLM_BACKEND` (`ollama` par défaut, sinon `openrouter`, `github_models` ou `auto`)
- `OLLAMA_BASE_URL` (défaut `http://localhost:11434`)
- `OLLAMA_MODEL` (défaut `qwen3:8b` pour latence plus basse)
- `OLLAMA_API_TIMEOUT` (défaut `45`)

Variables LLM OpenRouter (optionnel, souvent plus rapide que local sur laptop):

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (ex: `google/gemma-4-31b-it:free`)
- `OPENROUTER_BASE_URL` (défaut `https://openrouter.ai/api/v1/chat/completions` pour `tools_llm`)
- `OPENROUTER_API_TIMEOUT` (défaut `30`)
- `OPENROUTER_HTTP_REFERER` (optionnel, recommandé par OpenRouter)
- `OPENROUTER_APP_TITLE` (optionnel)

Note:

- `tools_llm.py` utilise l’endpoint complet `/chat/completions`.
- `langchain_mcp_agent.py` attend la base `https://openrouter.ai/api/v1`
  (le code normalise automatiquement si `/chat/completions` est fourni).

Variables recommandées:

- `GITHUB_API_VERSION`
- `GITHUB_API_TIMEOUT`

Variables data recommandées:

- `DATA_DIR`: racine dataset (obligatoire si `tools_data` doit merger les CSV environnementaux).
- `DATA_CSV_PATH`: CSV principal metadata PA (utilisé aussi par les tests legacy).
- `TIFF_ROOT`: racine réelle des patches TIFF (souvent `.../SatelitePatches/PA-train`).
- `DEFAULT_EXPLAIN_CHECKPOINT`: checkpoint explicabilité par défaut
  (défaut `fold4_best.pt`).

Important (multi-CSV):

- `tools_data.py` ne lit pas uniquement `GLC25_PA_metadata_train.csv`.
- Pour les fonctions de filtrage avancé/statistiques, il lit aussi plusieurs CSV dans
  `DATA_DIR/EnvironmentalValues/...` (bioclim, soilgrids, elevation, etc.).
- Donc sur serveur, configurer `DATA_DIR` correctement est indispensable.

Charger le fichier avant exécution:

```bash
set -a && source .env && set +a
```

## 3) Créer le token GitHub Models

Depuis GitHub:

1. `Settings`
2. `Developer settings`
3. `Personal access tokens` -> `Fine-grained tokens`
4. Créer un token (nom ex: `Plant_predict`)
5. Permission minimale: `Models` -> `Read`
6. Copier le token et le placer dans `GITHUB_TOKEN`

Important:

- Un abonnement Copilot VS Code ne remplace pas `GITHUB_TOKEN` côté Python.
- Ne jamais commiter `.env`.

## 4) Vérifications rapides

### 4.1 Test direct API GitHub Models

```bash
curl -sS -X POST "$GITHUB_MODELS_ENDPOINT" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: ${GITHUB_API_VERSION:-2026-03-10}" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$GITHUB_MODEL"'","messages":[{"role":"user","content":"Say hello in French"}]}'
```

Attendu: une réponse JSON avec `"choices"` et un message assistant.

### 4.2 Test Python `tools_llm.explain_term`

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, "MCP_server")
import tools_llm

resp = tools_llm.explain_term("Bio1", language="fr")
print(resp)
PY
```

Attendu: `{"ok": True, ...}` avec un texte d'explication.

## 5) API attendue des modules

### `tools_data.py` (Raihan)

- `filter_by_region(min_lat, max_lat, min_lon, max_lon, limit=1000)`
- `filter_by_region_name(region_name, limit=1000, case_sensitive=False)`
- `get_random_images(n=10, seed=None, region_name=None)`
- `select_model(model_name)`
- `get_server_status()`

### `tools_stats.py` (Malala)

- `get_species_stats(top_k=20, rare_threshold=5, records=None)`
- `get_env_features(survey_id)`
- `filter_by_features(min_altitude=None, max_altitude=None, min_temperature=None, max_temperature=None, country=None, limit=1000)`
- `get_cooccurrences(species_id, top_k=10, records=None)`
- `get_top_drivers(region=None, country=None, elevation_min=0.0, elevation_max=9000.0, bioclim_var=None, bioclim_min=None, bioclim_max=None, n_species_min=None, top_k_species=10, top_k_vars=8)`
  - `region` accepte les alias usuels (ex: `Méditerranée`, `Mediterranean`, `MEDITERRANEAN`)

### `tools_llm.py` (Guilhem)

- `explain_prediction(survey_id, top_k_species=5, language="fr")`
- `explain_term(term, language="fr")`
- `describe_survey(survey_id, language="fr")`
- `_call_llm(system_prompt, user_prompt, model=None, temperature=0.2, max_tokens=700)`

### `tools_explain.py` (Raihan)

- `explain_model_prediction(survey_id, top_k=5, checkpoint_name=None)`
- `get_branch_importance(survey_id, checkpoint_name=None)`
- `get_attention_weights(survey_id, checkpoint_name=None)`
- `list_available_checkpoints()`

Sorties explainability:

- Les tools retournent maintenant un bloc `artifacts` (chemins PNG) quand activé.
- Les tools `explain_model_prediction` et `get_attention_weights` retournent aussi:
  - `observation_quality.time_series_quality`
  - `observation_quality.tabular_quality`
  - `observation_quality.confidence_level` (`high|medium|low`)
  - `observation_quality.quality_confidence` (score `[0,1]`)
  - `observation_quality.quality_explanation_summary` (résumé prudent)
- Méthodes utilisées: Gradient×Input (proxy SHAP), importance temporelle par gradients (proxy attention), norme L2 des embeddings de branche.
- Backend additionnel optionnel: `Malala/explainability.py` (SHAP/attention avancés) exposé dans `artifacts.malala_backend`.
- Nouveaux artefacts qualité (si plots activés):
  - `missingness_timeline_landsat_png`
  - `missingness_timeline_bioclim_png`
  - `tabular_quality_overview_png`
- Variables:
  - `EXPLAIN_SAVE_PLOTS=true|false` (défaut: `true`)
  - `EXPLAIN_ARTIFACTS_DIR=/chemin/sortie` (défaut: `outputs/mcp_explainability`)
  - `EXPLAIN_TIFF_QUALITY_MODE=auto|always|off` (défaut: `auto`)
  - `EXPLAIN_ENABLE_MALALA_XAI=true|false` (défaut: `false`)
  - `EXPLAIN_MALALA_RUN_SHAP=true|false` (défaut: `false`)
  - `EXPLAIN_MALALA_RUN_ATTENTION=true|false` (défaut: `false`)
  - `EXPLAIN_MALALA_RUN_GRADCAM=true|false` (défaut: `false`)

Notes backend Malala:

- GradCAM nécessite une branche image active dans le checkpoint + `captum` + lecture TIFF (`rasterio`).
- SHAP repose sur `shap` (KernelExplainer, plus lent).

Checkpoint par défaut:

- Si `checkpoint_name` n'est pas fourni, `tools_explain.py` utilise
  `DEFAULT_EXPLAIN_CHECKPOINT` (fallback `fold4_best.pt`).

## 6) Agent local LangChain + MCP + LangSmith

Fichier:

- `langchain_mcp_agent.py`

Installation:

```bash
pip install -r MCP_server/requirements_agent.txt
```

Dépendances XAI avancées (optionnel):

```bash
pip install -r MCP_server/requirements_xai_optional.txt
```

Exécution one-shot:

```bash
python MCP_server/langchain_mcp_agent.py \
  --question "Quelles sont les espèces les plus rares en France ?"
```

Exécution one-shot avec OpenRouter:

```bash
OPENROUTER_API_KEY=... \
AGENT_LLM_PROVIDER=openrouter \
OPENROUTER_MODEL=google/gemma-4-31b-it:free \
python MCP_server/langchain_mcp_agent.py \
  --question "Quelles sont les espèces les plus rares en France ?"
```

Mode interactif:

```bash
python MCP_server/langchain_mcp_agent.py --interactive
```

Smoke test + latence médiane sur 10 questions:

```bash
python MCP_server/agent_smoke.py
```

Smoke test OpenRouter:

```bash
OPENROUTER_API_KEY=... \
python MCP_server/agent_smoke.py --provider openrouter --model google/gemma-4-31b-it:free
```

UI web locale (sans extension payante):

```bash
python MCP_server/mcp_web_ui.py --port 7860
```

Puis ouvrir: `http://127.0.0.1:7860`

Commande unique (startup auto + ouverture navigateur sur macOS):

```bash
./scripts/run_mcp_ui.sh
```

Options:

```bash
./scripts/run_mcp_ui.sh --port 7870 --host 127.0.0.1
./scripts/run_mcp_ui.sh --no-open
```

Sélection intelligente des artefacts (UI/agent):

- L'agent ne renvoie plus tous les PNG: il sélectionne par défaut les `4` plus pertinents selon la question et les tools appelés.
- Priorisation contextuelle (exemples):
  - question sur branche dominante -> `branch_importance.png`
  - question temporelle (saisons/mois) -> plots Landsat/Bioclim
  - question qualité/incertitude -> plots missingness + tabulaire
- Variable optionnelle:
  - `AGENT_MAX_ARTIFACTS` (défaut `4`, borne `1..12`)

Observabilité LangSmith (optionnel):

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=GeoLifeCLEF-MCP
```

## 7) Intégration serveur MCP

`server.py` doit importer et exposer les fonctions ci-dessus comme tools MCP.
Ce README n'impose pas d'implémentation serveur spécifique.

### Test local de démarrage MCP (stdio)

```bash
cd .
set -a && source .env && set +a
python MCP_server/server.py
```

Attendu: le process reste actif (il ne doit pas quitter immédiatement).
Si le process quitte tout de suite, Copilot ne pourra pas s'initialiser.

Pour VS Code (MCP):

- Ajouter votre serveur MCP dans la configuration MCP de VS Code.
- Vérifier que le process charge bien `.env` avant démarrage.

## 8) Erreurs fréquentes

- `Missing GITHUB_TOKEN`:
  - `GITHUB_TOKEN` absent ou vide dans `.env`.
- `Missing GITHUB_MODEL` / `Missing GITHUB_MODELS_ENDPOINT`:
  - la variable est requise et non définie.
- `LLM API error` avec `status_code: 400`:
  - modèle invalide pour votre compte.
  - endpoint non conforme.
  - payload mal formé.
- `bash: jq: command not found`:
  - utiliser `python3 -m json.tool` pour formater la sortie `curl`.

## 9) Notes opérationnelles

- Les chemins de données dépendent du serveur. Adapter `.env` sur chaque machine.
- Si `SatelitePatches` contient un sous-dossier `PA-train`, pointer `TIFF_ROOT` dessus.
  Sinon pointer directement vers le dossier qui contient la structure `.../<2digits>/<2digits>/<surveyId>.tiff`.
- Les fonctions LLM `explain_prediction` et `describe_survey` dépendent de `tools_data.py` et `tools_stats.py`.
- `explain_term` peut être testé indépendamment des données terrain.
