"""
test_pipeline.py
================
Teste l'analyse statistique (tools_stats) sur les sorties reelles
du module de filtrage (tools_data).

Usage sur RTX5 :
    DATA_DIR=./data python test_pipeline.py
    DATA_DIR=./data python test_pipeline.py --region ALPINE
"""

import argparse
import sys

import tools_data
import tools_stats


def run(region=None, country=None, elevation_min=0.0, elevation_max=9000.0,
        bioclim_var=None, bioclim_min=None, bioclim_max=None):

    # ── Étape 1 : filtrage ──────────────────────────────────────
    print("\n[1/2] tools_data.apply_filter() ...")
    fo = tools_data.apply_filter(
        region=region,
        country=country,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        bioclim_var=bioclim_var,
        bioclim_min=bioclim_min,
        bioclim_max=bioclim_max,
    )

    status = fo["execution_status"]
    df     = fo["df_combined"]
    print(f"    count    = {status['count']}")
    print(f"    is_empty = {status['is_empty']}")
    print(f"    colonnes = {df.columns.tolist()}")
    print(f"    context  = {fo['filter_context']}")

    if status["is_empty"]:
        print("    Aucun site - arret.")
        sys.exit(0)

    # ── Étape 2 : analyses statistiques ────────────────────────
    print("\n[2/2] tools_stats sur le FilterOutput ...")

    analyses = {
        "env_distribution":        lambda: tools_stats.env_distribution(fo),
        "species_richness":        lambda: tools_stats.species_richness(fo),
        "bioclim_profile":         lambda: tools_stats.bioclim_profile(fo),
        "soil_profile":            lambda: tools_stats.soil_profile(fo),
        "elevation_distribution":  lambda: tools_stats.elevation_distribution(fo),
        "geographic_spread":       lambda: tools_stats.geographic_spread(fo),
        "env_correlation":         lambda: tools_stats.env_correlation(fo),
        "detect_outliers":         lambda: tools_stats.detect_outliers(fo),
        "species_composition":     lambda: tools_stats.species_composition(fo),
        "diversity_indices":       lambda: tools_stats.diversity_indices(fo),
        "species_env_correlation": lambda: tools_stats.species_env_correlation(fo),
        "top_drivers":             lambda: tools_stats.top_drivers(fo),
    }

    for name, fn in analyses.items():
        result = fn()
        ok = result.get("ok", False)
        status_tag = "OK" if ok else "ERREUR"
        print(f"\n  [{status_tag}] {name}")
        if not ok:
            print(f"         {result.get('error')} - {result.get('hint')}")
            continue

        # Affichage résumé par analyse
        if name == "env_distribution":
            print(f"         {result['n_env_cols']} colonnes env")
            if "bio_1" in result["columns"]:
                b = result["columns"]["bio_1"]
                print(f"         bio_1  : mean={b['mean']:.2f}  std={b['std']:.2f}")
            if "elev" in result["columns"]:
                e = result["columns"]["elev"]
                print(f"         elev   : mean={e['mean']:.0f}m  [{e['min']:.0f}, {e['max']:.0f}]")

        elif name == "species_richness":
            s = result["richness_summary"]
            print(f"         {result['n_sites']} sites")
            print(f"         richesse : mean={s['mean']:.1f}  median={s['median']:.1f}  max={s['max']}")

        elif name == "bioclim_profile":
            print(f"         {result['n_bioclim_cols']} variables BioClim")
            print(f"         temperature : {result.get('temperature_vars', [])[:5]}")
            print(f"         precipitation: {result.get('precipitation_vars', [])[:5]}")

        elif name == "soil_profile":
            print(f"         {result['n_soil_cols']} variables sol")
            if "soil_pH" in result.get("variables", {}):
                ph = result["variables"]["soil_pH"]
                print(f"         pH : mean={ph['mean']:.2f}  [{ph['min']:.1f}, {ph['max']:.1f}]")

        elif name == "elevation_distribution":
            s = result.get("summary", {})
            print(f"         colonne : {result.get('col_used')}")
            print(f"         mean={s.get('mean', '?'):.0f}m  "
                  f"median={s.get('median', '?'):.0f}m")
            for b in result.get("histogram", []):
                if b["count"] > 0:
                    bar = "#" * max(1, int(b["pct"] / 5))
                    print(f"         {b['bin_label']:12s} {bar} {b['count']} sites")

        elif name == "geographic_spread":
            c = result.get("centroid", {})
            print(f"         {result['n_sites']} sites")
            print(f"         centroide : lat={c.get('lat'):.4f}  lon={c.get('lon'):.4f}")
            print(f"         emprise   : {result['lat_spread_km']:.0f} km N-S  "
                  f"{result['lon_spread_km']:.0f} km E-O")

        elif name == "env_correlation":
            print(f"         {len(result['columns'])} colonnes  n_obs={result['n_obs']}")
            for p in result.get("top_pairs", [])[:4]:
                print(f"         {p['col_a']} <-> {p['col_b']}  r={p['r']:+.3f}")

        elif name == "detect_outliers":
            print(f"         {result['n_outliers']} sites outliers (z > {result['z_threshold']})")
            for site in result["outlier_sites"][:3]:
                top = site["outlier_cols"][0]
                print(f"         surveyId={site['surveyId']}  "
                      f"{top['col']}={top['value']}  z={top['z_score']:.2f}")

        elif name == "species_composition":
            print(f"         {result['n_species_total']} especes  "
                  f"{result['n_rare_species']} rares (1 seul site)")
            s = result["prevalence_summary"]
            print(f"         prevalence : mean={s['mean']:.2f}%  max={s['max']:.1f}%")
            print(f"         top 5 especes :")
            for sp in result["top_species"][:5]:
                print(f"           id={sp['species_id']}  "
                      f"{sp['n_sites']} sites  ({sp['prevalence_pct']}%)")

        elif name == "diversity_indices":
            sh = result["shannon"]
            si = result["simpson"]
            print(f"         {result['n_sites']} sites")
            print(f"         Shannon : mean={sh['mean']:.3f}  "
                  f"median={sh['median']:.3f}  max={sh['max']:.3f}")
            print(f"         Simpson : mean={si['mean']:.3f}  "
                  f"median={si['median']:.3f}  max={si['max']:.3f}")
            if result["top_diverse_sites"]:
                top = result["top_diverse_sites"][0]
                print(f"         site le plus divers : surveyId={top['surveyId']}  "
                      f"H={top['shannon']:.3f}  S={top['richness']}")

        elif name == "species_env_correlation":
            print(f"         {result['n_sites']} sites  methode={result['method']}")
            for sp in result["species"][:3]:
                print(f"         espece {sp['species_id']} "
                      f"({sp['prevalence_pct']}% des sites) :")
                for pred in sp["top_predictors"][:3]:
                    print(f"           {pred['env_var']:20s} r={pred['r']:+.3f} "
                          f"({pred['direction']})")

        elif name == "top_drivers":
            print(f"         {result['n_sites']} sites  {result['n_env_vars']} variables env")
            print(f"         importance par famille : {result['family_importance']}")
            print(f"         top drivers :")
            for d in result["drivers"][:5]:
                print(f"           #{d['rank']} {d['env_var']:20s} "
                      f"score={d['composite_score']:5.1f}/100  "
                      f"r_richesse={d['richness_corr']:+.3f}  "
                      f"votes={d['species_votes']}  ({d['direction']})")
            print(f"         LLM summary :")
            print(f"           {result['llm_summary']}")

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region",        default=None)
    parser.add_argument("--country",       default=None)
    parser.add_argument("--elevation-min", type=float, default=0.0)
    parser.add_argument("--elevation-max", type=float, default=9000.0)
    parser.add_argument("--bioclim-var",   default=None)
    parser.add_argument("--bioclim-min",   type=float, default=None)
    parser.add_argument("--bioclim-max",   type=float, default=None)
    args = parser.parse_args()

    run(
        region        = args.region,
        country       = args.country,
        elevation_min = args.elevation_min,
        elevation_max = args.elevation_max,
        bioclim_var   = args.bioclim_var,
        bioclim_min   = args.bioclim_min,
        bioclim_max   = args.bioclim_max,
    )
