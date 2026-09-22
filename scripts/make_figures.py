"""Graphiques du README, de la fiche portfolio et du teaser, à partir de results/scores_kaggle.json.

Usage : uv run python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCORES = json.loads((ROOT / "results" / "scores_kaggle.json").read_text(encoding="utf-8"))
ACCENT = "#FF5C8F"

THEMES = {
    "clair": {"bg": "#FFFFFF", "fg": "#1F2328", "muted": "#8A8F98", "grid": "#E6E8EB", "bar": "#C9CDD3"},
    "sombre": {"bg": "#0D0F12", "fg": "#ECE8DF", "muted": "#8E8A82", "grid": "#23272E", "bar": "#4A4F58"},
}


def fr(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def style(ax, t, k=1.0):
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["fg"], length=0, labelsize=10 * k)


def progression(theme: str, out: Path, size=(10, 5.6), dpi=160, k=1.0):
    t = THEMES[theme]
    steps = SCORES["progression"]
    labels = ["Départ\n(Adrian)", "V1\nensemble 5 plis", "Pré-entraînement\nPO (5 époques)", "Ensemble final\nV1 + V6"]
    values = [s["score"] for s in steps]
    digits = [3, 5, 3, 5]
    fig, ax = plt.subplots(figsize=size, dpi=dpi, facecolor=t["bg"])
    style(ax, t, k)
    xs = range(len(values))
    ax.plot(xs, values, color=t["muted"], lw=1.5, zorder=1)
    colors = [t["bar"]] * (len(values) - 1) + [ACCENT]
    ax.scatter(xs, values, s=[90 * k * k] * (len(values) - 1) + [180 * k * k], color=colors, zorder=2)
    for x, v, d in zip(xs, values, digits):
        ax.annotate(fr(v, d), (x, v), textcoords="offset points", xytext=(0, 14), ha="center",
                    color=ACCENT if x == len(values) - 1 else t["fg"], fontsize=13 * k, fontweight="bold")
    ax.set_xticks(list(xs), labels, fontsize=10.5 * k)
    ax.set_ylim(0.19, 0.245)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fr(v, 2)))
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("F1 Kaggle annoncé en soutenance", color=t["fg"], fontsize=10.5 * k)
    ax.set_title("Progression du score au fil des versions", color=t["fg"], fontsize=15 * k, loc="left", pad=18,
                 fontweight="bold")
    fig.text(0.01, 0.01, "Sources : présentation de soutenance (p. 7, 10, 11) et yasmina/train.py (score de départ).",
             color=t["muted"], fontsize=8.5 * k)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)


def classement(theme: str, out: Path, size=(10, 5.6), dpi=160, k=1.0):
    t = THEMES[theme]
    rows = SCORES["classement_final"]["lignes"]
    names = [("Baseline random" if k > 1 else "Baseline participant random") if r["rang"] is None else f"{r['rang']}. {r['equipe']}" for r in rows]
    values = [r["score"] for r in rows]
    fig, ax = plt.subplots(figsize=size, dpi=dpi, facecolor=t["bg"])
    style(ax, t, k)
    ys = list(range(len(rows)))[::-1]
    for y, r, v in zip(ys, rows, values):
        if r["rang"] is None:
            ax.barh(y, v, color="none", edgecolor=t["muted"], hatch="///", lw=1.2, height=0.62)
        else:
            ax.barh(y, v, color=ACCENT if r["rang"] == 1 else t["bar"], height=0.62)
        ax.text(v + 0.0015, y, fr(v, 5), va="center", color=ACCENT if r["rang"] == 1 else t["fg"],
                fontsize=12 * k, fontweight="bold")
    ax.set_yticks(ys, names, fontsize=11 * k)
    ax.set_xlim(0.17, 0.225)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fr(v, 2)))
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("F1 Kaggle, classement final (axe tronqué à 0,17)", color=t["fg"], fontsize=10.5 * k)
    ax.set_title("Classement final : 1re des 4 équipes" if k > 1 else "Classement final : 1re des 4 équipes, sous la ligne de référence", color=t["fg"], fontsize=15 * k,
                 loc="left", pad=18, fontweight="bold")
    fig.text(0.01, 0.01, "Source : capture du tableau Kaggle, présentation de soutenance p. 11. "
             "La ligne hachurée est une ligne de référence du tableau, pas une équipe.",
             color=t["muted"], fontsize=8.5 * k)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)


def main():
    fig_dir = ROOT / "figures"
    teaser_dir = ROOT / "teaser"
    fig_dir.mkdir(exist_ok=True)
    teaser_dir.mkdir(exist_ok=True)
    progression("clair", fig_dir / "progression-scores.png")
    classement("clair", fig_dir / "classement-final.png")
    progression("sombre", teaser_dir / "figure-teaser.png", size=(16, 9), dpi=100, k=1.7)
    classement("sombre", fig_dir / "classement-final-sombre.png", size=(16, 9), dpi=100, k=1.7)
    print("figures écrites dans", fig_dir, "et", teaser_dir)


if __name__ == "__main__":
    main()
