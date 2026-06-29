"""
Seaborn/Matplotlib chart generation for the Oneiros Dataset Studio.
All charts use a dark theme that matches the app's UI.
Returns raw PNG bytes.
"""

from __future__ import annotations

import io
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Dark theme colours ────────────────────────────────────────────────────────

_BG       = "#0d0e14"
_AXES_BG  = "#18181b"
_GRID     = "#27272a"
_BORDER   = "#3f3f46"
_TEXT     = "#a1a1aa"
_TEXT_LT  = "#d4d4d8"
_ACCENT   = "#8b5cf6"

# Palettes that look good on dark backgrounds
PALETTES = [
    "mako", "rocket", "viridis", "plasma", "magma", "cividis",
    "flare", "crest", "twilight", "icefire",
    "Set2", "tab10", "pastel", "husl",
]

# ── Chart metadata (for the frontend to display) ──────────────────────────────

CHART_DEFS: list[dict] = [
    {"id": "scatter",    "label": "Scatter",         "needsY": True,  "allowHue": True,  "numX": True,  "numY": True},
    {"id": "line",       "label": "Line",             "needsY": True,  "allowHue": True,  "numX": True,  "numY": True},
    {"id": "regression", "label": "Regression",       "needsY": True,  "allowHue": False, "numX": True,  "numY": True},
    {"id": "bar",        "label": "Bar",              "needsY": True,  "allowHue": True,  "numX": False, "numY": True},
    {"id": "box",        "label": "Box",              "needsY": True,  "allowHue": True,  "numX": False, "numY": True},
    {"id": "violin",     "label": "Violin",           "needsY": True,  "allowHue": True,  "numX": False, "numY": True},
    {"id": "strip",      "label": "Strip",            "needsY": True,  "allowHue": True,  "numX": False, "numY": True},
    {"id": "swarm",      "label": "Swarm",            "needsY": True,  "allowHue": True,  "numX": False, "numY": True},
    {"id": "point",      "label": "Point",            "needsY": True,  "allowHue": True,  "numX": False, "numY": True},
    {"id": "count",      "label": "Count",            "needsY": False, "allowHue": True,  "numX": False, "numY": False},
    {"id": "hist",       "label": "Histogram",        "needsY": False, "allowHue": True,  "numX": True,  "numY": False},
    {"id": "kde",        "label": "KDE",              "needsY": False, "allowHue": True,  "numX": True,  "numY": False},
    {"id": "joint",      "label": "Joint",            "needsY": True,  "allowHue": False, "numX": True,  "numY": True},
    {"id": "heatmap",    "label": "Correlation Map",  "needsY": False, "allowHue": False, "numX": False, "numY": False},
    {"id": "pairplot",   "label": "Pair Plot",        "needsY": False, "allowHue": True,  "numX": False, "numY": False},
]


# ── Theme setup ───────────────────────────────────────────────────────────────

def _apply_dark_theme(fig: plt.Figure, axes) -> None:
    """Apply consistent dark theme to any figure + axes."""
    fig.patch.set_facecolor(_BG)

    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(_AXES_BG)
        for spine in ax.spines.values():
            spine.set_color(_BORDER)
        ax.tick_params(colors=_TEXT, labelsize=9)
        ax.xaxis.label.set_color(_TEXT_LT)
        ax.yaxis.label.set_color(_TEXT_LT)
        ax.title.set_color(_TEXT_LT)
        ax.grid(True, color=_GRID, alpha=0.6, linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
        if ax.get_legend() is not None:
            ax.get_legend().get_frame().set_facecolor(_AXES_BG)
            ax.get_legend().get_frame().set_edgecolor(_BORDER)
            for text in ax.get_legend().get_texts():
                text.set_color(_TEXT_LT)


def _save(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_plot(rows: list[dict], config: dict) -> bytes:
    """
    Generate a seaborn/matplotlib chart from a list of row dicts.

    config keys:
        chartType   str   one of CHART_DEFS ids
        xCol        str   x-axis column name (may be empty)
        yCol        str   y-axis column name (may be empty)
        hueCol      str   colour-by column (may be empty / '__none__')
        palette     str   seaborn palette name
        maxRows     int   row limit before sampling (default 4000)
    """
    chart_type: str = config.get("chartType", "scatter")
    x_col: str | None = config.get("xCol") or None
    y_col: str | None = config.get("yCol") or None
    hue_col: str | None = config.get("hueCol") or None
    if hue_col in ("__none__", "", None):
        hue_col = None
    palette: str = config.get("palette", "mako")
    max_rows: int = int(config.get("maxRows", 4000))

    df = pd.DataFrame(rows)

    # Sample for performance
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    # Coerce string columns to numeric where most values parse successfully.
    # Avoids using errors='ignore' which is deprecated in pandas 2.2+.
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors='coerce')
            original_valid = int(df[col].notna().sum())
            if original_valid == 0 or int(coerced.notna().sum()) / original_valid >= 0.8:
                df[col] = coerced

    sns.set_theme(style="darkgrid")

    # Only pass palette when hue is provided — seaborn 0.13+ raises
    # warnings/errors when palette is given without a hue mapping.
    hue_kw: dict = dict(palette=palette) if hue_col else {}
    title_parts = [chart_type.capitalize()]
    if x_col:
        title_parts.append(x_col)
    if y_col:
        title_parts += ["vs", y_col]
    title = " ".join(title_parts)

    # ── Charts that need their own figure (jointplot, pairplot) ───────────────

    if chart_type == "joint":
        if not x_col or not y_col:
            raise ValueError("Joint plot requires both X and Y columns.")
        g = sns.jointplot(data=df, x=x_col, y=y_col, kind="scatter",
                          height=7, marginal_kws={"bins": 20},
                          color=_ACCENT)
        g.fig.patch.set_facecolor(_BG)
        for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
            ax.set_facecolor(_AXES_BG)
            for spine in ax.spines.values():
                spine.set_color(_BORDER)
            ax.tick_params(colors=_TEXT, labelsize=9)
            ax.xaxis.label.set_color(_TEXT_LT)
            ax.yaxis.label.set_color(_TEXT_LT)
        buf = io.BytesIO()
        g.fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=_BG)
        plt.close(g.fig)
        buf.seek(0)
        return buf.read()

    if chart_type == "pairplot":
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            raise ValueError("Pair plot needs at least 2 numeric columns.")
        plot_cols = num_cols[:6]  # cap at 6 for readability
        g = sns.pairplot(
            df[plot_cols + ([hue_col] if hue_col else [])].dropna(),
            hue=hue_col,
            palette=palette if hue_col else None,
            diag_kind="kde",
            plot_kws={"alpha": 0.6},
        )
        g.fig.patch.set_facecolor(_BG)
        for ax in g.axes.flat:
            ax.set_facecolor(_AXES_BG)
            for spine in ax.spines.values():
                spine.set_color(_BORDER)
            ax.tick_params(colors=_TEXT, labelsize=8)
            ax.xaxis.label.set_color(_TEXT_LT)
            ax.yaxis.label.set_color(_TEXT_LT)
        buf = io.BytesIO()
        g.fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=_BG)
        plt.close(g.fig)
        buf.seek(0)
        return buf.read()

    if chart_type == "heatmap":
        num_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
        if num_df.shape[1] < 2:
            raise ValueError("Correlation heatmap needs at least 2 numeric columns.")
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.7 + 1), max(5, len(corr) * 0.6 + 1)))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="magma",
            linewidths=0.4, linecolor=_GRID,
            annot_kws={"size": 8, "color": _TEXT_LT},
            ax=ax, square=True,
        )
        ax.set_title("Correlation Matrix", color=_TEXT_LT, fontsize=13, pad=14)
        ax.tick_params(colors=_TEXT, labelsize=8, rotation=45)
        _apply_dark_theme(fig, [ax])
        return _save(fig)

    # ── Standard single-axis charts ───────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "scatter":
        if not x_col or not y_col:
            raise ValueError("Scatter plot requires X and Y columns.")
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
                        alpha=0.75, **hue_kw)

    elif chart_type == "line":
        if not x_col or not y_col:
            raise ValueError("Line plot requires X and Y columns.")
        sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
                     errorbar=None, **hue_kw)

    elif chart_type == "regression":
        if not x_col or not y_col:
            raise ValueError("Regression plot requires X and Y columns.")
        sns.regplot(data=df, x=x_col, y=y_col, ax=ax,
                    color=_ACCENT,
                    scatter_kws={"alpha": 0.5, "s": 20},
                    line_kws={"linewidth": 2})

    elif chart_type == "bar":
        if not x_col or not y_col:
            raise ValueError("Bar plot requires X and Y columns.")
        # errorbar=None prevents "invalid error value" when groups have 1 sample
        sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
                    errorbar=None, **hue_kw)
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "box":
        if not x_col or not y_col:
            raise ValueError("Box plot requires X and Y columns.")
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
                    **hue_kw)
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "violin":
        if not x_col or not y_col:
            raise ValueError("Violin plot requires X and Y columns.")
        try:
            sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
                           inner="quartile", **hue_kw)
        except Exception:
            # Fall back to box if there's not enough data for violin
            sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
                        **hue_kw)
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "strip":
        if not x_col or not y_col:
            raise ValueError("Strip plot requires X and Y columns.")
        sns.stripplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
                      alpha=0.6, jitter=True, **hue_kw)
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "swarm":
        if not x_col or not y_col:
            raise ValueError("Swarm plot requires X and Y columns.")
        df_sw = df if len(df) <= 500 else df.sample(500, random_state=42)
        sns.swarmplot(data=df_sw, x=x_col, y=y_col, hue=hue_col, ax=ax,
                      **hue_kw)
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "point":
        if not x_col or not y_col:
            raise ValueError("Point plot requires X and Y columns.")
        sns.pointplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax,
                      errorbar=None, **hue_kw)
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "count":
        if not x_col:
            raise ValueError("Count plot requires an X column.")
        sns.countplot(data=df, x=x_col, hue=hue_col, ax=ax, **hue_kw)
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "hist":
        if not x_col:
            raise ValueError("Histogram requires an X column.")
        sns.histplot(data=df, x=x_col, hue=hue_col, ax=ax,
                     bins="auto", kde=True, **hue_kw)

    elif chart_type == "kde":
        if not x_col:
            raise ValueError("KDE plot requires an X column.")
        sns.kdeplot(data=df, x=x_col, hue=hue_col, ax=ax,
                    fill=True, alpha=0.4, **hue_kw)

    else:
        raise ValueError(f"Unknown chart type: {chart_type!r}")

    ax.set_title(title, color=_TEXT_LT, fontsize=13, pad=14)
    if x_col:
        ax.set_xlabel(x_col, fontsize=11)
    if y_col:
        ax.set_ylabel(y_col, fontsize=11)

    _apply_dark_theme(fig, [ax])
    plt.tight_layout()
    return _save(fig)
