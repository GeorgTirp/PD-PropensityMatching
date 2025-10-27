from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

def _wilcoxon_paired(x: pd.Series, y: pd.Series) -> dict:
    paired = pd.concat([x, y], axis=1, join="inner").dropna()
    if len(paired) < 2:
        return {"n_pairs": len(paired), "stat": np.nan, "p_value": np.nan}
    stat, p_value = wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1])
    return {"n_pairs": len(paired), "stat": stat, "p_value": p_value}

def _mannwhitney_ind(x: pd.Series, y: pd.Series) -> dict:
    x, y = x.dropna(), y.dropna()
    if len(x) < 2 or len(y) < 2:
        return {"n_x": len(x), "n_y": len(y), "stat": np.nan, "p_value": np.nan}
    stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
    return {"n_x": len(x), "n_y": len(y), "stat": stat, "p_value": p_value}


@dataclass(frozen=True)
class MatchedGroups:
    custom: pd.DataFrame
    ppmi: pd.DataFrame


COVARIATE_LABELS = {
    "UPDRS_on": "UPDRS On Med",
    "TimeSinceDiag": "Time Since Diagnosis",
    "MoCA_sum_pre": "MoCA Total Baseline",
    "AGE_AT_OP": "Age at Operation",
    "AGE_AT_BASELINE": "Age at Baseline",
    # label for MoCA post will be inserted dynamically if present
}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

# --- minimal colors used for MoCA plots ---
_RC_COLORS = {
    "improvement": "#04E762",   # green
    "deterioration": "#FF5714", # orange/red
    "no_change": "grey",
    "scatter": "black",
}

def _prep_moca_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with both MoCA_sum_pre and MoCA_sum_post numeric; median-impute any missing first."""
    cols = ["MoCA_sum_pre", "MoCA_sum_post"]
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        # median impute MoCA
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    # after impute, no row should be dropped for MoCA
    return out[cols]

def _half_violin(ax, x_pos: float, values: np.ndarray, side: str, color: str, width: float = 0.5, shift: float = 0.075):
    """
    Draw a single half-violin at x_pos. side in {'left','right'}.
    """
    v = sns.violinplot(
        y=values,
        inner=None,
        cut=0,
        linewidth=0,
        color=mcolors.to_rgba(color, alpha=0.75),  # darker overall body
        width=width,
        ax=ax,
    )
    # Keep only the last collection we just drew
    for collection in ax.collections[-1:]:
        if not isinstance(collection, PolyCollection):
            continue
        fc = mcolors.to_rgba(color, alpha=0.75)   # darker fill
        ec = mcolors.to_rgba(color, alpha=1.0)
        collection.set_facecolor(fc)
        collection.set_edgecolor(ec)
        for path in collection.get_paths():
            verts = path.vertices
            mean_x = np.mean(verts[:, 0])
            if side == "left":
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, mean_x)
                delta = x_pos - shift
            else:  # "right"
                verts[:, 0] = np.clip(verts[:, 0], mean_x, np.inf)
                delta = x_pos + shift
            path.vertices[:, 0] += delta

def _raincloud_ax_for_moca(ax, data: pd.DataFrame, title: str):
    """
    Draw a single-panel raincloud for MoCA pre vs post on an axes.
    `data` must have columns ['MoCA_sum_pre','MoCA_sum_post'].
    """
    sns.set_theme(style="white", context="paper")
    x_positions = np.array([1, 2])
    pre = data["MoCA_sum_pre"].to_numpy()
    post = data["MoCA_sum_post"].to_numpy()
    n = len(data)

    # Violin halves
    base_col = sns.color_palette("deep")[0]
    _half_violin(ax, 1, pre,  "left",  base_col)
    _half_violin(ax, 2, post, "right", base_col)

    # Slim boxplots without outlier markers
    box_rgba = mcolors.to_rgba(base_col, alpha=0.33)
    edge_rgba = mcolors.to_rgba("black", alpha=1)
    sns.boxplot(
        data=data.rename(columns={"MoCA_sum_pre": "Pre", "MoCA_sum_post": "Post"}),
        width=0.15,
        showcaps=True,
        whiskerprops={"color": edge_rgba},
        medianprops={"color": edge_rgba},
        capprops={"color": edge_rgba},
        showfliers=False,  # <- NO outlier circles
        palette=[box_rgba, box_rgba],
        boxprops={"facecolor": box_rgba, "edgecolor": edge_rgba},
        ax=ax,
    )

    # Scatter with tiny jitter and connecting lines (MoCA: higher = improvement)
    rng = np.random.default_rng(42)
    x_noise = rng.uniform(-0.05, 0.05, size=n)
    y_noise = rng.uniform(-0.4, 0.4, size=n)  # light vertical jitter

    x_pre  = 1 + x_noise
    y_pre  = pre + y_noise
    x_post = 2 + x_noise
    y_post = post + y_noise

    ax.scatter(x_pre,  y_pre,  s=12, color=_RC_COLORS["scatter"], alpha=0.9, zorder=3)
    ax.scatter(x_post, y_post, s=12, color=_RC_COLORS["scatter"], alpha=0.9, zorder=3)

    for i in range(n):
        slope = y_post[i] - y_pre[i]
        if np.isclose(slope, 0, atol=1e-2):
            c = _RC_COLORS["no_change"]
        elif slope > 0:
            # MoCA higher -> improvement
            c = _RC_COLORS["improvement"]
        else:
            c = _RC_COLORS["deterioration"]
        ax.add_line(Line2D([x_pre[i], x_post[i]], [y_pre[i], y_post[i]], color=c, alpha=0.5, linewidth=1.2, zorder=2))

    # Labels, ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Pre", "Post"], fontsize=11)
    ax.set_ylabel("MoCA total", fontsize=11)
    ax.set_title(f"{title} (N = {n})", fontsize=14, loc="left", pad=10)
    ax.set_xlim(0.5, 2.5)

    # Dynamic limits with padding (≈10% bottom, ≈20% top so legend never overlaps)
    y_min = float(np.nanmin([pre.min(), post.min()]))
    y_max = float(np.nanmax([pre.max(), post.max()]))
    span = y_max - y_min if y_max != y_min else 1.0
    pad_bottom = 0.10 * span
    pad_top = 0.20 * span  # extra space for legend
    ax.set_ylim(max(0, y_min - pad_bottom), y_max + pad_top)  # clamp lower at 0, allow >30 above

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    sns.despine(ax=ax)
    ax.grid(False)

def raincloud_moca_side_by_side(groups: MatchedGroups, output_path: Path, show: bool = False) -> None:
    """
    Build a 1x2 figure:
      Left  : Tübingen DBS  (MoCA_sum_pre vs MoCA_sum_post)
      Right : PPMI matched  (MoCA_sum_pre vs MoCA_sum_post)
    Saves svg and SVG to `output_path` (stem used for both).
    Also writes a text field below with the mean (Pre − Post) difference and standard deviation for both groups.
    """
    # Align by matched index and keep pairs with both measurements (after impute, all kept)
    left  = _prep_moca_pairs(groups.custom)
    right = _prep_moca_pairs(groups.ppmi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    _raincloud_ax_for_moca(axes[0], left,  "Tübingen DBS")
    _raincloud_ax_for_moca(axes[1], right, "PPMI matched")

    # One legend explaining line colors
    legend_elems = [
        Line2D([0], [0], color=_RC_COLORS["improvement"], lw=2, label="Improvement"),
        Line2D([0], [0], color=_RC_COLORS["deterioration"], lw=2, label="Deterioration"),
        Line2D([0], [0], color=_RC_COLORS["no_change"], lw=2, label="No change"),
    ]
    axes[0].legend(handles=legend_elems, loc="lower left", frameon=False)

    # --- Mean (Pre − Post) difference and SD (per group) ---
    left_diff = left["MoCA_sum_pre"] - left["MoCA_sum_post"]
    right_diff = right["MoCA_sum_pre"] - right["MoCA_sum_post"]
    left_mean, left_std = float(left_diff.mean()), float(left_diff.std(ddof=1))
    right_mean, right_std = float(right_diff.mean()), float(right_diff.std(ddof=1))

    # place two small text fields centered under each subplot
    fig.text(
        0.25, 0.02,
        f"Pre − Post (DBS): mean {left_mean:.2f}, SD {left_std:.2f}",
        ha="center", va="center",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.25")
    )
    fig.text(
        0.75, 0.02,
        f"Pre − Post (PPMI): mean {right_mean:.2f}, SD {right_std:.2f}",
        ha="center", va="center",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.25")
    )

    plt.tight_layout(rect=(0, 0.05, 1, 1))  # leave room at bottom for text
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    return df

def load_groups(custom_path: Path, ppmi_path: Path) -> MatchedGroups:
    custom_df = _load_csv(custom_path)
    ppmi_df   = _load_csv(ppmi_path)

    # Prefer a stable ID join if available
    id_key_ppmi = None
    id_key_custom = None
    for cand in ("matched_custom_id", "matched_custom_PATNO", "matched_custom_patno"):
        if cand in ppmi_df.columns:
            id_key_ppmi = cand
            break
    # Guess the id column on the custom side
    for cand in ("PATNO", "patno", "ID", "id", "subject_id", "matched_custom_id"):
        if cand in custom_df.columns:
            id_key_custom = cand
            break

    if id_key_ppmi and id_key_custom:
        # 1:1 alignment by ID
        ppmi_aligned = ppmi_df.dropna(subset=[id_key_ppmi]).copy()
        ppmi_aligned[id_key_ppmi] = ppmi_aligned[id_key_ppmi].astype(str)
        custom_df[id_key_custom]  = custom_df[id_key_custom].astype(str)

        # inner-join order: follow the PPMI rows (one row per matched treated)
        custom_aligned = custom_df.merge(
            ppmi_aligned[[id_key_ppmi]].rename(columns={id_key_ppmi: "__join_id"}),
            left_on=id_key_custom, right_on="__join_id", how="inner"
        ).drop(columns="__join_id")

        # Now place both in the same order keyed by ID
        custom_aligned = custom_aligned.set_index(id_key_custom).loc[ppmi_aligned[id_key_ppmi].values]
        ppmi_aligned   = ppmi_aligned.set_index(id_key_ppmi)

        # Sanity: equal lengths
        if len(custom_aligned) != len(ppmi_aligned):
            raise ValueError("Aligned cohorts must have identical sizes; check for duplicate IDs.")
        return MatchedGroups(custom=custom_aligned.reset_index(drop=False),
                             ppmi=ppmi_aligned.reset_index(drop=False))

    # --- Fallback: align by matched_custom_index (original indices from the matcher) ---
    if "matched_custom_index" not in ppmi_df.columns:
        raise ValueError(
            "PPMI file has no 'matched_custom_id' or 'matched_custom_index' to align with the treated cohort."
        )

    # If the custom file already contains that index as a column, use it;
    # otherwise assume the CSV preserved the original index in a column with that name.
    if "matched_custom_index" in custom_df.columns:
        custom_df = custom_df.set_index("matched_custom_index")
    elif "custom_index" in custom_df.columns:
        custom_df = custom_df.set_index("custom_index")
    else:
        # Best effort: do NOT reset_index; try to use the current index as-is
        # (works only if the CSV saved the original index)
        pass

    match_indices = list(dict.fromkeys(ppmi_df["matched_custom_index"].tolist()))
    missing = [i for i in match_indices if i not in custom_df.index]
    if missing:
        # Warn loudly: this is exactly what causes tiny N like 21–22
        print(f"[WARN] {len(missing)} matched_custom_index values are not in custom_df.index "
              f"(e.g., first few: {missing[:5]}). "
              f"Prefer aligning by 'matched_custom_id' to avoid index drift.")

    # Align in the PPMI order
    custom_matched = custom_df.loc[[i for i in match_indices if i in custom_df.index]].copy()
    ppmi_matched   = ppmi_df.set_index("matched_custom_index").loc[custom_matched.index].copy()

    if len(custom_matched) != len(ppmi_matched):
        raise ValueError("Aligned cohorts must have identical sizes; check the matching inputs.")
    return MatchedGroups(custom=custom_matched.reset_index(drop=False),
                         ppmi=ppmi_matched.reset_index(drop=False))



def plot_moca_radar(
    groups: MatchedGroups,
    output_stem: Path,
    score_suffix: str,
    title: str,
) -> None:
    # Only include MoCA subscores, not the total
    moca_cols = sorted(
        col
        for col in groups.custom.columns
        if col.startswith("MoCA_")
        and col.endswith(score_suffix)
        and col != f"MoCA{score_suffix}"
    )
    moca_cols = [col for col in moca_cols if col in groups.ppmi.columns]
    if not moca_cols:
        raise ValueError(f"No MoCA columns ending with '{score_suffix}' found for radar plot.")

    custom_means = groups.custom[moca_cols].mean(numeric_only=True)
    ppmi_means = groups.ppmi[moca_cols].mean(numeric_only=True)

    # Set up angles and data for radar
    angles = np.linspace(0, 2 * np.pi, len(moca_cols), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    custom_vals = np.concatenate([custom_means.values, custom_means.values[:1]])
    ppmi_vals = np.concatenate([ppmi_means.values, ppmi_means.values[:1]])

    # Determine radial limit
    radial_max = float(np.nanmax(np.concatenate([custom_means.values, ppmi_means.values])))
    if not np.isfinite(radial_max) or radial_max <= 0:
        radial_max = 1.0

    # Create radar figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, custom_vals, label="Tübingen DBS", color="#2a9d8f")
    ax.fill(angles, custom_vals, alpha=0.25, color="#2a9d8f")
    ax.plot(angles, ppmi_vals, label="PPMI matched", color="#264653")
    ax.fill(angles, ppmi_vals, alpha=0.25, color="#264653")

    # Axis labels
    ax.set_xticks(angles[:-1])
    axis_labels = []
    for col in moca_cols:
        label = col.replace("MoCA_", "").replace(score_suffix, "")
        axis_labels.append(label)
    ax.set_xticklabels(axis_labels)
    ax.set_ylim(0, radial_max)

    # clear, properly spaced title (no overlapping)
    ax.set_title(title, pad=25, y=1.08, fontsize=14, weight="bold")

    # Legend placement
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)



def _welch_ttest(x: pd.Series, y: pd.Series) -> dict:
    x_clean = x.dropna()
    y_clean = y.dropna()
    if len(x_clean) < 2 or len(y_clean) < 2:
        return {
            "n_custom": len(x_clean),
            "n_ppmi": len(y_clean),
            "mean_custom": x_clean.mean(),
            "mean_ppmi": y_clean.mean(),
            "t_stat": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "n_pairs": 0,
        }

    t_stat, p_value = stats.ttest_ind(x_clean, y_clean, equal_var=False)
    pooled_sd = np.sqrt(
        ((len(x_clean) - 1) * x_clean.var(ddof=1) + (len(y_clean) - 1) * y_clean.var(ddof=1))
        / (len(x_clean) + len(y_clean) - 2)
    )
    effect = (x_clean.mean() - y_clean.mean()) / pooled_sd if pooled_sd else np.nan

    return {
        "n_custom": len(x_clean),
        "n_ppmi": len(y_clean),
        "mean_custom": x_clean.mean(),
        "mean_ppmi": y_clean.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "effect_size": effect,
        "n_pairs": min(len(x_clean), len(y_clean)),
    }


def _paired_ttest(x: pd.Series, y: pd.Series) -> dict:
    paired = pd.concat([x, y], axis=1, join="inner")
    paired.columns = ["custom", "ppmi"]
    paired = paired.dropna()
    if len(paired) < 2:
        return {"n_pairs": len(paired), "t_stat": np.nan, "p_value": np.nan, "effect_size": np.nan}

    t_stat, p_value = stats.ttest_rel(paired["custom"], paired["ppmi"])
    diff = paired["custom"] - paired["ppmi"]
    sd_diff = diff.std(ddof=1)
    effect = diff.mean() / sd_diff if sd_diff else np.nan
    return {"n_pairs": len(paired), "t_stat": t_stat, "p_value": p_value, "effect_size": effect}


def _pvalue_to_stars(p: float) -> str:
    if pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def ttest_plots(
    groups: MatchedGroups,
    variables: Sequence[str],
    output_dir: Path,
) -> pd.DataFrame:
    sns.set_theme(style="whitegrid", context="paper")
    results = []
    available_vars = [v for v in variables if v in groups.custom.columns and v in groups.ppmi.columns]

    base_palette = sns.color_palette("deep")
    box_color = base_palette[0]

    for var in available_vars:
        pretty = COVARIATE_LABELS.get(var, var.replace("_", " "))
        x = pd.to_numeric(groups.custom[var], errors="coerce")
        y = pd.to_numeric(groups.ppmi[var], errors="coerce")

        # median-impute for MoCA variables only (keeps other plots unchanged)
        if "MoCA" in var:
            if x.isna().any():
                x = x.fillna(x.median())
            if y.isna().any():
                y = y.fillna(y.median())

        # Always compute parametric tests
        welch = _welch_ttest(x, y)
        paired = _paired_ttest(x, y)

        is_moca = "MoCA" in var
        mw_p = np.nan
        wil_p = np.nan
        n_pairs_shown = paired["n_pairs"]

        # Add nonparametric tests for bounded MoCA variables
        if is_moca:
            mw = _mannwhitney_ind(x, y)
            wil = _wilcoxon_paired(x, y)
            mw_p = mw["p_value"]
            wil_p = wil["p_value"]

        # ---- store results ----
        stats_row = {
            "variable": var,
            "label": pretty,
            "n_custom": welch["n_custom"],
            "n_ppmi": welch["n_ppmi"],
            "n_pairs_paired": paired["n_pairs"],
            "welch_p": welch["p_value"],
            "paired_p": paired["p_value"],
            "significance_welch": _pvalue_to_stars(welch["p_value"]),
            "significance_paired": _pvalue_to_stars(paired["p_value"]),
            "mannwhitney_p": mw_p,
            "wilcoxon_p": wil_p,
        }
        results.append(stats_row)

        # ---- plotting ----
        df = pd.DataFrame(
            {
                "group": ["Tübingen DBS"] * len(x) + ["PPMI"] * len(y),
                "value": pd.concat([x, y], ignore_index=True),
            }
        ).dropna(subset=["value"])

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df, 
            x="group", 
            y="value", 
            ax=ax,
            palette=[box_color, box_color], 
            linewidth=1.5, 
            showfliers=False
        )
        for p in ax.patches:
            p.set_facecolor(box_color)
            p.set_alpha(0.3)
            p.set_edgecolor(box_color)
            p.set_linewidth(1.5)
        for line in ax.lines:
            line.set_color(box_color)
            line.set_alpha(1.0)
            line.set_linewidth(1.2)

        sns.stripplot(
            data=df, x="group", y="value", ax=ax,
            color=box_color, size=3, alpha=0.6, jitter=0.2
        )

        # Title and axis
        # inside ttest_plots loop, after computing welch and paired
        title_N = f"N(custom={welch['n_custom']}, ppmi={welch['n_ppmi']}, paired={paired['n_pairs']})"
        ax.set_title(f"{pretty} — {title_N}", loc="left", pad=12, fontsize=13)

        ax.set_xlabel("")
        ax.set_ylabel(pretty)
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Axis limits with padding
        if not df.empty:
            ymin, ymax = df["value"].min(), df["value"].max()
            rng = ymax - ymin if ymax != ymin else 1.0
            ax.set_ylim(ymin - 0.05 * rng, ymax + 0.20 * rng)

        # Text box contents
        if is_moca:
            # Only show Mann–Whitney and Paired t-test
            text_lines = [
                f"Mann–Whitney p = {mw_p:.3g}" if pd.notna(mw_p) else "Mann–Whitney p = n/a",
                f"Paired t-test p = {paired['p_value']:.3g}" if pd.notna(paired["p_value"]) else "Paired t-test p = n/a",
            ]
        else:
            text_lines = [
                f"Welch p = {welch['p_value']:.3g}" if pd.notna(welch["p_value"]) else "Welch p = n/a",
                f"Paired t-test p = {paired['p_value']:.3g}" if pd.notna(paired['p_value']) else "Paired t-test p = n/a",
            ]

        ax.text(
            0.98, 0.96,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, boxstyle="round,pad=0.2"),
            zorder=10,  # ensures it's on top of gridlines and boxplot
        )

        sns.despine(ax=ax)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{var}_ttest.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return pd.DataFrame(results)




def run_analysis(custom_path: Path, ppmi_path: Path, output_dir: Path) -> None:
    groups = load_groups(custom_path, ppmi_path)

    # Radar plots for MoCA subscores (pre and post)
    radar_dir = output_dir / "radar_plots"
    plot_moca_radar(
        groups,
        radar_dir / "moca_sum_pre_radar",
        score_suffix="_sum_pre",
        title="MoCA Baseline Subscores (Radar)",
    )
    plot_moca_radar(
        groups,
        radar_dir / "moca_sum_post_radar",
        score_suffix="_sum_post",
        title="MoCA Postoperative Subscores (Radar)",
    )
    print(f"Saved MoCA radar plots to {radar_dir}")

    # Variables for t-tests (add MoCA post if present; support the 'um' typo too)
    variables = ["AGE_AT_OP", "AGE_AT_BASELINE", "TimeSinceDiag", "UPDRS_on", "MoCA_sum_pre"]
    for candidate in ("MoCA_sum_post", "MoCA_um_post"):
        if candidate in groups.custom.columns and candidate in groups.ppmi.columns:
            variables.append(candidate)
            COVARIATE_LABELS[candidate] = "MoCA Total Postoperative"

    results_df = ttest_plots(groups, variables, output_dir / "ttest_plots")
    results_path = output_dir / "ttest_summary.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Wrote t-test summary to {results_path}")

    # Side-by-side MoCA raincloud (median impute + darker violins + extra top padding + avg distance text)
    raincloud_moca_side_by_side(groups, output_dir / "moca_raincloud")
    print(f"Saved MoCA raincloud plots to {output_dir}/moca_raincloud.svg and .svg")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare matched cohorts and generate summary plots.")
    parser.add_argument("--custom", type=Path, default=Path("matched_custom.csv"))
    parser.add_argument("--ppmi", type=Path, default=Path("eligible_matched_ppmi.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("matched_group_outputs"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_analysis(args.custom, args.ppmi, args.output_dir)
    


if __name__ == "__main__":
    main()
