"""
visualize.py — Generate all 5 benchmark plots from the CSV
Run:  python src/visualize.py
Outputs PNG files to results/plots/
"""
import pathlib, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#e6edf3",  "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",      "text.color": "#c9d1d9",
    "grid.color": "#21262d",       "grid.alpha": 0.8,
    "font.family": "DejaVu Sans",  "font.size": 11,
    "axes.titlesize": 14,          "axes.labelsize": 12,
    "figure.titlesize": 16,        "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d", "legend.labelcolor": "#c9d1d9",
})

COLORS = {
    "vllm":    "#58a6ff",
    "llamacpp": "#3fb950",
}
CMAP_ACCENT = ["#58a6ff", "#3fb950", "#e3b341", "#f78166", "#d2a8ff", "#79c0ff"]

RESULTS_DIR = pathlib.Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
csvs = sorted(RESULTS_DIR.glob("benchmark_matrix_*.csv"))
if not csvs:
    sys.exit("No benchmark_matrix_*.csv found. Run: python src/aggregate_results.py")

df = pd.read_csv(csvs[-1])
df = df[df["status"] == "success"].copy()
for col in ["throughput_rps","cost_per_1k_requests_usd","composite_quality_score",
            "p50_latency_ms","p95_latency_ms","p99_latency_ms","mean_ttft_ms"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Loaded {len(df)} rows from {csvs[-1].name}")
print(f"Engines:  {df['engine'].unique().tolist()}")
print(f"Configs:  {df['config_label'].nunique()}")
print(f"Profiles: {df['load_profile'].unique().tolist()}\n")


# ── Helper ────────────────────────────────────────────────────────────────────
def save(fig, name):
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  OK: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Pareto Frontier: Cost vs Throughput vs Quality
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 1: Pareto Frontier…")
fig, ax = plt.subplots(figsize=(13, 8))
fig.patch.set_facecolor("#0d1117")

for engine, grp in df.groupby("engine"):
    grp_m = grp.groupby("config_label").agg(
        rps =("throughput_rps",           "mean"),
        cost=("cost_per_1k_requests_usd", "mean"),
        qual=("composite_quality_score",  "mean"),
    )
    grp_meta = grp.groupby("config_label").first()[["weight_format_label"]]
    grp_agg  = grp_m.join(grp_meta).rename(columns={"weight_format_label": "label"})
    grp_agg  = grp_agg.dropna(subset=["rps","cost"])

    sc = ax.scatter(
        grp_agg["cost"], grp_agg["rps"],
        s=grp_agg["qual"].fillna(0.5) * 600 + 80,
        c=COLORS.get(engine, "#c9d1d9"),
        alpha=0.85, edgecolors="white", linewidths=0.8,
        label=engine, zorder=4,
    )
    for _, row in grp_agg.iterrows():
        ax.annotate(row["label"], (row["cost"], row["rps"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8.5, color="#e6edf3")

ax.set_xlabel("Cost per 1K Requests (USD)")
ax.set_ylabel("Throughput (Requests/sec)")
ax.set_title("Pareto Frontier — Cost vs Throughput\n(bubble size = composite quality score)")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=11)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.4f"))
plt.tight_layout()
save(fig, "pareto_frontier.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Latency Breakdown: P50 / P95 / P99 + TTFT
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 2: Latency Breakdown…")
lat_m = df.groupby("config_label").agg(
    p50      =("p50_latency_ms", "mean"),
    p95      =("p95_latency_ms", "mean"),
    p99      =("p99_latency_ms", "mean"),
    mean_ttft=("mean_ttft_ms",   "mean"),
)
lat_meta = df.groupby("config_label").first()[["engine", "weight_format_label"]]
lat = lat_m.join(lat_meta).rename(columns={"weight_format_label": "label"})
lat = lat.dropna(subset=["p50"]).sort_values("p95")

x = np.arange(len(lat))
w = 0.26
fig, ax = plt.subplots(figsize=(max(10, len(lat)*4), 7))
fig.patch.set_facecolor("#0d1117")

b50 = ax.bar(x-w, lat["p50"], w, label="P50", color="#3fb950", alpha=0.85, zorder=3)
b95 = ax.bar(x,   lat["p95"], w, label="P95", color="#e3b341", alpha=0.85, zorder=3)
b99 = ax.bar(x+w, lat["p99"], w, label="P99", color="#f78166", alpha=0.85, zorder=3)

for i, (a, b, c) in enumerate(zip(b50, b95, b99)):
    edge = COLORS.get(lat["engine"].values[i], "#8b949e")
    for bar in [a, b, c]:
        bar.set_edgecolor(edge); bar.set_linewidth(1.5)

ax2 = ax.twinx()
ttft = lat["mean_ttft"].fillna(0).values
ax2.plot(x, ttft, "o--", color="#58a6ff", lw=2, ms=6, label="Avg TTFT", zorder=6)
ax2.set_ylabel("TTFT (ms)", color="#58a6ff")
ax2.tick_params(axis="y", colors="#58a6ff")
ax2.set_ylim(0, max(ttft.max(), 1) * 2.2)

ax.set_xlabel("Configuration")
ax.set_ylabel("Latency (ms)")
ax.set_title("Latency Breakdown: P50 / P95 / P99 + Mean TTFT")
ax.set_xticks(x)
ax.set_xticklabels(lat["label"].values, rotation=20, ha="right", fontsize=10)
ax.grid(True, alpha=0.25, axis="y")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
save(fig, "latency_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Load Profile Comparison: Throughput across profiles
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 3: Profile Comparison…")
prof_df = df.pivot_table(
    index="load_profile", columns="engine",
    values="throughput_rps", aggfunc="mean"
).reindex(["short_qa","medium_reasoning","long_context","burst_spike"])
prof_df.dropna(how="all", inplace=True)

n_engines = len(prof_df.columns)
n_profiles = len(prof_df)
x = np.arange(n_profiles)
w = 0.35
fig, ax = plt.subplots(figsize=(max(10, n_profiles*3), 6))
fig.patch.set_facecolor("#0d1117")

for j, engine in enumerate(prof_df.columns):
    offset = (j - n_engines/2 + 0.5) * w
    bars = ax.bar(x + offset, prof_df[engine].fillna(0), w,
                  label=engine, color=COLORS.get(engine, CMAP_ACCENT[j]),
                  alpha=0.85, zorder=3)
    for bar, val in zip(bars, prof_df[engine].fillna(0)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#c9d1d9")

ax.set_xlabel("Load Profile")
ax.set_ylabel("Mean Throughput (RPS)")
ax.set_title("Throughput by Load Profile & Engine")
ax.set_xticks(x)
ax.set_xticklabels(prof_df.index, fontsize=10)
ax.legend()
ax.grid(True, alpha=0.25, axis="y")
plt.tight_layout()
save(fig, "profile_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Cost vs Quality scatter (per engine, per profile)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 4: Cost vs Quality…")
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0d1117")

for engine, grp in df.groupby("engine"):
    color = COLORS.get(engine, "#c9d1d9")
    for _, row in grp.iterrows():
        if pd.isna(row.get("cost_per_1k_requests_usd")) or pd.isna(row.get("composite_quality_score")):
            continue
        ax.scatter(row["cost_per_1k_requests_usd"], row["composite_quality_score"],
                   c=color, s=100, alpha=0.7, edgecolors="white", lw=0.5, zorder=3)
    # Engine label
    g_mean = grp[["cost_per_1k_requests_usd","composite_quality_score"]].mean()
    ax.scatter([], [], c=color, s=80, label=f"{engine}", alpha=0.9)

ax.set_xlabel("Cost per 1K Requests (USD)")
ax.set_ylabel("Composite Quality Score (0–1)")
ax.set_title("Cost vs Quality — Each point is one load profile run")
ax.legend()
ax.grid(True, alpha=0.25)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.4f"))
plt.tight_layout()
save(fig, "cost_vs_quality.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Summary Table
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 5: Summary Table…")
sum_m = df.groupby("config_label").agg(
    rps        =("throughput_rps",           "mean"),
    p50_ms     =("p50_latency_ms",           "mean"),
    p95_ms     =("p95_latency_ms",           "mean"),
    ttft_ms    =("mean_ttft_ms",             "mean"),
    cost_per_1k=("cost_per_1k_requests_usd", "mean"),
    quality    =("composite_quality_score",  "mean"),
    json_acc   =("json_accuracy_pct",        "mean"),
    error_pct  =("error_rate_pct",           "mean"),
)
sum_meta = df.groupby("config_label").first()[["engine", "weight_format_label"]]
summary  = sum_m.join(sum_meta).reset_index().sort_values("cost_per_1k")

min_rps = summary["rps"].min() or 1.0
summary["speedup"] = (summary["rps"] / min_rps).round(2)

col_labels = ["Config","Engine","Weight Format","RPS","Speedup",
              "P50 (ms)","P95 (ms)","TTFT (ms)","$/1K Req","Quality","JSON Acc%","Err%"]

def fmt(v, spec):
    return spec.format(v) if pd.notna(v) else "N/A"

cell_data = []
for _, row in summary.iterrows():
    cell_data.append([
        str(row["config_label"])[:30],
        str(row["engine"]),
        str(row["weight_format_label"]),
        fmt(row["rps"],        "{:.2f}"),
        fmt(row["speedup"],    "{:.2f}x"),
        fmt(row["p50_ms"],     "{:.0f}"),
        fmt(row["p95_ms"],     "{:.0f}"),
        fmt(row["ttft_ms"],    "{:.0f}"),
        fmt(row["cost_per_1k"],"${:.5f}"),
        fmt(row["quality"],    "{:.3f}"),
        fmt(row["json_acc"],   "{:.1f}%"),
        fmt(row["error_pct"],  "{:.1f}%"),
    ])

fig, ax = plt.subplots(figsize=(24, max(3, len(summary)*0.8 + 1.5)))
fig.patch.set_facecolor("#0d1117")
ax.axis("off")

tbl = ax.table(cellText=cell_data, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#161b22" if r == 0 else "#0d1117")
    cell.set_text_props(color="#e6edf3" if r == 0 else "#c9d1d9")
    cell.set_edgecolor("#30363d")

ax.set_title("Cost / Benefit Summary — All Configurations", fontsize=14, pad=16, color="#e6edf3")
plt.tight_layout()
save(fig, "summary_table.png")


print(f"\n✅ All 5 plots saved to: {PLOTS_DIR.resolve()}")
print("   Open results/plots/ to view them.")
