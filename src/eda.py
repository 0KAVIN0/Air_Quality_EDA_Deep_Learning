"""
Exploratory Data Analysis and Visualization Module.
Generates publication-quality charts, descriptive tables,
and an interactive Plotly HTML visualization.

Student: Kavin M (24AD053)
Institution: KPR Institute of Engineering and Technology
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import load_raw_dataset, inspect_data_quality
from src.preprocessing import clean_air_quality_dataframe
from src.utils import get_project_root, ensure_directories

# Configure Matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def generate_descriptive_statistics_table(df_clean: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Generate and save formal descriptive statistical summary of numeric variables."""
    numeric_cols = [
        "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
        "T", "RH", "AH"
    ]
    sub_df = df_clean[numeric_cols]
    desc = sub_df.describe().T
    desc["median"] = sub_df.median()
    desc["IQR"] = desc["75%"] - desc["25%"]
    desc["skewness"] = sub_df.skew()
    desc["kurtosis"] = sub_df.kurt()

    desc = desc[[
        "count", "mean", "std", "min", "25%", "median", "75%", "max", "IQR", "skewness", "kurtosis"
    ]]
    desc = desc.round(2)
    desc.index.name = "Variable"
    csv_path = tables_dir / "descriptive_statistics.csv"
    desc.to_csv(csv_path)
    print(f"[EDA] Descriptive statistics table saved to: {csv_path}")
    return desc


def plot_01_missing_values(df_raw: pd.DataFrame, out_path: Path):
    """1. Missing Value Percentage Bar Chart across all original attributes."""
    cols = [c for c in df_raw.columns if c not in ["Date", "Time", "Unnamed: 15", "Unnamed: 16"]]
    total = len(df_raw[df_raw["Date"].notnull()])
    missing_pcts = {}
    for c in cols:
        cnt = (df_raw[c] == -200).sum() + df_raw[c].isnull().sum()
        missing_pcts[c] = (cnt / total) * 100

    s = pd.Series(missing_pcts).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d9534f" if v > 50 else ("#f0ad4e" if v > 10 else "#0275d8") for v in s.values]
    bars = ax.barh(s.index, s.values, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Missing Data Percentage (%)")
    ax.set_ylabel("Original Attribute Name")
    ax.set_title("Figure 1: Missing Value Percentage by Attribute (-200 Sentinel & Nulls)", fontweight="bold")
    ax.set_xlim(0, 100)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%", xy=(w + 1, bar.get_y() + bar.get_height() / 2),
                    va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[EDA] Saved Figure 1 to: {out_path}")


def plot_02_target_distribution(df_clean: pd.DataFrame, out_path: Path):
    """2. CO(GT) Distribution with KDE and Mean/Median annotations."""
    co = df_clean["CO(GT)"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(co, kde=True, color="#2b5c8f", bins=40, ax=ax, edgecolor="black", alpha=0.6)
    mean_val = co.mean()
    median_val = co.median()

    ax.axvline(mean_val, color="#d9534f", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f} mg/m³")
    ax.axvline(median_val, color="#5cb85c", linestyle="-.", linewidth=2, label=f"Median: {median_val:.2f} mg/m³")

    skew = co.skew()
    ax.text(0.70, 0.75, f"Skewness: {skew:.2f}\nSpread (IQR): {co.quantile(0.75)-co.quantile(0.25):.2f}\nMin: {co.min():.1f} | Max: {co.max():.1f}",
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc"),
            fontsize=10)

    ax.set_xlabel("Ground-Truth Carbon Monoxide Concentration CO(GT) [mg/m³]")
    ax.set_ylabel("Hourly Frequency (Count)")
    ax.set_title("Figure 2: Distribution of Measured CO(GT) Concentration", fontweight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[EDA] Saved Figure 2 to: {out_path}")


def plot_03_sensor_distributions(df_clean: pd.DataFrame, out_path: Path):
    """3. Multi-panel distribution plots of sensor responses and weather variables."""
    features = [
        ("PT08.S1(CO)", "Tin Oxide Sensor (CO) [mV]", "#1f77b4"),
        ("PT08.S2(NMHC)", "Titania Sensor (NMHC) [mV]", "#ff7f0e"),
        ("PT08.S4(NO2)", "Tungsten Oxide Sensor (NO2) [mV]", "#2ca02c"),
        ("T", "Ambient Temperature [°C]", "#d62728"),
        ("RH", "Relative Humidity [%]", "#9467bd"),
        ("AH", "Absolute Humidity [g/m³]", "#8c564b")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (col, label, color) in enumerate(features):
        data = df_clean[col].dropna()
        sns.histplot(data, kde=True, color=color, ax=axes[i], bins=35, edgecolor="black", alpha=0.6)
        axes[i].set_title(col, fontweight="bold", fontsize=11)
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Count")

    plt.suptitle("Figure 3: Histograms and Continuous KDE Distributions for Key Predictors", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[EDA] Saved Figure 3 to: {out_path}")


def plot_04_boxplots_outliers(df_clean: pd.DataFrame, out_path: Path):
    """4. Normalized boxplots comparing sensor responses, weather variables, and target."""
    cols_to_plot = ["CO(GT)", "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH"]
    subset = df_clean[cols_to_plot].dropna()

    # Min-max scale each to [0, 1] purely for simultaneous boxplot comparison visualization
    scaled_sub = (subset - subset.min()) / (subset.max() - subset.min())

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=scaled_sub, ax=ax, palette="Set2", fliersize=3)
    ax.set_ylabel("Normalized Range [0, 1]")
    ax.set_xlabel("Monitored Variable")
    ax.set_xticklabels(cols_to_plot, rotation=25, ha="right")
    ax.set_title("Figure 4: Normalized Boxplots Identifying Spread and Potential Outliers", fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[EDA] Saved Figure 4 to: {out_path}")


def plot_05_correlation_heatmap(df_clean: pd.DataFrame, out_path: Path, tables_dir: Path):
    """5. Correlation heatmap across sensor telemetry, weather, and ground truth."""
    numeric_cols = [
        "CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)",
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
        "T", "RH", "AH"
    ]
    corr_df = df_clean[numeric_cols].corr(method="pearson")
    corr_df.round(3).to_csv(tables_dir / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=True,
                fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Figure 5: Pearson Correlation Heatmap (Ground Truth and Sensors)", fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[EDA] Saved Figure 5 to: {out_path}")


def plot_06_time_series(df_clean: pd.DataFrame, out_path: Path):
    """6. Chronological time series plot of CO(GT) with rolling trend."""
    df_ts = df_clean.dropna(subset=["CO(GT)"]).set_index("DateTime").sort_index()
    rolling_24h = df_ts["CO(GT)"].rolling(window=24, min_periods=12).mean()

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(df_ts.index, df_ts["CO(GT)"], color="#a6c8e0", alpha=0.45, label="Hourly Observed CO(GT)")
    ax.plot(rolling_24h.index, rolling_24h, color="#0b3c5d", linewidth=2.0, label="24-Hour Rolling Mean Trend")

    ax.set_ylabel("CO(GT) Concentration [mg/m³]")
    ax.set_xlabel("Monitoring Date (March 2004 – April 2005)")
    ax.set_title("Figure 6: Longitudinal Time-Series of Carbon Monoxide Concentration", fontweight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[EDA] Saved Figure 6 to: {out_path}")


def plot_07_bivariate_scatter(df_clean: pd.DataFrame, out_path: Path):
    """7. Bivariate scatterplot between CO(GT) and PT08.S1(CO) sensor response."""
    valid_sub = df_clean.dropna(subset=["CO(GT)", "PT08.S1(CO)"])

    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.regplot(
        data=valid_sub, x="PT08.S1(CO)", y="CO(GT)",
        scatter_kws={"alpha": 0.25, "color": "#2c7bb6", "s": 15},
        line_kws={"color": "#d7191c", "linewidth": 2.5},
        ax=ax
    )
    r = valid_sub["PT08.S1(CO)"].corr(valid_sub["CO(GT)"])
    ax.text(0.05, 0.90, f"Pearson r: {r:.2f}\nR² = {r**2:.2f}",
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc"),
            fontsize=11, fontweight="bold")

    ax.set_xlabel("PT08.S1(CO) Chemical Sensor Nominal Resistance Response [mV]")
    ax.set_ylabel("Ground-Truth CO(GT) [mg/m³]")
    ax.set_title("Figure 7: Bivariate Relationship Between Primary CO Sensor and Actual CO", fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[EDA] Saved Figure 7 to: {out_path}")


def plot_08_multivariate_relationship(df_clean: pd.DataFrame, out_path: Path):
    """8. Multivariate visualization: Hexbin plot of PT08.S1(CO) vs PT08.S2(NMHC) colored by CO(GT)."""
    valid_sub = df_clean.dropna(subset=["PT08.S1(CO)", "PT08.S2(NMHC)", "CO(GT)"])

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sc = ax.scatter(
        valid_sub["PT08.S1(CO)"],
        valid_sub["PT08.S2(NMHC)"],
        c=valid_sub["CO(GT)"],
        cmap="viridis",
        alpha=0.6,
        s=18,
        edgecolors="none"
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("CO(GT) [mg/m³]", fontsize=11)

    ax.set_xlabel("PT08.S1(CO) Sensor Response [mV]")
    ax.set_ylabel("PT08.S2(NMHC) Sensor Response [mV]")
    ax.set_title("Figure 8: Multivariate Interaction: PT08.S1(CO) vs PT08.S2(NMHC) vs Target CO(GT)", fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[EDA] Saved Figure 8 to: {out_path}")


def plot_09_diurnal_hourly_pattern(df_clean: pd.DataFrame, out_path: Path):
    """9. Diurnal rush-hour cycle: Average CO(GT) and sensor responses by hour of day."""
    df_clean = df_clean.copy()
    df_clean["hour"] = df_clean["DateTime"].dt.hour
    hourly_grp = df_clean.groupby("hour")[["CO(GT)", "PT08.S1(CO)"]].agg(["mean", "std"])

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    hours = np.arange(24)

    co_mean = hourly_grp["CO(GT)"]["mean"]
    co_std = hourly_grp["CO(GT)"]["std"]
    sensor_mean = hourly_grp["PT08.S1(CO)"]["mean"]

    color1 = "#d9534f"
    ax1.plot(hours, co_mean, color=color1, marker="o", linewidth=2.5, label="Mean CO(GT) [mg/m³]")
    ax1.fill_between(hours, co_mean - 0.5 * co_std, co_mean + 0.5 * co_std, color=color1, alpha=0.15)
    ax1.set_xlabel("Hour of Day (00:00 - 23:00 UTC)")
    ax1.set_ylabel("CO(GT) Concentration [mg/m³]", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(hours)

    ax2 = ax1.twinx()
    color2 = "#0275d8"
    ax2.plot(hours, sensor_mean, color=color2, marker="s", linestyle="--", linewidth=2.0, label="PT08.S1(CO) Sensor [mV]")
    ax2.set_ylabel("PT08.S1(CO) Sensor Response [mV]", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    plt.title("Figure 9: Diurnal Hourly Cycle Revealing Morning & Evening Commuter Rush Hour Peaks", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[EDA] Saved Figure 9 to: {out_path}")


def generate_interactive_plotly(df_clean: pd.DataFrame, out_path: Path):
    """
    Generate an interactive Plotly HTML chart (Stage 7 of Visualization: Interact).
    Features dual-axis time series, hover info, rangeslider, and zoom/pan.
    """
    df_sub = df_clean.dropna(subset=["CO(GT)"]).sort_values("DateTime")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # CO(GT) trace
    fig.add_trace(
        go.Scatter(
            x=df_sub["DateTime"],
            y=df_sub["CO(GT)"],
            name="Ground Truth CO(GT) [mg/m³]",
            mode="lines",
            line=dict(color="#d9534f", width=1.5),
            hovertemplate="<b>Date/Time:</b> %{x}<br><b>CO(GT):</b> %{y:.2f} mg/m³<extra></extra>"
        ),
        secondary_y=False
    )

    # PT08.S1(CO) sensor trace
    fig.add_trace(
        go.Scatter(
            x=df_sub["DateTime"],
            y=df_sub["PT08.S1(CO)"],
            name="Sensor PT08.S1(CO) [mV]",
            mode="lines",
            line=dict(color="#0275d8", width=1.2, dash="dot"),
            opacity=0.7,
            hovertemplate="<b>Date/Time:</b> %{x}<br><b>PT08.S1(CO):</b> %{y:.1f} mV<extra></extra>"
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="<b>Interactive Air Quality Monitoring: Measured CO(GT) vs PT08.S1(CO) Sensor Telemetry</b><br><sup>The Seven Stages of Visualization: Stage 7 (Interact) — Zoom, Pan, and Inspect Hover Values</sup>",
        xaxis_title="Date and Time",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.8)"),
        hovermode="x unified",
        template="plotly_white",
        height=600
    )
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_yaxes(title_text="CO(GT) Concentration [mg/m³]", secondary_y=False)
    fig.update_yaxes(title_text="Sensor PT08.S1(CO) Response [mV]", secondary_y=True)

    fig.write_html(str(out_path))
    print(f"[EDA] Interactive Plotly visualization saved to: {out_path}")


def run_full_eda():
    """Execute complete EDA pipeline and generate all artifacts."""
    root = get_project_root()
    dirs = ensure_directories(root)

    raw_df = load_raw_dataset()
    clean_df = clean_air_quality_dataframe(raw_df)

    # Descriptive statistics
    generate_descriptive_statistics_table(clean_df, dirs["tables"])

    # Figure outputs
    fig_dir = dirs["figures_eda"]
    plot_01_missing_values(raw_df, fig_dir / "fig01_missing_values.png")
    plot_02_target_distribution(clean_df, fig_dir / "fig02_target_distribution.png")
    plot_03_sensor_distributions(clean_df, fig_dir / "fig03_sensor_distributions.png")
    plot_04_boxplots_outliers(clean_df, fig_dir / "fig04_boxplots_outliers.png")
    plot_05_correlation_heatmap(clean_df, fig_dir / "fig05_correlation_heatmap.png", dirs["tables"])
    plot_06_time_series(clean_df, fig_dir / "fig06_time_series.png")
    plot_07_bivariate_scatter(clean_df, fig_dir / "fig07_bivariate_scatter.png")
    plot_08_multivariate_relationship(clean_df, fig_dir / "fig08_multivariate_scatter.png")
    plot_09_diurnal_hourly_pattern(clean_df, fig_dir / "fig09_diurnal_hourly_profile.png")

    # Interactive Stage 7 visual
    generate_interactive_plotly(clean_df, dirs["interactive"] / "interactive_air_quality.html")
    print("[EDA] All 9 publication figures and interactive HTML successfully created.")


if __name__ == "__main__":
    run_full_eda()
