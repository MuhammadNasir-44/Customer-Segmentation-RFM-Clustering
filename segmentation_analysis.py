"""
Customer Segmentation — RFM + K-Means clustering
================================================

End-to-end unsupervised-learning project: group an online retailer's customers
into distinct behavioural segments, so marketing can treat each group
differently (reward the loyal, win back the lapsing, nurture the new).

Pipeline
--------
1. Load & clean the raw transaction data
2. Exploratory analysis (what the sales data looks like)
3. Feature engineering — build an RFM table (Recency, Frequency, Monetary)
   per customer
4. Choose k — elbow (inertia) and silhouette score
5. Cluster with K-Means on scaled RFM features
6. Profile & name each segment, and save charts

Run:  python segmentation_analysis.py
Outputs a segment summary to the console and saves charts to ./images/.

Author: Muhammad Nasiruddin
Dataset: UCI "Online Retail" (public) — a UK gift-ware e-commerce store.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render charts to file without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "Online Retail.xlsx"
IMG_DIR = Path(__file__).parent / "images"
GREEN, BLUE, GREY = "#2a9d8f", "#264653", "#adb5bd"
PALETTE = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#8ab17d"]

RANDOM_STATE = 42
N_CLUSTERS = 4  # justified by the elbow + silhouette analysis below


def load_and_clean() -> pd.DataFrame:
    """Load the raw transactions and fix the known data-quality issues."""
    df = pd.read_excel(DATA_PATH)

    # Cancelled orders have an InvoiceNo starting with 'C' — drop them.
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # We can only build RFM for known customers.
    df = df.dropna(subset=["CustomerID"])

    # Remove non-sales / bad rows: zero or negative quantity or price.
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def explore(df: pd.DataFrame) -> None:
    """Print headline facts about the cleaned transaction data."""
    print(f"Transactions: {len(df):,}")
    print(f"Unique customers: {df['CustomerID'].nunique():,}")
    print(f"Date range: {df['InvoiceDate'].min():%Y-%m-%d} "
          f"to {df['InvoiceDate'].max():%Y-%m-%d}")
    print(f"Total revenue: £{df['TotalPrice'].sum():,.0f}\n")

    # Revenue by country (excluding the dominant UK) — shows the customer base.
    by_country = (
        df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
    )
    top = by_country.drop("United Kingdom", errors="ignore").head(8)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    top.iloc[::-1].plot(kind="barh", color=BLUE, ax=ax)
    ax.set_xlabel("Revenue (£)")
    ax.set_title("Top export markets (excl. UK)")
    ax.xaxis.set_major_formatter(lambda x, _: f"£{x/1000:.0f}k")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "revenue_by_country.png", dpi=110)
    plt.close(fig)


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the transaction log into one row per customer with R, F, M."""
    # "Today" = day after the last transaction, so recency is never negative.
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )

    print("RFM table (per customer):")
    print(rfm.describe().round(1).to_string(), "\n")
    return rfm


def rfm_quantile_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Classic rule-based RFM: score each customer 1–5 on R, F, and M.

    This is the traditional (non-ML) segmentation method. We compute it
    alongside the K-Means clusters so the two approaches can be compared —
    K-Means finds groups from the data, while this applies fixed business
    rules (quintiles). A combined "RFM score" of 555 is the best customer.
    """
    scores = pd.DataFrame(index=rfm.index)

    # Recency is reversed: a *smaller* gap since last order is better, so the
    # most-recent quintile earns the top score of 5.
    scores["R"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # Frequency has many ties at 1–2 orders, so rank first to break them.
    scores["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5,
                          labels=[1, 2, 3, 4, 5]).astype(int)
    scores["M"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    scores["RFM_Score"] = scores["R"] + scores["F"] + scores["M"]
    scores["RFM_Cell"] = (scores["R"].astype(str) + scores["F"].astype(str)
                          + scores["M"].astype(str))

    print("Rule-based RFM scores (1–5 per dimension, 3–15 combined):")
    print(scores[["R", "F", "M", "RFM_Score"]].describe().round(1).to_string(), "\n")
    print(f"  'Best' customers (RFM_Score >= 13): "
          f"{(scores['RFM_Score'] >= 13).sum():,}")
    print(f"  'Lost' customers (RFM_Score <= 5):  "
          f"{(scores['RFM_Score'] <= 5).sum():,}\n")
    return scores


def choose_k(X_scaled: np.ndarray, k_range=range(2, 9)) -> None:
    """Save an elbow (inertia) + silhouette chart to justify the k we pick."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    print("Choosing k:")
    for k, inertia, sil in zip(k_range, inertias, silhouettes):
        print(f"  k={k}  inertia={inertia:9.0f}  silhouette={sil:.3f}")
    print()

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(list(k_range), inertias, "o-", color=BLUE, label="Inertia (elbow)")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia", color=BLUE)
    ax1.axvline(N_CLUSTERS, color=GREY, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(list(k_range), silhouettes, "s-", color=GREEN, label="Silhouette")
    ax2.set_ylabel("Silhouette score", color=GREEN)

    ax1.set_title(f"Choosing k  (chose k={N_CLUSTERS})")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "choosing_k.png", dpi=110)
    plt.close(fig)


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a plain-English name from its RFM profile."""
    names: dict[int, str] = {}
    r_median = profile["Recency"].median()
    m_rank = profile["Monetary"].rank(ascending=False)

    for cluster, row in profile.iterrows():
        # Best cluster overall: highest spend + most frequent.
        if m_rank[cluster] == 1 and row["Frequency"] >= profile["Frequency"].median():
            names[cluster] = "Champions"
        elif row["Recency"] > r_median and row["Frequency"] <= profile["Frequency"].median():
            names[cluster] = "At-risk / Lapsing"
        elif row["Recency"] <= r_median and row["Frequency"] <= profile["Frequency"].median():
            names[cluster] = "New / Promising"
        else:
            names[cluster] = "Loyal regulars"
    return names


def plot_pca(X_scaled: np.ndarray, labels: np.ndarray, names: dict[int, str]) -> None:
    """Project the 3-D scaled RFM space down to 2-D with PCA, to *see* the
    clusters. PCA finds the two directions that capture the most variance, so
    well-separated clusters show up as visually distinct blobs."""
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cluster in sorted(np.unique(labels)):
        m = labels == cluster
        ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.5,
                   color=PALETTE[cluster % len(PALETTE)], label=names[cluster])
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Clusters in 2-D (PCA projection)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "pca_clusters.png", dpi=110)
    plt.close(fig)


def plot_snake(rfm_log_scaled: pd.DataFrame, names: dict[int, str]) -> None:
    """Snake plot — the classic RFM segmentation chart. Shows each segment's
    average *standardised* R, F, M as a line, so you can read a segment's
    'shape' at a glance (e.g. Champions: low recency, high frequency & spend)."""
    melted = rfm_log_scaled.copy()
    melted["Segment"] = melted["Cluster"].map(names)
    means = melted.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (segment, row) in enumerate(means.iterrows()):
        ax.plot(["Recency", "Frequency", "Monetary"], row.values, "o-",
                color=PALETTE[i % len(PALETTE)], label=segment)
    ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Average standardised value")
    ax.set_title("Snake plot — RFM profile of each segment")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "snake_plot.png", dpi=110)
    plt.close(fig)


def compare_models(X_scaled: np.ndarray, k: int) -> None:
    """Sanity-check the choice of K-Means against a different algorithm.

    Agglomerative (hierarchical) clustering builds groups a completely
    different way — merging nearest points bottom-up rather than moving
    centroids. If both find similarly clean groups at the same k, that's
    evidence the segments are real structure in the data, not an artefact
    of one algorithm.
    """
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    agg = AgglomerativeClustering(n_clusters=k)
    km_sil = silhouette_score(X_scaled, km.fit_predict(X_scaled))
    agg_sil = silhouette_score(X_scaled, agg.fit_predict(X_scaled))

    print(f"Model comparison at k={k} (silhouette, higher is better):")
    print(f"  K-Means                    {km_sil:.3f}")
    print(f"  Agglomerative (Ward)       {agg_sil:.3f}")
    winner = "K-Means" if km_sil >= agg_sil else "Agglomerative"
    print(f"  -> {winner} wins; K-Means also scales far better, so we keep it.\n")


def plot_revenue_share(rfm: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """How much of total revenue does each segment actually drive?

    Customer *counts* and *revenue* are very different stories — a small
    segment can carry a large share of the money. This is the classic
    "80/20" (Pareto) lens that turns segments into budget decisions.
    """
    rev = rfm.groupby("Cluster")["Monetary"].sum()
    share = (rev / rev.sum()).sort_values(ascending=False)
    share.index = share.index.map(names)

    print("Revenue share by segment:")
    for segment, pct in share.items():
        print(f"  {segment:<20} {pct:6.1%}")
    print()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    share.iloc[::-1].plot(kind="barh", color=PALETTE[: len(share)], ax=ax)
    ax.set_xlabel("Share of total revenue")
    ax.set_title("Which segments drive the revenue?")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "revenue_share.png", dpi=110)
    plt.close(fig)
    return share


def main() -> None:
    df = load_and_clean()
    explore(df)

    rfm = build_rfm(df)

    # Rule-based RFM scoring (traditional method) alongside the ML clustering.
    scores = rfm_quantile_scores(rfm)
    rfm = rfm.join(scores)

    # RFM is heavily right-skewed (a few huge spenders). Log-transform before
    # scaling so K-Means (a distance method) isn't dominated by outliers.
    # Only the three raw RFM columns feed the clustering — not the 1–5 scores.
    rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_log)

    choose_k(X_scaled)
    compare_models(X_scaled, N_CLUSTERS)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    rfm["Cluster"] = km.fit_predict(X_scaled)

    # Profile each cluster on the ORIGINAL (interpretable) RFM values.
    profile = rfm.groupby("Cluster").agg(
        Recency=("Recency", "median"),
        Frequency=("Frequency", "median"),
        Monetary=("Monetary", "median"),
        Customers=("Recency", "size"),
    )
    names = name_segments(profile)
    profile["Segment"] = profile.index.map(names)

    print("Segment profiles (median values):")
    print(profile.to_string(), "\n")

    sil = silhouette_score(X_scaled, rfm["Cluster"])
    print(f"Final silhouette score (k={N_CLUSTERS}): {sil:.3f}\n")

    # PCA projection (see the clusters) and snake plot (segment RFM shapes).
    plot_pca(X_scaled, rfm["Cluster"].to_numpy(), names)
    scaled_df = pd.DataFrame(X_scaled, columns=["Recency", "Frequency", "Monetary"],
                             index=rfm.index)
    scaled_df["Cluster"] = rfm["Cluster"].to_numpy()
    plot_snake(scaled_df, names)

    # Revenue concentration — small segments can carry most of the money.
    plot_revenue_share(rfm, names)

    # --- Chart 1: segment sizes ---
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sizes = profile.set_index("Segment")["Customers"].sort_values()
    sizes.plot(kind="barh", color=PALETTE[: len(sizes)], ax=ax)
    ax.set_xlabel("Number of customers")
    ax.set_title("How many customers in each segment")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "segment_sizes.png", dpi=110)
    plt.close(fig)

    # --- Chart 2: RFM fingerprint of each segment (normalised bars) ---
    norm = profile[["Recency", "Frequency", "Monetary"]].copy()
    # Invert recency so "higher = better" for all three bars.
    norm["Recency"] = norm["Recency"].max() - norm["Recency"]
    norm = norm / norm.max()
    norm.index = profile["Segment"]
    fig, ax = plt.subplots(figsize=(7, 4))
    norm.plot(kind="bar", color=[GREEN, BLUE, "#e9c46a"], ax=ax)
    ax.set_ylabel("Relative score (0–1)")
    ax.set_title("RFM fingerprint by segment\n(Recency inverted: taller = more recent)")
    ax.legend(title="", loc="upper right")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "segment_fingerprint.png", dpi=110)
    plt.close(fig)

    # --- Chart 3: scatter of Frequency vs Monetary, coloured by segment ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for cluster in sorted(rfm["Cluster"].unique()):
        sub = rfm[rfm["Cluster"] == cluster]
        ax.scatter(
            sub["Frequency"], sub["Monetary"],
            s=12, alpha=0.5, color=PALETTE[cluster % len(PALETTE)],
            label=names[cluster],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (orders)")
    ax.set_ylabel("Monetary (£ spent)")
    ax.set_title("Customers by segment (log scale)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "segment_scatter.png", dpi=110)
    plt.close(fig)

    # Save the labelled customer table for reference.
    rfm.to_csv(Path(__file__).parent / "customer_segments.csv")
    print("Saved per-customer segments to customer_segments.csv")


if __name__ == "__main__":
    main()
