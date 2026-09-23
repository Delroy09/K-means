import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path
import time

st.set_page_config(page_title="K-Means Segmentation", page_icon="K", layout="centered")

DATASETS_DIR = Path(__file__).parent / "datasets"
COLORS = px.colors.qualitative.Plotly


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def kmeans_steps(X_scaled, k, n_steps=12):
    """Run K-Means one iteration at a time, yielding labels + centroids."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_scaled), size=k, replace=False)
    centroids = X_scaled[idx].copy()

    for _ in range(n_steps):
        dists = np.linalg.norm(X_scaled[:, None] - centroids[None, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = np.array([
            X_scaled[labels == c].mean(axis=0) if (labels == c).any() else centroids[c]
            for c in range(k)
        ])
        centroids = new_centroids
        yield labels, centroids.copy()


def build_frame(df, x_col, y_col, labels, centroids, title, scaler_mean, scaler_scale):
    """Build a Plotly figure showing points coloured by cluster + centroid markers."""
    cx = centroids[:, 0] * scaler_scale[0] + scaler_mean[0]
    cy = centroids[:, 1] * scaler_scale[1] + scaler_mean[1]

    fig = go.Figure()
    for c in range(centroids.shape[0]):
        mask = labels == c
        subset = df[mask]
        fig.add_trace(go.Scatter(
            x=subset[x_col], y=subset[y_col],
            mode="markers",
            marker=dict(size=7, color=COLORS[c % len(COLORS)], opacity=0.75,
                        line=dict(width=0.4, color="#222")),
            name=f"Cluster {c}",
            showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers",
        marker=dict(size=16, color="white", symbol="x",
                    line=dict(width=2, color="#333")),
        name="Centroids",
    ))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=36, b=0), height=420,
        title=title, xaxis_title=x_col, yaxis_title=y_col,
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    source = st.radio("Data source", ["Built-in dataset", "Upload CSV"], horizontal=True)

    if source == "Built-in dataset":
        dataset_name = st.selectbox("Dataset", ["Customer Spending", "Student Marks"])
    else:
        dataset_name = None

    k = st.slider("Clusters (K)", 2, 6, 2)
    speed = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"], value="Normal")
    st.divider()
    st.caption("Streamlit / scikit-learn / Plotly")

SPEED_MAP = {"Slow": 0.9, "Normal": 0.5, "Fast": 0.2}
delay = SPEED_MAP[speed]

# ── Load Data ────────────────────────────────────────────────────────────

BUILTIN = {
    "Customer Spending": {
        "file": "customer_spending.csv",
        "features": ["Age", "AnnualIncome", "SpendingScore"],
        "id_col": "CustomerID",
    },
    "Student Marks": {
        "file": "student_marks.csv",
        "features": ["Math", "Science", "English"],
        "id_col": "StudentID",
    },
}

if source == "Built-in dataset":
    cfg = BUILTIN[dataset_name]
    file_path = DATASETS_DIR / cfg["file"]
    if not file_path.exists():
        st.error(f"Dataset not found: `{file_path}`")
        st.stop()
    df = load_csv(str(file_path))
    features = cfg["features"]
    id_col = cfg["id_col"]
else:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with at least 3 numeric columns to cluster on.")
        st.stop()
    df = pd.read_csv(uploaded)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("Need at least 3 numeric columns. Found: " + ", ".join(numeric_cols))
        st.stop()
    st.subheader("Select columns")
    id_col = st.selectbox("ID column (optional)", ["(none)"] + df.columns.tolist())
    features = st.multiselect("Feature columns (pick 3)", numeric_cols, default=numeric_cols[:3])
    if len(features) != 3:
        st.warning("Select exactly 3 feature columns.")
        st.stop()

# ── Header ───────────────────────────────────────────────────────────────

st.title("K-Means Segmentation")
st.caption("Pick a dataset, cluster it, explore the results.")

# ── Tabs ─────────────────────────────────────────────────────────────────

tab_data, tab_cluster = st.tabs(["Data", "Clustering"])

# ── Data Tab ─────────────────────────────────────────────────────────────

with tab_data:
    st.dataframe(df, use_container_width=True, height=360)
    cols = st.columns(len(features))
    for i, feat in enumerate(features):
        cols[i].metric(feat, f"{df[feat].mean():.1f}", delta=f"std {df[feat].std():.1f}")

# ── Clustering Tab ───────────────────────────────────────────────────────

with tab_cluster:
    X_raw = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # -- Animated 2D: slow-motion cluster formation ------------------------

    st.subheader("2D Cluster Formation")
    chart_2d = st.empty()
    status_text = st.empty()

    anim_key = f"{source}_{dataset_name}_{k}_{speed}"
    if "anim_done" not in st.session_state:
        st.session_state["anim_done"] = {}

    if not st.session_state["anim_done"].get(anim_key, False):
        steps = list(kmeans_steps(X_scaled, k, n_steps=12))
        for i, (labels, centroids) in enumerate(steps):
            title = f"Iteration {i + 1} / {len(steps)}"
            fig = build_frame(df, features[0], features[1], labels, centroids,
                              title, scaler.mean_, scaler.scale_)
            chart_2d.plotly_chart(fig, use_container_width=True, key=f"a2d_{i}")
            status_text.caption(f"Step {i + 1} of {len(steps)} — centroids converging...")
            time.sleep(delay)

        status_text.caption("Converged.")
        st.session_state["anim_done"][anim_key] = True
        final_labels, final_centroids = steps[-1]
    else:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        final_labels = model.fit_predict(X_scaled)
        final_centroids = model.cluster_centers_
        fig = build_frame(df, features[0], features[1], final_labels, final_centroids,
                          "Final Clusters", scaler.mean_, scaler.scale_)
        chart_2d.plotly_chart(fig, use_container_width=True)
        status_text.caption("Converged.")

    if st.button("Replay animation"):
        st.session_state["anim_done"][anim_key] = False
        st.rerun()

    # -- 3D scatter --------------------------------------------------------

    df_out = df.copy()
    df_out["Cluster"] = final_labels.astype(str)
    f = features

    st.subheader("3D Cluster View")
    fig_3d = px.scatter_3d(
        df_out, x=f[0], y=f[1], z=f[2],
        color="Cluster",
        color_discrete_sequence=COLORS,
        template="plotly_dark",
        opacity=0.85,
        hover_data=[id_col] if id_col != "(none)" else None,
    )
    fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.3, color="#333")))
    fig_3d.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=550,
        scene=dict(
            xaxis=dict(title=f[0], backgroundcolor="#0E1117", gridcolor="#1f2937"),
            yaxis=dict(title=f[1], backgroundcolor="#0E1117", gridcolor="#1f2937"),
            zaxis=dict(title=f[2], backgroundcolor="#0E1117", gridcolor="#1f2937"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        paper_bgcolor="#0E1117",
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # -- Summary + Download ------------------------------------------------

    st.subheader("Cluster Summary")
    summary = df_out.groupby("Cluster")[f].mean().round(1)
    summary["Count"] = df_out.groupby("Cluster")["Cluster"].count().values
    st.dataframe(summary, use_container_width=True)

    csv = df_out.to_csv(index=False)
    st.download_button("Download results", csv, "segmented_data.csv", "text/csv")
