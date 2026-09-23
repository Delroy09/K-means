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
PLOTLY_CFG = {"displayModeBar": False, "scrollZoom": False}
PLOTLY_CFG_FULL = {"scrollZoom": False}
SPEED_MAP = {"Slow": 0.9, "Normal": 0.5, "Fast": 0.2}

BUILTIN = {
    "Customer Spending": {
        "file": "customer_spending.csv",
        "features": ["Age", "Annual Income", "Spending Score"],
        "id_col": "CustomerID",
    },
    "Student Marks": {
        "file": "student_marks.csv",
        "features": ["Math", "Science", "English"],
        "id_col": "StudentID",
    },
}


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def kmeans_steps(X_scaled, k, n_steps=12):
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


def build_2d(df, x_col, y_col, labels, centroids, title, sc_mean, sc_scale):
    cx = centroids[:, 0] * sc_scale[0] + sc_mean[0]
    cy = centroids[:, 1] * sc_scale[1] + sc_mean[1]
    fig = go.Figure()
    for c in range(centroids.shape[0]):
        mask = labels == c
        subset = df[mask]
        fig.add_trace(go.Scatter(
            x=subset[x_col], y=subset[y_col], mode="markers",
            marker=dict(size=7, color=COLORS[c % len(COLORS)], opacity=0.75,
                        line=dict(width=0.4, color="#222")),
            name=f"Cluster {c}", showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers",
        marker=dict(size=16, color="white", symbol="x",
                    line=dict(width=2, color="#333")),
        name="Centroids",
    ))
    fig.update_layout(
        template="plotly_dark", title=title,
        margin=dict(l=0, r=0, t=36, b=0), height=420,
        xaxis_title=x_col, yaxis_title=y_col,
        legend=dict(orientation="h", y=-0.12),
        dragmode="pan", xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    return fig


def build_line(df, features, labels):
    fig = go.Figure()
    for c in sorted(set(labels)):
        mask = labels == c
        means = df.loc[mask, features].mean()
        fig.add_trace(go.Scatter(
            x=features, y=means.values, mode="lines+markers",
            marker=dict(size=8), line=dict(width=2.5),
            name=f"Cluster {c}",
        ))
    fig.update_layout(
        template="plotly_dark", title="Cluster Feature Profiles",
        margin=dict(l=0, r=0, t=36, b=0), height=380,
        xaxis_title="Feature", yaxis_title="Mean Value",
        legend=dict(orientation="h", y=-0.15),
        dragmode="pan", xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Dataset Selection
# ═════════════════════════════════════════════════════════════════════════

if "dataset_ready" not in st.session_state:
    st.session_state["dataset_ready"] = False

if not st.session_state["dataset_ready"]:
    st.title("K-Means Segmentation")
    st.caption("Choose a data source to get started.")

    source = st.radio("Data source", ["Built-in dataset", "Upload CSV"], horizontal=True)

    if source == "Built-in dataset":
        dataset_name = st.selectbox("Select dataset", list(BUILTIN.keys()))
        assert dataset_name is not None
        cfg = BUILTIN[dataset_name]
        file_path = DATASETS_DIR / cfg["file"]
        if not file_path.exists():
            st.error(f"Dataset not found: `{file_path}`")
            st.stop()
        if st.button("Continue"):
            st.session_state["df"] = load_csv(str(file_path))
            st.session_state["features"] = cfg["features"]
            st.session_state["id_col"] = cfg["id_col"]
            st.session_state["dataset_ready"] = True
            st.rerun()
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            numeric_cols = df_up.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) < 3:
                st.error("Need at least 3 numeric columns. Found: " + ", ".join(numeric_cols))
                st.stop()
            id_col = st.selectbox("ID column (optional)", ["(none)"] + df_up.columns.tolist())
            features = st.multiselect(
                "Feature columns (pick 3)", numeric_cols, default=numeric_cols[:3],
            )
            if len(features) != 3:
                st.warning("Select exactly 3 feature columns.")
                st.stop()
            if st.button("Continue"):
                st.session_state["df"] = df_up
                st.session_state["features"] = features
                st.session_state["id_col"] = id_col
                st.session_state["dataset_ready"] = True
                st.rerun()

    st.stop()

# ═════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Main App
# ═════════════════════════════════════════════════════════════════════════

df = st.session_state["df"]
features = st.session_state["features"]
id_col = st.session_state["id_col"]

st.title("K-Means Segmentation")

# -- Inline controls (no sidebar) -----------------------------------------

c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 1])
k = c1.slider("Clusters (K)", 2, 6, 2)
speed = c2.select_slider("Speed", ["Slow", "Normal", "Fast"], value="Normal")
animate = c3.toggle("Animations", value=True)
if c4.button("Change dataset"):
    st.session_state["dataset_ready"] = False
    st.session_state.pop("anim_done", None)
    st.rerun()

delay = SPEED_MAP[speed]

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────

tab_data, tab_cluster = st.tabs(["Data", "Clustering"])

# ── Data Tab ─────────────────────────────────────────────────────────────

with tab_data:
    st.dataframe(df, width="stretch", height=360)
    cols = st.columns(len(features))
    for i, feat in enumerate(features):
        cols[i].metric(feat, f"{df[feat].mean():.1f}", delta=f"std {df[feat].std():.1f}")

# ── Clustering Tab ───────────────────────────────────────────────────────

with tab_cluster:
    X_raw = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # -- 2D cluster formation (animated or static) -------------------------

    st.subheader("Cluster Formation")
    chart_2d = st.empty()
    status_text = st.empty()

    anim_key = f"{k}_{speed}"
    if "anim_done" not in st.session_state:
        st.session_state["anim_done"] = {}

    run_anim = animate and not st.session_state["anim_done"].get(anim_key, False)

    if run_anim:
        steps = list(kmeans_steps(X_scaled, k, n_steps=12))
        for i, (labels, centroids) in enumerate(steps):
            title = f"Iteration {i + 1} / {len(steps)}"
            fig = build_2d(df, features[0], features[1], labels, centroids,
                           title, scaler.mean_, scaler.scale_)
            chart_2d.plotly_chart(fig, width="stretch", key=f"a2d_{i}", config=PLOTLY_CFG)
            status_text.caption(f"Step {i + 1} of {len(steps)} — centroids converging...")
            time.sleep(delay)
        status_text.caption("Converged.")
        st.session_state["anim_done"][anim_key] = True
        final_labels, final_centroids = steps[-1]
    else:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        final_labels = model.fit_predict(X_scaled)
        final_centroids = model.cluster_centers_
        fig = build_2d(df, features[0], features[1], final_labels, final_centroids,
                       "Final Clusters", scaler.mean_, scaler.scale_)
        chart_2d.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        status_text.caption("Converged.")

    if animate:
        if st.button("Replay animation"):
            st.session_state["anim_done"][anim_key] = False
            st.rerun()

    # -- Line chart (cluster feature profiles) -----------------------------

    st.subheader("Feature Profiles")
    line_placeholder = st.empty()

    if animate and not st.session_state.get(f"line_done_{anim_key}", False):
        fig_line = go.Figure()
        fig_line.update_layout(
            template="plotly_dark", title="Cluster Feature Profiles",
            margin=dict(l=0, r=0, t=36, b=0), height=380,
            xaxis_title="Feature", yaxis_title="Mean Value",
            legend=dict(orientation="h", y=-0.15),
            dragmode="pan", xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
        )
        for c in sorted(set(final_labels)):
            mask = final_labels == c
            means = df.loc[mask, features].mean()
            fig_line.add_trace(go.Scatter(
                x=features, y=means.values, mode="lines+markers",
                marker=dict(size=8), line=dict(width=2.5),
                name=f"Cluster {c}",
            ))
            line_placeholder.plotly_chart(fig_line, width="stretch",
                                         key=f"line_{c}", config=PLOTLY_CFG)
            time.sleep(delay)
        st.session_state[f"line_done_{anim_key}"] = True
    else:
        fig_line = build_line(df, features, final_labels)
        line_placeholder.plotly_chart(fig_line, width="stretch", config=PLOTLY_CFG)

    # -- 3D scatter (with fullscreen toggle) -------------------------------

    df_out = df.copy()
    df_out["Cluster"] = final_labels.astype(str)
    f = features

    st.subheader("3D View")
    fullscreen = st.toggle("Maximize", value=False, key="fs_3d")
    h3d = 800 if fullscreen else 550

    fig_3d = px.scatter_3d(
        df_out, x=f[0], y=f[1], z=f[2],
        color="Cluster", color_discrete_sequence=COLORS,
        template="plotly_dark", opacity=0.85,
        hover_data=[id_col] if id_col != "(none)" else None,
    )
    fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.3, color="#333")))
    fig_3d.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=h3d,
        scene=dict(
            xaxis=dict(title=f[0], backgroundcolor="#0E1117", gridcolor="#1f2937"),
            yaxis=dict(title=f[1], backgroundcolor="#0E1117", gridcolor="#1f2937"),
            zaxis=dict(title=f[2], backgroundcolor="#0E1117", gridcolor="#1f2937"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
            dragmode="turntable",
        ),
        paper_bgcolor="#0E1117",
    )
    cfg_3d = PLOTLY_CFG_FULL if fullscreen else PLOTLY_CFG
    st.plotly_chart(fig_3d, width="stretch", config=cfg_3d)

    # -- Summary + Download ------------------------------------------------

    st.subheader("Cluster Summary")
    summary = df_out.groupby("Cluster")[f].mean().round(1)
    summary["Count"] = df_out.groupby("Cluster")["Cluster"].count().values
    st.dataframe(summary, width="stretch")

    csv = df_out.to_csv(index=False)
    st.download_button("Download results", csv, "segmented_data.csv", "text/csv")
