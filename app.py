import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path
import time

st.set_page_config(page_title="K-Means Segmentation", page_icon="K", layout="centered")

DATASETS_DIR = Path(__file__).parent / "datasets"

DATASETS = {
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


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def run_kmeans(df: pd.DataFrame, features: list[str], k: int):
    X = df[features].values
    scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(scaled)
    return labels, scaled


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
    k = st.slider("Clusters (K)", 2, 6, 2)
    st.divider()
    st.caption("Streamlit / scikit-learn / Plotly")

cfg = DATASETS[dataset_name]

# ── Header ───────────────────────────────────────────────────────────────

st.title("K-Means Segmentation")
st.caption("Pick a dataset, cluster it, explore the results.")

# ── Load Data ────────────────────────────────────────────────────────────

file_path = DATASETS_DIR / cfg["file"]
if not file_path.exists():
    st.error(f"Dataset not found: `{file_path}`")
    st.stop()

df = load_data(str(file_path))

# ── Tabs ─────────────────────────────────────────────────────────────────

tab_data, tab_cluster = st.tabs(["Data", "Clustering"])

# ── Data Tab ─────────────────────────────────────────────────────────────

with tab_data:
    st.dataframe(df, use_container_width=True, height=360)
    cols = st.columns(len(cfg["features"]))
    for i, feat in enumerate(cfg["features"]):
        cols[i].metric(feat, f"{df[feat].mean():.1f}", delta=f"std {df[feat].std():.1f}")

# ── Clustering Tab ───────────────────────────────────────────────────────

with tab_cluster:
    labels, scaled = run_kmeans(df, cfg["features"], k)
    df_out = df.copy()
    df_out["Cluster"] = labels.astype(str)
    f = cfg["features"]

    # -- Animated 2D scatter (iterative cluster assignment) ----------------

    st.subheader("2D Cluster View")
    pair_x, pair_y = f[0], f[1]
    chart_placeholder = st.empty()

    if "animated" not in st.session_state:
        st.session_state["animated"] = {}

    anim_key = f"{dataset_name}_{k}"
    already_animated = st.session_state["animated"].get(anim_key, False)

    if not already_animated:
        X_raw = df[f].values
        X_scaled = StandardScaler().fit_transform(X_raw)
        n_steps = min(k + 4, 8)

        for step in range(1, n_steps + 1):
            partial = KMeans(
                n_clusters=k, init="k-means++", n_init=1,
                max_iter=step, random_state=42,
            )
            step_labels = partial.fit_predict(X_scaled)
            temp = df.copy()
            temp["Cluster"] = step_labels.astype(str)

            fig_2d = px.scatter(
                temp, x=pair_x, y=pair_y,
                color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                template="plotly_dark",
                opacity=0.8,
            )
            fig_2d.update_layout(
                margin=dict(l=0, r=0, t=30, b=0), height=380,
                title=f"Iteration {step}",
                xaxis_title=pair_x, yaxis_title=pair_y,
            )
            fig_2d.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#222")))
            chart_placeholder.plotly_chart(fig_2d, use_container_width=True, key=f"anim_{step}")
            time.sleep(0.45)

        st.session_state["animated"][anim_key] = True
    else:
        fig_2d = px.scatter(
            df_out, x=pair_x, y=pair_y,
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            template="plotly_dark",
            opacity=0.8,
        )
        fig_2d.update_layout(
            margin=dict(l=0, r=0, t=30, b=0), height=380,
            title="Final Clusters",
            xaxis_title=pair_x, yaxis_title=pair_y,
        )
        fig_2d.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#222")))
        chart_placeholder.plotly_chart(fig_2d, use_container_width=True)

    # -- 3D scatter --------------------------------------------------------

    st.subheader("3D Cluster View")
    fig_3d = px.scatter_3d(
        df_out, x=f[0], y=f[1], z=f[2],
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        template="plotly_dark",
        opacity=0.85,
        hover_data=[cfg["id_col"]],
    )
    fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.3, color="#333")))
    fig_3d.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
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
