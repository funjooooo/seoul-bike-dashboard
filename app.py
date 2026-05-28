# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    from scipy.stats import kruskal
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import scikit_posthocs as sp
    POSTHOCS_AVAILABLE = True
except Exception:
    POSTHOCS_AVAILABLE = False


warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="서울 공공자전거 군집 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
    margin: 0.8rem 0 !important;
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.3rem;
}
section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    padding: 0.45rem 0.7rem !important;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] small,
section[data-testid="stSidebar"] .caption {
    color: #64748b !important;
    font-size: 0.75rem !important;
}

/* Page titles */
h1 {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1.4rem !important;
}

/* Sub-headings */
h2 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    margin-top: 1.6rem !important;
}
h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

/* Alert/info boxes */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
}

/* Dataframe */
.stDataFrame iframe {
    border-radius: 8px;
}

/* Block container */
.block-container {
    padding-top: 1.6rem !important;
    max-width: 1400px;
}

/* Selectbox / multiselect */
div[data-baseweb="select"] > div {
    border-radius: 8px !important;
}

/* Cluster badge style (used in markdown) */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    color: white;
}
</style>
""", unsafe_allow_html=True)
# ───────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

APP_VERSION = "SOURCE_CNTDAY_FIX_V8_FULL_WEATHER_LABEL_FIXED"

SOURCE_FILE = "bike_rawdata_with_cluster_cntday.csv"

RAIN_SUMMARY_FILE = "cluster_rain_summary.csv"
TEMP_SUMMARY_FILE = "cluster_temp_summary.csv"
RAIN_LEVEL_SUMMARY_FILE = "cluster_rainlevel_summary.csv"
BAD_WEATHER_SUMMARY_FILE = "cluster_bad_weather_summary.csv"

FEATURE_COLS = [
    "cnt_per_day",
    "return_rate",
    "bw_rate",
    "weekend_rate",
    "avg_distance",
    "avg_used_time",
    "avg_user_age",
    "morning_ratio",
    "evening_ratio",
    "peak_ratio",
    "variance_norm",
    "entropy",
]

CLUSTER_LABELS = {
    0: "여가형",
    1: "퇴근/중심지역",
    2: "출근/거주지역",
    3: "고수요 중심지역",
}

CLUSTER_COLORS = {
    "여가형": "#3B82F6",
    "퇴근/중심지역": "#F97316",
    "출근/거주지역": "#10B981",
    "고수요 중심지역": "#EF4444",
}

CLUSTER_ICONS = {
    "여가형": "🚴",
    "퇴근/중심지역": "🌆",
    "출근/거주지역": "🏘️",
    "고수요 중심지역": "🔥",
}

CLUSTER_STRATEGY = {
    "여가형": "주말·장시간·장거리 이용이 상대적으로 높으므로 주말 전 자전거 확보, 한강·하천 주변 관리, 관광/레저 수요 대응이 필요합니다.",
    "퇴근/중심지역": "저녁 이용 비율이 높으므로 오후~저녁 시간대 반납·대여 균형을 점검하고, 업무지구 주변 거치대 포화 가능성을 관리해야 합니다.",
    "출근/거주지역": "아침 이용 비율이 높으므로 오전 피크 이전 자전거 공급을 확보하고, 외곽 주거지역 출발 수요를 선제적으로 관리해야 합니다.",
    "고수요 중심지역": "일평균 이용량과 피크 집중도가 높으므로 출퇴근 시간대 재배치 우선순위를 높이고, 자전거 부족·거치대 포화 여부를 집중 점검해야 합니다.",
}

# ── Chart style helpers ─────────────────────────────────────────────────────
_CHART_LAYOUT = dict(
    font=dict(family="'Noto Sans KR', 'Malgun Gothic', sans-serif", size=12),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    title_font=dict(size=14, color="#1e293b"),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(size=12),
    ),
    margin=dict(t=50, b=30, l=30, r=20),
)


def apply_chart_style(fig):
    fig.update_layout(**_CHART_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False)
    return fig
# ───────────────────────────────────────────────────────────────────────────


def file_exists(name: str) -> bool:
    return (DATA_DIR / name).exists()


@st.cache_data(show_spinner=False)
def read_csv_flexible(filename: str) -> pd.DataFrame | None:
    path = DATA_DIR / filename

    if not path.exists():
        return None

    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    return pd.read_csv(path)


def valid_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


@st.cache_data(show_spinner=False)
def load_source_data() -> pd.DataFrame:
    df = read_csv_flexible(SOURCE_FILE)

    if df is None:
        st.error(
            f"{SOURCE_FILE} 파일이 필요합니다. "
            "이 파일은 30일 미만 대여소가 포함된 원본 기준 파일이어야 합니다."
        )
        st.stop()

    for col in FEATURE_COLS + ["cnt_day", "rent_station_no", "cluster", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cnt_day" not in df.columns:
        st.error(f"{SOURCE_FILE} 파일에 cnt_day 컬럼이 없습니다.")
        st.stop()

    under_30_count = int((df["cnt_day"] < 30).sum())

    if under_30_count == 0:
        st.error(
            f"현재 {SOURCE_FILE} 파일 안에 cnt_day < 30 대여소가 0개입니다. "
            "즉, 이미 30일 기준으로 필터링된 파일을 읽고 있습니다."
        )
        st.stop()

    return df


def prepare_model_data(df: pd.DataFrame, min_days: int = 30):
    data = df.copy()

    if "cnt_day" in data.columns and min_days > 0:
        data["cnt_day"] = pd.to_numeric(data["cnt_day"], errors="coerce")
        data = data[data["cnt_day"] >= min_days].copy()

    features = valid_feature_cols(data)

    if not features:
        return data, features, np.array([])

    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=features).copy()

    if data.empty:
        return data, features, np.array([])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return data, features, X_scaled


@st.cache_data(show_spinner=False)
def create_kmeans_cluster(df: pd.DataFrame, min_days: int = 30, k: int = 4) -> pd.DataFrame:
    data, features, X_scaled = prepare_model_data(df, min_days=min_days)

    if len(data) <= k or len(features) == 0:
        return data

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    data = data.copy()
    data["cluster"] = model.fit_predict(X_scaled)
    data["cluster"] = data["cluster"].astype(int)
    data["cluster_type"] = data["cluster"].map(CLUSTER_LABELS)

    return data


@st.cache_data(show_spinner=False)
def load_base_data(source_df: pd.DataFrame) -> pd.DataFrame:
    df = create_kmeans_cluster(source_df, min_days=30, k=4)

    if df.empty:
        st.error("cnt_day >= 30 조건 적용 후 분석 가능한 데이터가 없습니다.")
        st.stop()

    if ("latitude" not in df.columns or "longitude" not in df.columns) and file_exists("station_master.csv"):
        station = read_csv_flexible("station_master.csv")

        if station is not None and {"station_no", "latitude", "longitude"}.issubset(station.columns):
            station = station.copy()
            station["station_no"] = pd.to_numeric(station["station_no"], errors="coerce")
            df["rent_station_no"] = pd.to_numeric(df["rent_station_no"], errors="coerce")

            keep_cols = [
                c for c in [
                    "station_no",
                    "latitude",
                    "longitude",
                    "district_name",
                    "detail_address",
                ]
                if c in station.columns
            ]

            df = pd.merge(
                df,
                station[keep_cols],
                left_on="rent_station_no",
                right_on="station_no",
                how="left",
            )

    if "cluster" in df.columns:
        df = df.dropna(subset=["cluster"]).copy()
        df["cluster"] = df["cluster"].astype(int)
        df["cluster_type"] = df["cluster"].map(CLUSTER_LABELS)

    if "rent_station_name" not in df.columns:
        df["rent_station_name"] = df.get("rent_station_no", pd.Series(range(len(df)))).astype(str)

    return df


@st.cache_data(show_spinner=False)
def kmeans_k_compare(df: pd.DataFrame, min_days: int = 30, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    data, features, X_scaled = prepare_model_data(df, min_days=min_days)

    rows = []

    if data.empty or len(features) == 0:
        return pd.DataFrame()

    if len(data) < k_max:
        k_max = max(2, len(data) - 1)

    for k in range(k_min, k_max + 1):
        if len(data) <= k:
            continue

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = model.fit_predict(X_scaled)

        rows.append(
            {
                "k": k,
                "station_count": len(data),
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(X_scaled, labels),
                "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
                "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def sensitivity_analysis(source_df: pd.DataFrame, thresholds=(0, 7, 14, 30, 60), k: int = 4) -> pd.DataFrame:
    rows = []

    total_station_count = len(source_df)

    for day in thresholds:
        try:
            if day == 0:
                threshold_label = "미제거"
                before_feature_drop = source_df.copy()
                removed_by_cnt_day = 0
            else:
                threshold_label = f"{day}일 이상"
                before_feature_drop = source_df[source_df["cnt_day"] >= day].copy()
                removed_by_cnt_day = int((source_df["cnt_day"] < day).sum())

            data, features, X_scaled = prepare_model_data(source_df, min_days=day)
            feature_drop_count = len(before_feature_drop) - len(data)

            if len(data) <= k or len(features) == 0:
                rows.append(
                    {
                        "threshold": threshold_label,
                        "min_cnt_day": day,
                        "source_total_count": total_station_count,
                        "removed_by_cnt_day": removed_by_cnt_day,
                        "model_input_count": len(data),
                        "feature_drop_count": feature_drop_count,
                        "error": "분석 가능한 데이터 수가 부족합니다.",
                    }
                )
                continue

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=20,
            )

            labels = model.fit_predict(X_scaled)
            vc = pd.Series(labels).value_counts().sort_index()

            row = {
                "threshold": threshold_label,
                "min_cnt_day": day,
                "source_total_count": total_station_count,
                "removed_by_cnt_day": removed_by_cnt_day,
                "model_input_count": len(data),
                "feature_drop_count": feature_drop_count,
                "silhouette_score": silhouette_score(X_scaled, labels),
                "min_cluster_size": int(vc.min()),
                "max_cluster_size": int(vc.max()),
                "abnormal_cluster_exists": bool(int(vc.min()) <= 10),
            }

            for c, n in vc.items():
                row[f"cluster_{int(c)}_count"] = int(n)

            rows.append(row)

        except Exception as e:
            rows.append(
                {
                    "threshold": "오류",
                    "min_cnt_day": day,
                    "source_total_count": total_station_count,
                    "error": str(e),
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "cluster" not in d.columns:
        return pd.DataFrame()

    agg_dict = {"rent_station_name": "count"}

    for c in [
        "cnt_per_day",
        "weekend_rate",
        "avg_distance",
        "avg_used_time",
        "morning_ratio",
        "evening_ratio",
        "peak_ratio",
        "entropy",
        "variance_norm",
    ]:
        if c in d.columns:
            agg_dict[c] = "mean"

    summary = d.groupby("cluster").agg(agg_dict).reset_index()
    summary = summary.rename(columns={"rent_station_name": "station_count"})

    summary["cluster_type"] = summary["cluster"].map(CLUSTER_LABELS)
    summary["station_share"] = summary["station_count"] / summary["station_count"].sum()

    if "cnt_per_day" in d.columns:
        use_sum = d.groupby("cluster")["cnt_per_day"].sum().reset_index(name="cnt_per_day_sum")
        summary = summary.merge(use_sum, on="cluster", how="left")
        summary["usage_share"] = summary["cnt_per_day_sum"] / summary["cnt_per_day_sum"].sum()

    return summary


@st.cache_data(show_spinner=False)
def algorithm_compare(df: pd.DataFrame) -> pd.DataFrame:
    data, features, X_scaled = prepare_model_data(df, min_days=30)

    rows = []

    if len(data) < 10 or len(features) == 0:
        return pd.DataFrame()

    km = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20,
    )

    labels = km.fit_predict(X_scaled)

    rows.append(
        {
            "algorithm": "K-means",
            "cluster_count": len(set(labels)),
            "noise_count": 0,
            "silhouette_score": silhouette_score(X_scaled, labels),
            "interpretation": "상대적 거리 차이를 기준으로 운영 유형 분류 가능",
        }
    )

    agg = AgglomerativeClustering(
        n_clusters=3,
        linkage="ward",
    )

    labels = agg.fit_predict(X_scaled)

    rows.append(
        {
            "algorithm": "Agglomerative",
            "cluster_count": len(set(labels)),
            "noise_count": 0,
            "silhouette_score": silhouette_score(X_scaled, labels),
            "interpretation": "일부 군집이 더 큰 상위 유형으로 병합되는 경향",
        }
    )

    db = DBSCAN(
        eps=1.2,
        min_samples=10,
    )

    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())

    valid = labels != -1
    sil = np.nan

    if n_clusters >= 2 and valid.sum() > 0:
        sil = silhouette_score(X_scaled[valid], labels[valid])

    rows.append(
        {
            "algorithm": "DBSCAN",
            "cluster_count": n_clusters,
            "noise_count": noise_count,
            "silhouette_score": sil,
            "interpretation": "절대적 밀도 기반 군집이 약해 명확한 군집 형성 어려움",
        }
    )

    return pd.DataFrame(rows)


def operating_priority(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    needed = [c for c in ["cnt_per_day", "peak_ratio", "variance_norm", "bw_rate"] if c in d.columns]

    if not needed:
        return pd.DataFrame()

    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        min_v = d[c].min()
        max_v = d[c].max()

        if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
            d[f"{c}_score"] = 0
        else:
            d[f"{c}_score"] = (d[c] - min_v) / (max_v - min_v)

    d["priority_score"] = 0

    weights = {
        "cnt_per_day": 0.4,
        "peak_ratio": 0.3,
        "variance_norm": 0.2,
        "bw_rate": 0.1,
    }

    for c, w in weights.items():
        score_col = f"{c}_score"

        if score_col in d.columns:
            d["priority_score"] += d[score_col] * w

    cols = [
        c for c in [
            "rent_station_no",
            "rent_station_name",
            "cluster",
            "cluster_type",
            "cnt_per_day",
            "peak_ratio",
            "variance_norm",
            "bw_rate",
            "priority_score",
        ]
        if c in d.columns
    ]

    return d[cols].sort_values("priority_score", ascending=False)


def generate_station_comment(row: pd.Series, df: pd.DataFrame) -> str:
    cluster_type = row.get("cluster_type", "해당 군집")

    comments = [f"해당 대여소는 {cluster_type}에 속합니다."]

    for col, label in [
        ("cnt_per_day", "일평균 이용량"),
        ("morning_ratio", "아침 이용 비율"),
        ("evening_ratio", "저녁 이용 비율"),
        ("weekend_rate", "주말 이용 비율"),
        ("peak_ratio", "피크 집중도"),
    ]:
        if col in row.index and col in df.columns and pd.notna(row[col]):
            avg = df[col].mean()
            direction = "높습니다" if row[col] >= avg else "낮습니다"
            comments.append(f"{label}은 전체 평균 대비 {direction}.")

    comments.append(
        CLUSTER_STRATEGY.get(
            cluster_type,
            "개별 지표를 함께 확인하여 운영 판단에 활용해야 합니다.",
        )
    )

    return " ".join(comments)


def plot_map(df: pd.DataFrame):
    if not {"latitude", "longitude"}.issubset(df.columns):
        st.warning("지도 표시를 위해 latitude, longitude 컬럼 또는 station_master.csv가 필요합니다.")
        return

    mdf = df.dropna(subset=["latitude", "longitude"]).copy()

    if mdf.empty:
        st.warning("좌표가 있는 데이터가 없습니다.")
        return

    size_col = "cnt_per_day" if "cnt_per_day" in mdf.columns else None

    hover_cols = [
        c for c in [
            "rent_station_no",
            "rent_station_name",
            "cluster_type",
            "cnt_per_day",
            "morning_ratio",
            "evening_ratio",
            "weekend_rate",
        ]
        if c in mdf.columns
    ]

    fig = px.scatter_mapbox(
        mdf,
        lat="latitude",
        lon="longitude",
        color="cluster_type" if "cluster_type" in mdf.columns else "cluster",
        color_discrete_map=CLUSTER_COLORS,
        size=size_col,
        size_max=14,
        hover_name="rent_station_name",
        hover_data=hover_cols,
        zoom=10,
        height=720,
        mapbox_style="carto-positron",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        legend_title_text="군집",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1),
        font=dict(family="'Noto Sans KR', sans-serif", size=12),
    )

    st.plotly_chart(fig, use_container_width=True)


def page_intro(df: pd.DataFrame):
    st.title("서울 공공자전거 대여소 군집 분석 기반 운영 의사결정 대시보드")

    st.markdown(
        """
        본 대시보드는 서울 공공자전거 대여소의 이용 패턴을 군집화하고,
        군집별 공간 분포와 시간대 특성을 바탕으로 운영 전략을 도출하기 위한 도구입니다.

        핵심 관점은 단순히 군집 결과를 보여주는 것이 아니라,
        각 대여소가 어떤 운영 유형에 가까운지 확인하고
        재배치·증차·관리 우선순위를 판단하는 것입니다.
        """
    )

    st.info(
        "Silhouette score가 낮은 편이므로, 본 군집은 절대적으로 뚜렷한 분류가 아니라 "
        "운영 해석을 위한 상대적 유형화로 해석해야 합니다."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("분석 대여소 수", f"{len(df):,}")

    if "cnt_per_day" in df.columns:
        c2.metric("평균 일 이용량", f"{df['cnt_per_day'].mean():,.1f}")

    if "avg_distance" in df.columns:
        c3.metric("평균 이동거리", f"{df['avg_distance'].mean():,.2f}")

    if "avg_used_time" in df.columns:
        c4.metric("평균 이용시간", f"{df['avg_used_time'].mean():,.1f}")

    st.subheader("군집 유형 정의")

    active_clusters = set(df.get("cluster", pd.Series(dtype=int)).dropna().astype(int).unique())

    cluster_cols = st.columns(len([k for k in CLUSTER_LABELS if k in active_clusters]) or 1)

    for idx, (_, v) in enumerate([item for item in CLUSTER_LABELS.items() if item[0] in active_clusters]):
        color = CLUSTER_COLORS.get(v, "#64748b")
        icon = CLUSTER_ICONS.get(v, "📍")
        strategy = CLUSTER_STRATEGY.get(v, "")
        with cluster_cols[idx]:
            st.markdown(
                f"""<div style="background:{color}14;border:1.5px solid {color};border-radius:12px;
                    padding:1rem 1.1rem;height:100%;">
                    <div style="font-size:1.4rem;margin-bottom:4px;">{icon}</div>
                    <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:6px;">{v}</div>
                    <div style="font-size:0.82rem;color:#475569;line-height:1.55;">{strategy}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)


def page_overview(df: pd.DataFrame):
    st.title("전체 현황")

    summary = cluster_summary(df)

    if summary.empty:
        st.warning("군집 요약을 생성할 수 없습니다.")
        return

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            summary,
            x="cluster_type",
            y="station_count",
            text="station_count",
            color="cluster_type",
            color_discrete_map=CLUSTER_COLORS,
            title="군집별 대여소 수",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "cnt_per_day_sum" in summary.columns:
            fig = px.bar(
                summary,
                x="cluster_type",
                y="cnt_per_day_sum",
                text="cnt_per_day_sum",
                color="cluster_type",
                color_discrete_map=CLUSTER_COLORS,
                title="군집별 일평균 이용량 합계",
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            apply_chart_style(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("군집별 운영 비중")

    show = summary.copy()

    for c in ["station_share", "usage_share"]:
        if c in show.columns:
            show[c] = (show[c] * 100).round(2)

    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("주요 변수 분포")

    features = valid_feature_cols(df)

    if not features:
        st.warning("표시 가능한 feature 컬럼이 없습니다.")
        return

    col = st.selectbox("분포를 확인할 변수", features, index=0)

    fig = px.box(
        df,
        x="cluster_type",
        y=col,
        color="cluster_type",
        color_discrete_map=CLUSTER_COLORS,
        points="outliers",
        title=f"군집별 {col} 분포",
    )
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def page_map(df: pd.DataFrame):
    st.title("지도 기반 군집 시각화")

    fdf = df.copy()

    cols = st.columns(4)

    with cols[0]:
        types = sorted(fdf["cluster_type"].dropna().unique()) if "cluster_type" in fdf.columns else []
        selected_types = st.multiselect("군집 유형", types, default=types)

    with cols[1]:
        if "district_name" in fdf.columns:
            districts = sorted(fdf["district_name"].dropna().unique())
            selected_districts = st.multiselect("자치구", districts, default=[])
        else:
            selected_districts = []

    with cols[2]:
        if "cnt_per_day" in fdf.columns:
            mn = float(fdf["cnt_per_day"].min())
            mx = float(fdf["cnt_per_day"].max())
            cnt_range = st.slider("일평균 이용량", mn, mx, (mn, mx))
        else:
            cnt_range = None

    with cols[3]:
        if "weekend_rate" in fdf.columns:
            mn = float(fdf["weekend_rate"].min())
            mx = float(fdf["weekend_rate"].max())
            weekend_range = st.slider("주말 이용률", mn, mx, (mn, mx))
        else:
            weekend_range = None

    if selected_types:
        fdf = fdf[fdf["cluster_type"].isin(selected_types)]

    if selected_districts:
        fdf = fdf[fdf["district_name"].isin(selected_districts)]

    if cnt_range and "cnt_per_day" in fdf.columns:
        fdf = fdf[fdf["cnt_per_day"].between(cnt_range[0], cnt_range[1])]

    if weekend_range and "weekend_rate" in fdf.columns:
        fdf = fdf[fdf["weekend_rate"].between(weekend_range[0], weekend_range[1])]

    st.caption(f"필터 적용 후 대여소 수: {len(fdf):,}")

    plot_map(fdf)


def page_cluster_compare(df: pd.DataFrame):
    st.title("군집 특성 비교")

    features = valid_feature_cols(df)

    if not features:
        st.warning("분석 가능한 feature 컬럼이 없습니다.")
        return

    summary = df.groupby("cluster_type")[features].mean().reset_index()

    heat = summary.set_index("cluster_type")
    z = (heat - heat.mean()) / heat.std(ddof=0)

    fig = px.imshow(
        z.round(2),
        text_auto=True,
        aspect="auto",
        title="군집별 평균 특성 Heatmap(Z-score)",
        color_continuous_scale="RdBu_r",
    )
    fig.update_layout(
        font=dict(family="'Noto Sans KR', sans-serif", size=12),
        title_font=dict(size=14, color="#1e293b"),
        margin=dict(t=50, b=30, l=30, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Radar Chart")

    default_vars = [
        c for c in [
            "cnt_per_day",
            "weekend_rate",
            "avg_distance",
            "avg_used_time",
            "morning_ratio",
            "evening_ratio",
            "peak_ratio",
            "entropy",
        ]
        if c in features
    ]

    selected = st.multiselect("비교 변수", features, default=default_vars)

    if selected:
        radar_base = summary[["cluster_type"] + selected].copy()

        for c in selected:
            mn = radar_base[c].min()
            mx = radar_base[c].max()

            if mx == mn:
                radar_base[c] = 0
            else:
                radar_base[c] = (radar_base[c] - mn) / (mx - mn)

        fig = go.Figure()

        for _, row in radar_base.iterrows():
            cluster_name = row["cluster_type"]
            color = CLUSTER_COLORS.get(cluster_name, "#64748b")
            fig.add_trace(
                go.Scatterpolar(
                    r=row[selected].tolist() + [row[selected[0]]],
                    theta=selected + [selected[0]],
                    fill="toself",
                    name=cluster_name,
                    line=dict(color=color, width=2),
                    fillcolor="rgba({},{},{},0.15)".format(
                        int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    ) if color.startswith("#") else color,
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#e2e8f0"),
                angularaxis=dict(gridcolor="#e2e8f0"),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=650,
            font=dict(family="'Noto Sans KR', sans-serif", size=12),
            legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=30),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("알고리즘 비교")

    comp = algorithm_compare(df)

    st.dataframe(comp, use_container_width=True, hide_index=True)


def page_station_detail(df: pd.DataFrame):
    st.title("대여소 상세 조회")

    keyword = st.text_input("대여소명 또는 대여소 번호 검색")

    sdf = df.copy()

    if keyword:
        keyword_lower = keyword.lower()

        mask = sdf["rent_station_name"].astype(str).str.lower().str.contains(keyword_lower, na=False)

        if "rent_station_no" in sdf.columns:
            mask = mask | sdf["rent_station_no"].astype(str).str.contains(keyword, na=False)

        sdf = sdf[mask]

    st.caption(f"검색 결과: {len(sdf):,}건")

    if sdf.empty:
        return

    display_cols = [
        c for c in [
            "rent_station_no",
            "rent_station_name",
            "cluster_type",
            "cnt_per_day",
            "weekend_rate",
            "avg_distance",
            "avg_used_time",
            "morning_ratio",
            "evening_ratio",
            "peak_ratio",
            "entropy",
        ]
        if c in sdf.columns
    ]

    st.dataframe(sdf[display_cols].head(100), use_container_width=True, hide_index=True)

    names = sdf["rent_station_name"].astype(str).tolist()
    selected_name = st.selectbox("상세 확인 대여소", names)

    row = sdf[sdf["rent_station_name"].astype(str) == selected_name].iloc[0]

    st.subheader(selected_name)

    cluster_type = row.get("cluster_type", "-")
    cluster_color = CLUSTER_COLORS.get(cluster_type, "#64748b")
    cluster_icon = CLUSTER_ICONS.get(cluster_type, "📍")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        f"""<div style="background:{cluster_color}18;border:1.5px solid {cluster_color};
            border-radius:10px;padding:0.8rem 1rem;text-align:center;">
            <div style="font-size:0.75rem;font-weight:600;color:#64748b;text-transform:uppercase;
                letter-spacing:0.05em;">군집</div>
            <div style="font-size:1.1rem;font-weight:700;color:{cluster_color};margin-top:4px;">
                {cluster_icon} {cluster_type}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if "cnt_per_day" in row.index:
        c2.metric("일평균 이용량", f"{row['cnt_per_day']:.2f}")

    if "morning_ratio" in row.index:
        c3.metric("아침 비율", f"{row['morning_ratio']:.3f}")

    if "evening_ratio" in row.index:
        c4.metric("저녁 비율", f"{row['evening_ratio']:.3f}")

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
    st.info(generate_station_comment(row, df))

    if {"latitude", "longitude"}.issubset(df.columns):
        plot_map(pd.DataFrame([row]))


def page_strategy(df: pd.DataFrame):
    st.title("운영 전략 추천")

    st.subheader("군집별 운영 전략")

    summary = cluster_summary(df)

    if not summary.empty:
        strategy = summary[["cluster", "cluster_type", "station_count"]].copy()

        if "cnt_per_day_sum" in summary.columns:
            strategy["cnt_per_day_sum"] = summary["cnt_per_day_sum"]
            strategy["usage_share_pct"] = (summary["usage_share"] * 100).round(2)

        strategy["운영 전략"] = strategy["cluster_type"].map(CLUSTER_STRATEGY)

        st.dataframe(strategy, use_container_width=True, hide_index=True)

    st.subheader("운영 우선순위 TOP 20")

    priority = operating_priority(df)

    if priority.empty:
        st.warning("우선순위 점수를 계산할 수 없습니다.")
    else:
        st.caption(
            "점수 = cnt_per_day 0.4 + peak_ratio 0.3 + variance_norm 0.2 + bw_rate 0.1의 Min-Max 정규화 가중합"
        )

        st.dataframe(priority.head(20), use_container_width=True, hide_index=True)

        fig = px.bar(
            priority.head(20).sort_values("priority_score"),
            x="priority_score",
            y="rent_station_name",
            orientation="h",
            color="cluster_type",
            color_discrete_map=CLUSTER_COLORS,
            title="운영 우선순위 상위 대여소",
        )
        fig.update_traces(marker_line_width=0)
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)


def page_validation(source_df: pd.DataFrame):
    st.title("교수님 피드백 대응 검증")

    st.subheader("0. 기준 데이터 확인")

    st.write(
        {
            "APP_VERSION": APP_VERSION,
            "source_file": SOURCE_FILE,
            "source_total_count": len(source_df),
            "cnt_day_under_7": int((source_df["cnt_day"] < 7).sum()),
            "cnt_day_under_14": int((source_df["cnt_day"] < 14).sum()),
            "cnt_day_under_30": int((source_df["cnt_day"] < 30).sum()),
            "cnt_day_under_60": int((source_df["cnt_day"] < 60).sum()),
        }
    )

    st.subheader("1. K-means Elbow / Silhouette")

    result = kmeans_k_compare(source_df, min_days=30)

    st.dataframe(result, use_container_width=True, hide_index=True)

    if not result.empty and {"k", "silhouette_score"}.issubset(result.columns):
        c1, c2 = st.columns(2)

        with c1:
            fig = px.line(
                result,
                x="k",
                y="inertia",
                markers=True,
                title="Elbow Method",
                line_shape="spline",
            )
            fig.update_traces(line=dict(color="#3B82F6", width=2.5), marker=dict(size=7))
            apply_chart_style(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.line(
                result,
                x="k",
                y="silhouette_score",
                markers=True,
                title="Silhouette Score",
                line_shape="spline",
            )
            fig.update_traces(line=dict(color="#10B981", width=2.5), marker=dict(size=7))
            apply_chart_style(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "k=4의 Silhouette score가 낮으면 군집 경계가 약하다는 뜻입니다. "
        "따라서 군집은 절대적 분류가 아니라 운영 해석을 위한 상대적 유형화로 사용해야 합니다."
    )

    st.subheader("2. 운영일 기준 Sensitivity Analysis")

    st.caption(
        "이 분석은 30일 미만 대여소가 포함된 bike_rawdata_with_cluster_cntday.csv 원본 기준으로 계산합니다. "
        "'미제거' 행은 운영일 기준을 적용하지 않았을 때 비정상 소규모 군집이 발생하는지 확인하기 위한 비교 기준입니다."
    )

    sens = sensitivity_analysis(source_df, thresholds=(0, 7, 14, 30, 60), k=4)

    st.dataframe(sens, use_container_width=True, hide_index=True)

    if not sens.empty and "silhouette_score" in sens.columns:
        fig = px.line(
            sens,
            x="threshold",
            y="silhouette_score",
            markers=True,
            title="운영일 기준별 Silhouette 변화",
            line_shape="spline",
        )
        fig.update_traces(line=dict(color="#3B82F6", width=2.5), marker=dict(size=7))
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    if not sens.empty and "min_cluster_size" in sens.columns:
        fig = px.bar(
            sens,
            x="threshold",
            y="min_cluster_size",
            text="min_cluster_size",
            title="운영일 기준별 최소 군집 크기 변화",
            color_discrete_sequence=["#3B82F6"],
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

        abnormal_rows = sens[sens["min_cluster_size"] <= 10]

        if not abnormal_rows.empty:
            st.error(
                "운영일 기준이 낮거나 미제거 상태에서는 10개 이하의 비정상 소규모 군집이 발생합니다. "
                "이는 실제 운영 유형이라기보다 운영 기간이 짧은 대여소의 통계값 왜곡으로 해석할 수 있습니다."
            )

        stable_rows = sens[sens["min_cnt_day"] >= 30]

        if not stable_rows.empty:
            st.success(
                "민감도 분석 결과, 비정상 소규모 군집은 운영일 기준 미적용 상태에서 발생했으며 "
                "7일 이상 기준부터는 해당 문제가 해소되었습니다. "
                "다만 이번 분석에서는 주중·주말 패턴과 월 단위 운영 안정성을 함께 고려하기 위해 30일 기준을 최종 전처리 기준으로 채택하였습니다."
            )

    st.subheader("3. Kruskal-Wallis 검정")

    analysis_df = create_kmeans_cluster(source_df, min_days=30, k=4)

    if not SCIPY_AVAILABLE:
        st.error("scipy가 설치되어 있지 않아 Kruskal-Wallis 검정을 실행할 수 없습니다.")
    else:
        features = valid_feature_cols(analysis_df)
        rows = []

        for col in features:
            groups = [
                g[col].dropna().values
                for _, g in analysis_df.groupby("cluster")
                if len(g[col].dropna()) > 0
            ]

            if len(groups) >= 2:
                stat, p = kruskal(*groups)

                rows.append(
                    {
                        "variable": col,
                        "kruskal_statistic": stat,
                        "p_value": p,
                        "significant_0.05": p < 0.05,
                    }
                )

        test_df = pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame()

        st.dataframe(test_df, use_container_width=True, hide_index=True)

        st.caption(
            "이용량 분포가 right-skewed일 가능성이 높으므로 ANOVA보다 비모수 검정인 Kruskal-Wallis를 우선 적용합니다."
        )

        if POSTHOCS_AVAILABLE and not test_df.empty:
            sig_vars = test_df[test_df["significant_0.05"]]["variable"].tolist()

            if sig_vars:
                var = st.selectbox("Dunn's test 사후분석 변수", sig_vars)

                posthoc = sp.posthoc_dunn(
                    analysis_df,
                    val_col=var,
                    group_col="cluster",
                    p_adjust="bonferroni",
                )

                st.dataframe(posthoc, use_container_width=True)
        else:
            st.info("Dunn's test를 실행하려면 scikit-posthocs 설치가 필요합니다. 설치 명령: pip install scikit-posthocs")


def load_weather_summary(filename: str) -> pd.DataFrame | None:
    df = read_csv_flexible(filename)

    if df is None:
        return None

    if "avg_rent_count" in df.columns:
        df["avg_rent_count"] = pd.to_numeric(df["avg_rent_count"], errors="coerce")

    if "total_rent_count" in df.columns:
        df["total_rent_count"] = pd.to_numeric(df["total_rent_count"], errors="coerce")

    if "observation_count" in df.columns:
        df["observation_count"] = pd.to_numeric(df["observation_count"], errors="coerce")

    return df


def page_external(df: pd.DataFrame):
    st.title("날씨·교통 활용")

    st.markdown(
        """
        이 탭은 날씨 데이터를 군집 생성 변수로 직접 넣는 것이 아니라,
        이미 생성된 군집이 날씨 조건에 따라 이용량이 어떻게 달라지는지 확인하기 위한 외부 해석 분석입니다.
        """
    )

    rain_summary = load_weather_summary(RAIN_SUMMARY_FILE)
    temp_summary = load_weather_summary(TEMP_SUMMARY_FILE)
    rain_level_summary = load_weather_summary(RAIN_LEVEL_SUMMARY_FILE)
    bad_weather_summary = load_weather_summary(BAD_WEATHER_SUMMARY_FILE)

    st.subheader("1. 비 여부별 군집 평균 이용량")

    if rain_summary is None:
        st.warning(f"{RAIN_SUMMARY_FILE} 파일이 없습니다.")
    else:
        st.dataframe(rain_summary, use_container_width=True, hide_index=True)

        fig = px.bar(
            rain_summary,
            x="cluster_type",
            y="avg_rent_count",
            color="rain_yn",
            barmode="group",
            text="avg_rent_count",
            title="비 여부별 군집 평균 이용량",
            color_discrete_sequence=["#3B82F6", "#94A3B8"],
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

        if {"cluster_type", "rain_yn", "avg_rent_count"}.issubset(rain_summary.columns):
            pivot = rain_summary.pivot_table(
                index="cluster_type",
                columns="rain_yn",
                values="avg_rent_count",
                aggfunc="mean",
            ).reset_index()

            if "비" in pivot.columns and "비 안옴" in pivot.columns:
                pivot["rain_change_rate_pct"] = (
                    (pivot["비"] - pivot["비 안옴"]) / pivot["비 안옴"] * 100
                ).round(2)

                st.subheader("비 오는 날 이용량 변화율")
                st.dataframe(pivot, use_container_width=True, hide_index=True)

                fig = px.bar(
                    pivot,
                    x="cluster_type",
                    y="rain_change_rate_pct",
                    text="rain_change_rate_pct",
                    title="비 오는 날 군집별 이용량 변화율",
                    color="cluster_type",
                    color_discrete_map=CLUSTER_COLORS,
                )
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_line_width=0)
                apply_chart_style(fig)
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. 기온 구간별 군집 평균 이용량")

    if temp_summary is None:
        st.warning(f"{TEMP_SUMMARY_FILE} 파일이 없습니다.")
    else:
        temp_order = ["저온", "쌀쌀", "적정", "고온"]
        if "temp_group" in temp_summary.columns:
            temp_summary["temp_group"] = pd.Categorical(
                temp_summary["temp_group"],
                categories=temp_order,
                ordered=True,
            )
            temp_summary = temp_summary.sort_values(["cluster_type", "temp_group"])

        st.dataframe(temp_summary, use_container_width=True, hide_index=True)

        fig = px.line(
            temp_summary,
            x="temp_group",
            y="avg_rent_count",
            color="cluster_type",
            color_discrete_map=CLUSTER_COLORS,
            markers=True,
            title="기온 구간별 군집 평균 이용량",
            line_shape="spline",
        )
        fig.update_traces(marker=dict(size=7), line=dict(width=2.5))
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. 강수 강도별 군집 평균 이용량")

    if rain_level_summary is None:
        st.warning(f"{RAIN_LEVEL_SUMMARY_FILE} 파일이 없습니다.")
    else:
        rain_order = ["강수 없음", "약한 비", "보통 비", "강한 비"]
        if "rain_level" in rain_level_summary.columns:
            rain_level_summary["rain_level"] = pd.Categorical(
                rain_level_summary["rain_level"],
                categories=rain_order,
                ordered=True,
            )
            rain_level_summary = rain_level_summary.sort_values(["cluster_type", "rain_level"])

        st.dataframe(rain_level_summary, use_container_width=True, hide_index=True)

        fig = px.line(
            rain_level_summary,
            x="rain_level",
            y="avg_rent_count",
            color="cluster_type",
            color_discrete_map=CLUSTER_COLORS,
            markers=True,
            title="강수 강도별 군집 평균 이용량",
            line_shape="spline",
        )
        fig.update_traces(marker=dict(size=7), line=dict(width=2.5))
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("4. 정상/악천후별 군집 평균 이용량")

    if bad_weather_summary is None:
        st.warning(f"{BAD_WEATHER_SUMMARY_FILE} 파일이 없습니다.")
    else:
        st.dataframe(bad_weather_summary, use_container_width=True, hide_index=True)

        fig = px.bar(
            bad_weather_summary,
            x="cluster_type",
            y="avg_rent_count",
            color="bad_weather_yn",
            barmode="group",
            text="avg_rent_count",
            title="정상/악천후별 군집 평균 이용량",
            color_discrete_sequence=["#10B981", "#EF4444"],
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

        if {"cluster_type", "bad_weather_yn", "avg_rent_count"}.issubset(bad_weather_summary.columns):
            pivot = bad_weather_summary.pivot_table(
                index="cluster_type",
                columns="bad_weather_yn",
                values="avg_rent_count",
                aggfunc="mean",
            ).reset_index()

            if "악천후" in pivot.columns and "정상" in pivot.columns:
                pivot["bad_weather_change_rate_pct"] = (
                    (pivot["악천후"] - pivot["정상"]) / pivot["정상"] * 100
                ).round(2)

                st.subheader("악천후 시 이용량 변화율")
                st.dataframe(pivot, use_container_width=True, hide_index=True)

    st.subheader("5. 시간대별 군집 활성도")

    bike_hour = read_csv_flexible("시간대별정보_kepler_cluster.csv")

    if bike_hour is not None:
        b = bike_hour.copy()

        for c in ["rent_hour", "hour_rent_cnt", "cluster"]:
            if c in b.columns:
                b[c] = pd.to_numeric(b[c], errors="coerce")

        if {"rent_hour", "hour_rent_cnt", "cluster"}.issubset(b.columns):
            b = b.dropna(subset=["rent_hour", "hour_rent_cnt", "cluster"])
            b["cluster"] = b["cluster"].astype(int)
            b["cluster_type"] = b["cluster"].map(CLUSTER_LABELS)

            hourly = b.groupby(["cluster_type", "rent_hour"], as_index=False)["hour_rent_cnt"].mean()

            fig = px.line(
                hourly,
                x="rent_hour",
                y="hour_rent_cnt",
                color="cluster_type",
                color_discrete_map=CLUSTER_COLORS,
                markers=True,
                title="군집별 시간대 평균 이용량",
                line_shape="spline",
            )
            fig.update_traces(marker=dict(size=6), line=dict(width=2.5))
            apply_chart_style(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("시간대별정보_kepler_cluster.csv에 rent_hour, hour_rent_cnt, cluster 컬럼이 필요합니다.")
    else:
        st.info("시간대별정보_kepler_cluster.csv 파일이 있으면 군집별 시간대 활성도 그래프를 표시합니다.")


source_df = load_source_data()
raw_df = load_base_data(source_df)

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚲 서울 공공자전거")
st.sidebar.markdown("### 군집 분석 대시보드")
st.sidebar.divider()

st.sidebar.markdown(
    f"<small style='color:#64748b'>ver. {APP_VERSION[:20]}…</small>",
    unsafe_allow_html=True,
)
st.sidebar.caption(f"기준 파일: {SOURCE_FILE}")
st.sidebar.divider()

st.sidebar.markdown("**데이터 현황**")
st.sidebar.caption(f"원본 전체 대여소: **{len(source_df):,}**")
st.sidebar.caption(f"7일 미만: {int((source_df['cnt_day'] < 7).sum()):,}")
st.sidebar.caption(f"14일 미만: {int((source_df['cnt_day'] < 14).sum()):,}")
st.sidebar.caption(f"30일 미만: {int((source_df['cnt_day'] < 30).sum()):,}")
st.sidebar.caption(f"60일 미만: {int((source_df['cnt_day'] < 60).sum()):,}")
st.sidebar.caption(f"분석 적용(≥30일): **{len(raw_df):,}**")
st.sidebar.divider()

page = st.sidebar.radio(
    "페이지 선택",
    [
        "프로젝트 개요",
        "전체 현황",
        "군집 지도",
        "군집 특성 비교",
        "대여소 상세 조회",
        "운영 전략 추천",
        "피드백 대응 검증",
        "날씨·교통 활용",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("공통 필터")

df = raw_df.copy()

if "cluster_type" in df.columns:
    types = sorted(df["cluster_type"].dropna().unique())
    selected = st.sidebar.multiselect("군집 유형", types, default=types)

    if selected:
        df = df[df["cluster_type"].isin(selected)].copy()

if "district_name" in df.columns:
    districts = sorted(df["district_name"].dropna().unique())
    selected_d = st.sidebar.multiselect("자치구", districts, default=[])

    if selected_d:
        df = df[df["district_name"].isin(selected_d)].copy()

st.sidebar.caption(f"현재 데이터: **{len(df):,}**건")
# ───────────────────────────────────────────────────────────────────────────

if page == "프로젝트 개요":
    page_intro(df)
elif page == "전체 현황":
    page_overview(df)
elif page == "군집 지도":
    page_map(df)
elif page == "군집 특성 비교":
    page_cluster_compare(df)
elif page == "대여소 상세 조회":
    page_station_detail(df)
elif page == "운영 전략 추천":
    page_strategy(df)
elif page == "피드백 대응 검증":
    page_validation(source_df)
elif page == "날씨·교통 활용":
    page_external(df)
