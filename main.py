import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

FEATURES = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}
RAW_FEATURES = ["first_scrn", "total_audi", "days_in_top10", "long_run_index"]
RAW_LABELS = {
    "first_scrn": "스크린 수",
    "total_audi": "누적 관객",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수",
}

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 특성을 바탕으로 K-평균 알고리즘이 비슷한 영화끼리 세 유형으로 묶습니다.")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    for col in ["first_scrn", "total_audi", "first_week_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log_first_scrn"] = df["first_scrn"].where(df["first_scrn"] > 0).map(
        lambda x: __import__("math").log10(x) if pd.notna(x) else float("nan")
    )
    df["log_total_audi"] = df["total_audi"].where(df["total_audi"] > 0).map(
        lambda x: __import__("math").log10(x) if pd.notna(x) else float("nan")
    )
    # 롱런 지수는 누적 관객 / 첫 주 관객이며 20을 상한으로 둡니다.
    df["long_run_index"] = (df["total_audi"] / df["first_week_audi"].where(df["first_week_audi"] > 0)).clip(upper=20)
    return df

try:
    data = load_data()
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 URL을 확인해 주세요. ({exc})")
    st.stop()

# 네 속성 모두 값이 있고 첫 주 관객이 0보다 큰 영화만 분석합니다.
valid = data.dropna(subset=RAW_FEATURES + ["first_week_audi"]).copy()
valid = valid[(valid["first_week_audi"] > 0) & (valid["first_scrn"] > 0) & (valid["total_audi"] > 0)]
st.write(f"전체 영화 **{len(data):,}편** · 분석에 사용한 영화 **{len(valid):,}편**")

st.subheader("분석 속성 선택")
feature_names = list(FEATURES.keys())
selected_names = st.multiselect(
    "군집을 만드는 데 사용할 속성 (둘 이상 선택)",
    options=feature_names,
    default=feature_names,
)
if len(selected_names) < 2:
    st.warning("군집 분석을 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()

selected_cols = [FEATURES[name] for name in selected_names]
if len(valid) < 3:
    st.error("세 군집을 만들기에 유효한 영화 데이터가 부족합니다.")
    st.stop()

scaler = StandardScaler()
X = scaler.fit_transform(valid[selected_cols])
model = KMeans(n_clusters=3, random_state=42, n_init=10)
valid["cluster_id"] = model.fit_predict(X)

# 군집 라벨은 원래 단위 누적 관객 평균이 큰 군집부터 ㉮, ㉯, ㉰로 정합니다.
cluster_order = (
    valid.groupby("cluster_id")["total_audi"].mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_labels = {cluster_id: chr(ord("㉮") + i) for i, cluster_id in enumerate(cluster_order)}
valid["군집"] = valid["cluster_id"].map(cluster_labels)
cluster_color_order = [cluster_labels[c] for c in cluster_order]

st.subheader("2차원 산점도")
c1, c2 = st.columns(2)
with c1:
    x_name = st.selectbox("가로축 속성", feature_names, index=0, key="2d_x")
with c2:
    y_default = 1 if len(feature_names) > 1 else 0
    y_name = st.selectbox("세로축 속성", feature_names, index=y_default, key="2d_y")

fig2 = px.scatter(
    valid, x=FEATURES[x_name], y=FEATURES[y_name], color="군집",
    hover_name="movieNm", hover_data={"군집": True, "movieNm": False},
    category_orders={"군집": cluster_color_order},
    labels={FEATURES[x_name]: x_name, FEATURES[y_name]: y_name, "군집": "군집"},
    title="영화별 특성 분포",
)
fig2.update_traces(marker={"size": 7, "opacity": 0.8})
fig2.update_layout(legend_title_text="군집")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3차원 산점도")
if len(selected_names) < 3:
    st.info("3차원 산점도를 보려면 위에서 군집 분석 속성을 세 개 이상 선택해 주세요.")
else:
    d1, d2, d3 = st.columns(3)
    with d1:
        x3_name = st.selectbox("X축 속성", selected_names, index=0, key="3d_x")
    with d2:
        y3_name = st.selectbox("Y축 속성", selected_names, index=1, key="3d_y")
    with d3:
        z3_name = st.selectbox("Z축 속성", selected_names, index=2, key="3d_z")
    fig3 = px.scatter_3d(
        valid, x=FEATURES[x3_name], y=FEATURES[y3_name], z=FEATURES[z3_name],
        color="군집", hover_name="movieNm",
        hover_data={"군집": True, "movieNm": False},
        category_orders={"군집": cluster_color_order},
        labels={
            FEATURES[x3_name]: x3_name,
            FEATURES[y3_name]: y3_name,
            FEATURES[z3_name]: z3_name,
            "군집": "군집",
        },
        title="영화별 3차원 특성 분포",
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="군집")
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("군집별 요약")
summary = valid.groupby("군집").agg(
    편수=("movieNm", "size"),
    스크린수_평균=("first_scrn", "mean"),
    누적관객_평균=("total_audi", "mean"),
    **{"10위권 일수_평균": ("days_in_top10", "mean"), "롱런 지수_평균": ("long_run_index", "mean")},
).reindex(cluster_color_order)
summary = summary.rename(columns={
    "스크린수_평균": "스크린 수 평균",
    "누적관객_평균": "누적 관객 평균",
    "10위권 일수_평균": "10위권 일수 평균",
    "롱런 지수_평균": "롱런 지수 평균",
})
summary["편수"] = summary["편수"].astype(int)
summary["스크린 수 평균"] = summary["스크린 수 평균"].round(1)
summary["누적 관객 평균"] = summary["누적 관객 평균"].round(0).astype("int64")
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].round(2)
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].round(2)
st.dataframe(summary, use_container_width=True)

st.subheader("군집별 누적 관객 상위 영화")
for label in cluster_color_order:
    with st.expander(f"군집 {label} — 누적 관객 상위 5편", expanded=True):
        top5 = (
            valid[valid["군집"] == label]
            .sort_values("total_audi", ascending=False)
            .head(5)[["movieNm", "total_audi"]]
            .rename(columns={"movieNm": "영화 제목", "total_audi": "누적 관객"})
            .reset_index(drop=True)
        )
        top5.index = top5.index + 1
        top5["누적 관객"] = top5["누적 관객"].map(lambda x: f"{x:,.0f}")
        st.table(top5)
