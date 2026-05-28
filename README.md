# 서울 공공자전거 대여소 군집 분석 대시보드

서울 공공자전거(따릉이) 대여소의 이용 패턴을 K-means 군집화하고,  
군집별 공간 분포·시간대 특성·날씨 영향을 시각화하는 Streamlit 대시보드입니다.

## 대시보드 페이지 구성

| 페이지 | 내용 |
|---|---|
| 프로젝트 개요 | 군집 유형 정의 및 분석 목적 요약 |
| 전체 현황 | 군집별 대여소 수, 이용량, 변수 분포 |
| 군집 지도 | 서울 지도 위 군집 시각화 (필터 지원) |
| 군집 특성 비교 | Heatmap, Radar Chart, 알고리즘 비교 |
| 대여소 상세 조회 | 대여소 검색 및 운영 코멘트 |
| 운영 전략 추천 | 군집별 전략 및 우선순위 TOP 20 |
| 피드백 대응 검증 | Elbow, Silhouette, 민감도 분석, Kruskal-Wallis |
| 날씨·교통 활용 | 강수·기온·시간대별 이용량 분석 |

## 군집 유형

- 🚴 **여가형** - 주말·장시간·장거리 이용 비율 높음
- 🌆 **퇴근/중심지역** - 저녁 이용 비율 높음
- 🏘️ **출근/거주지역** - 아침 이용 비율 높음
- 🔥 **고수요 중심지역** - 일평균 이용량·피크 집중도 높음

## 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/<your-username>/seoul-bike-dashboard.git
cd seoul-bike-dashboard
```

### 2. 가상환경 생성 및 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 대시보드 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 으로 접속합니다.

## 프로젝트 구조

```
seoul-bike-dashboard/
├── app.py                               # Streamlit 대시보드 메인
├── requirements.txt                     # 의존 패키지 목록
├── README.md
└── data/
    ├── bike_rawdata_with_cluster_cntday.csv   # 대여소별 피처 + 군집 레이블 (필수)
    ├── station_master.csv                     # 대여소 위치(위경도) 정보
    ├── 시간대별정보_kepler_cluster.csv         # 시간대별 이용량 데이터
    ├── cluster_rain_summary.csv               # 강수 여부별 군집 이용량 요약
    ├── cluster_temp_summary.csv               # 기온 구간별 군집 이용량 요약
    ├── cluster_rainlevel_summary.csv          # 강수 강도별 군집 이용량 요약
    └── cluster_bad_weather_summary.csv        # 악천후 여부별 군집 이용량 요약
```

> **참고**: 원본 대여 이력 CSV(2025년 1~12월, 약 6GB)와 날씨 DB는 용량 문제로 포함되지 않습니다.  
> 대시보드 실행에는 `data/` 폴더 내 파일만 있으면 됩니다.

## 사용 데이터 출처

| 데이터 | 출처 |
|---|---|
| 서울 공공자전거 대여 이력 | [서울 열린데이터광장](https://data.seoul.go.kr) |
| 서울 기상 데이터 | 기상청 ASOS |
| 대여소 위치 정보 | 서울시 공공자전거 운영사 제공 |

## 주요 분석 피처

| 피처 | 설명 |
|---|---|
| `cnt_per_day` | 일평균 이용량 |
| `return_rate` | 같은 대여소 반납 비율 |
| `bw_rate` | 복귀(왕복) 이용 비율 |
| `weekend_rate` | 주말 이용 비율 |
| `avg_distance` | 평균 이동 거리 |
| `avg_used_time` | 평균 이용 시간 |
| `avg_user_age` | 평균 이용자 연령 |
| `morning_ratio` | 오전(6~9시) 이용 비율 |
| `evening_ratio` | 저녁(18~21시) 이용 비율 |
| `peak_ratio` | 피크 시간대 집중도 |
| `variance_norm` | 이용량 분산(정규화) |
| `entropy` | 시간대별 이용 분산도 |

## 요구 사양

- Python 3.11 이상
- 주요 패키지: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `scipy`, `scikit-posthocs`
