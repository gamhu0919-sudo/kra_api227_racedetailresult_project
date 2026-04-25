# Stage 05: 다음 경주 입력 예측 파이프라인

본 단계는 다가올 미래 경주의 편성 정보를 입력받아, 학습된 LightGBM 모델과 동일한 피처를 생성하고 Top3 진입 확률을 산출하는 자동화 프로세스를 담당합니다.

## 📋 입력 데이터 요구사항 (next_race_entries.csv)

사용자는 아래 컬럼이 포함된 데이터를 `data/input/next_race_entries.csv` 위치에 준비해야 합니다.

### 필수 컬럼 (Base Info)
- `race_id`: 경주 식별자 (예: 20260425_1R)
- `schdRaceDt`: 경주 일자 (YYYY-MM-DD)
- `pthrHrno`: 출전번호 (문자열 권장)
- `pthrHrnm`: 마명
- `hrmJckyId`, `hrmTrarId`: 기수/조교사 식별 ID (문자열)
- `pthrBurdWgt`, `pthrRatg`: 부담중량, 레이팅

### 가변 기입 컬럼 (User Input)
- **`fe_horse_weight`**: 당일 측정 마체중
- **`rsutWetr`**: 기상 상태 ('맑음', '흐림', '비' 등)
- **`rsutTrckStus`**: 주로 상태 ('건조 (3%)', '양호 (6%)' 등)
- **`fe_track_humidity`**: 주로 함수율

---

## ⚙️ 실행 순서 (Pipeline Steps)

루트 디렉토리의 `run_inference.py`를 실행하면 아래 7단계가 순차적으로 처리됩니다.

1.  **`01_create_reference_tables.py`**: 과거 데이터에서 최신 스탯(승률 등) 추출
2.  **`02_create_next_race_template.py`**: 입력 양식(Template) 생성 및 샘플 데이터 전파
3.  **`03_validate_next_race_input.py`**: 입력 데이터의 무결성 및 Leakage 변수 검증
4.  **`04_build_inference_features.py`**: 기준 통계 Join 및 마체중 증감 계산
5.  **`05_make_relative_features.py`**: 경주 내 상대적 순위/점수(Rank, Z-score) 산출
6.  **`06_predict_next_race.py`**: 모델 피처 정합성 검토 및 확률 예측
7.  **`07_export_dashboard_output.py`**: 대시보드 연동용 최종 스키마 정제 및 요약 보고서 생성

---

## 🧪 주요 처리 로직

### Cold-start 처리
- 신규 말/기수/조교사의 경우 과거 통계값이 없으므로 기본값(0.0 또는 평균 수준)을 부여하며, 보고서에 해당 내역을 기록합니다.

### 마체중 증감
- `last_horse_weight` 정보가 있을 경우 현재 입력된 `fe_horse_weight`와 비교하여 `fe_weight_diff`를 생성합니다. 정보가 없는 경우 0으로 간주합니다.

### 모델 피처 정합성
- 추론 시점에 모델이 기대하는 피처 중 일부가 부족할 경우, 에러로 중단되지 않고 기본값으로 보완하여 안정성을 유지하며 해당 내역은 `reports/model_feature_alignment_report.md`에 남깁니다.

---

## 📊 결과 확인
최종 예측 결과는 `data/output/next_race_predictions.csv`에 저장되며, Stage 04 대시보드(Streamlit)의 **"🚀 실시간 미래 경주 예측"** 모드를 통해 즉시 시각화됩니다.
