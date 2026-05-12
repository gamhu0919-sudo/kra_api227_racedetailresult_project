# KRA 경주마 Top3 예측 플랫폼

본 프로젝트는 한국마사회(KRA) 상세 성적 데이터를 기반으로 경주마의 Top3(1, 2, 3위) 진입 가능성을 분석하는 머신러닝 프로토타입입니다. 현재 통합 대시보드는 **Stage03 model upgrade v2 산출물**을 우선 사용하며, **Stage06 Walk-forward 검증 탭**을 포함합니다.

> ⚠️ 본 프로젝트는 분석용 프로토타입이며 실제 경주 결과를 보장하지 않습니다. 도박 또는 베팅 권유 목적이 아닙니다.

## 🚀 주요 단계

1. **Stage03 baseline / v2 모델링**
   - LightGBM 기반 Top3 분류 모델 및 비교 성능표 생성
   - v2 대시보드 기본 성능표: `kra_race_prediction_stage03_model_upgrade_v2/outputs/tables/model_v2_comparison_table.csv`
2. **Stage04 Streamlit 대시보드**
   - v2 모델 성능, 경주별 예측, 변수 중요도, 오류 분석, 파일 상태 점검 표시
3. **Stage05 미래 경주 추론 파이프라인**
   - `next_race_predictions.csv` 생성 후 대시보드 미래 예측 모드에서 표시
4. **Stage06 Walk-forward 검증**
   - 월별 누적 학습/검증 결과를 별도 탭에서 표시
   - 테스트 월 이전 데이터만 학습에 사용해 미래 데이터 누수를 방지하는 검증 방식

## 🛠️ 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Streamlit 대시보드 실행

```bash
streamlit run streamlit_app.py
```

Streamlit Cloud 배포 시 main file은 다음으로 지정합니다.

```text
streamlit_app.py
```

### 3. 미래 경주 예측 파일 생성

미래 경주 예측 탭을 보려면 먼저 Stage05 추론 파이프라인을 실행합니다.

```bash
python run_inference.py
```

생성/사용 파일:

```text
kra_race_prediction_stage05_inference_pipeline/data/output/next_race_predictions.csv
```

### 4. Stage06 Walk-forward 재검증

```bash
python kra_race_prediction_stage06_walk_forward_backtest/run_walk_forward.py
```

주요 결과 파일:

```text
kra_race_prediction_stage06_walk_forward_backtest/outputs/metrics/walk_forward_monthly_metrics.csv
kra_race_prediction_stage06_walk_forward_backtest/reports/walk_forward_report.md
```

## 📊 대시보드 탭 구성

1. 프로젝트 개요
2. 경주별 Top3 예측
3. 예측 근거/피처 설명
4. v2 모델 성능 요약
5. Baseline vs v2 비교
6. Walk-forward 검증
7. 변수 중요도
8. 오류 분석
9. 데이터/파일 상태 점검

각 탭은 필요한 CSV 또는 모델 파일이 누락되어도 앱 전체를 중단하지 않고, 해당 기능의 제한 사항을 안내합니다. 특히 v2 모델 pkl이나 `data/predictions/lgbm_top3_feature_v2_predictions.csv`가 없어도 `outputs/tables/walk_forward_race_predictions.csv`를 fallback으로 사용해 날짜별 과거 예측 결과를 확인할 수 있습니다.

## 📂 핵심 산출물

- v2 모델 파일: `kra_race_prediction_stage03_model_upgrade_v2/models/lgbm_top3_feature_v2.pkl`
- v2 모델링 데이터: `kra_race_prediction_stage03_model_upgrade_v2/data/modeling/modeling_data_v2_with_preds.csv`
- v2 예측 결과: `kra_race_prediction_stage03_model_upgrade_v2/data/predictions/lgbm_top3_feature_v2_predictions.csv`
- v2 성능 비교표: `kra_race_prediction_stage03_model_upgrade_v2/outputs/tables/model_v2_comparison_table.csv`
- v2 변수 중요도: `kra_race_prediction_stage03_model_upgrade_v2/outputs/tables/feature_importance_v2.csv`
- Stage05 미래 예측 결과: `kra_race_prediction_stage05_inference_pipeline/data/output/next_race_predictions.csv`
- Stage06 월별 Walk-forward 지표: `kra_race_prediction_stage06_walk_forward_backtest/outputs/metrics/walk_forward_monthly_metrics.csv`
- baseline 오류 분석: `kra_race_prediction_stage03_top3_modeling/outputs/tables/error_analysis_by_distance.csv`, `error_analysis_by_class.csv`

일부 파일이 저장소 또는 배포 환경에 없으면 관련 탭은 제한될 수 있습니다. 누락 파일과 영향 범위는 대시보드의 **데이터/파일 상태 점검** 탭에서 확인할 수 있습니다.

## ☁️ Streamlit Cloud 배포 메모

- Main file path: `streamlit_app.py`
- Python 런타임: `runtime.txt` (`python-3.11`)
- 서버 설정: `.streamlit/config.toml`
- 대용량 모델/CSV가 GitHub에 포함되지 않은 경우, 해당 파일이 없는 탭은 안내 메시지만 표시됩니다.
