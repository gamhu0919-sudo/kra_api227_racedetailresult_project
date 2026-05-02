# Stage 06: Walk-forward Backtest

본 스테이지는 KRA 경주 예측 모델의 **월별 누적 학습(Walk-forward) 성능**을 검증하기 위한 독립적인 시스템입니다.

## 1. 목적
- 실제 운영 환경과 동일한 시계열 제약 조건 하에서 모델의 실전 성능을 평가합니다.
- 데이터가 누적됨에 따라 모델의 성능이 어떻게 변화하는지 정량적으로 분석합니다.
- 특정 기간의 성능 하락 여부를 파악하여 모델의 강건성을 검증합니다.

## 2. 검증 방식
1. **월별 루프**: 2024년 1월부터 최신 데이터까지 매월 반복합니다.
2. **동적 학습**: 테스트 월(M)의 성적을 예측하기 위해, M월 이전의 모든 데이터를 사용하여 매번 새 모델을 학습합니다.
3. **리크지 차단**: 피처 생성 시점(As-of)을 엄격히 관리하여 미래 데이터가 과거 예측에 섞이지 않도록 합니다.
4. **9개 핵심 지표**: Precision@3, Hit@3, NDCG@3 등을 월별로 산출합니다.

## 3. 실행 방법
프로젝트 루트 폴더에서 다음 명령을 실행합니다.

```bash
python kra_race_prediction_stage06_walk_forward_backtest/run_walk_forward.py
```

## 4. 주요 산출물
- **리포트**: `reports/walk_forward_report.md` (시각화 포함)
- **전체 예측 결과**: `data/predictions/walk_forward_predictions_all.csv`
- **월별 지표**: `outputs/metrics/walk_forward_monthly_metrics.csv`
- **시각화 차트**: `outputs/figures/` 폴더 내 PNG 파일들

## 5. 주의사항
- 본 스테이지는 기존 Stage 03, 04, 05의 파일을 수정하거나 덮어쓰지 않습니다.
- 실행 시 대용량 데이터 로드 및 반복 학습으로 인해 수 분 이상의 시간이 소요될 수 있습니다.
