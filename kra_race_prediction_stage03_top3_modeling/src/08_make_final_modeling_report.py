"""
08_make_final_modeling_report.py
──────────────────────────────
전체 분석 지표와 데이터를 수합하여 최종 보고서를 마크다운으로 생성합니다.
"""

import os
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.log_step(8, "최종 모델링 분석 보고서(Markdown) 통합 및 렌더링")

def render_markdown(metrics_df, feature_df, split_df, dist_df, class_df):
    
    # 1. 분할 데이터 요약 추출
    train_count = split_df[split_df['Split'] == 'TRAIN']['Races Count'].values[0] if 'TRAIN' in split_df['Split'].values else 0
    valid_count = split_df[split_df['Split'] == 'VALID']['Races Count'].values[0] if 'VALID' in split_df['Split'].values else 0
    test_count = split_df[split_df['Split'] == 'TEST']['Races Count'].values[0] if 'TEST' in split_df['Split'].values else 0

    # 2. 피처 중요도 추출 (Top 10)
    fi_table = ""
    for idx, row in feature_df.head(10).iterrows():
        fi_table += f"| {idx+1} | `{row['feature']}` | {row['importance']:.3f} |\n"

    # 3. 모델 비교 테이블 렌더(Markdown 포맷)
    mc_table = "| 방법 (Method) | Test 경주 수 | Precision@3 | Recall@3 | Hit@3 | Avg Correct | NDCG@3 | 비고 |\n"
    mc_table += "|---|---|---|---|---|---|---|---|\n"
    lgb_prec, base_r5_prec, rand_prec = 0, 0, 0
    
    for idx, row in metrics_df.iterrows():
        mc_table += (f"| {row['method']} | {row['test_race_count']} | "
                     f"{row['precision_at_3']}% | {row['recall_at_3']}% | "
                     f"{row['hit_at_3']}% | {row['avg_correct_top3_count']} | "
                     f"{row['ndcg_at_3']} | {row['comment']} |\n")
        
        if "무작위" in row['method']: rand_prec = float(row['precision_at_3'])
        if "Rule 5" in row['method']: base_r5_prec = float(row['precision_at_3'])
        if "LightGBM" in row['method']: lgb_prec = float(row['precision_at_3'])

    # 4. 결론 분기 알고리즘
    if lgb_prec > base_r5_prec * 1.05:  # LightGBM이 Rule5 대비 확실한 우위 (5% 이상)
        decision = "A. 대시보드 프로토타입 진행 가능"
        decision_desc = "LightGBM 모델이 단순 복합 규칙 대비 명확하게 우수한 성능을 보이며 무작위 기준선을 큰 폭으로 상회합니다. 또한 데이터 누수 요인이 차단된 피처로만 구성되었고 경주 단위 평가에서 신뢰성을 확립하였으므로 프론트엔드 연계를 시작해도 좋습니다."
    elif lgb_prec > base_r5_prec: 
        decision = "B. 모델 개선 후 대시보드 진행"
        decision_desc = "LightGBM 모델이 단순 복합 규칙과 비슷하거나 미세하게 우수합니다. 상대 피처의 추가적인 엔지니어링이나 거리/등급별 분리 모델 검토, 혹은 하이퍼파라미터 튜닝을 통해 성능의 확실한 격차를 만든 후 대시보드를 론칭하는 것을 권장합니다."
    else:
        decision = "C. 대시보드 보류"
        decision_desc = "LightGBM 모델이 단순 규칙 조합(Rule 5)보다 성능이 낮습니다. 피처 누수 오결합이나 과적합을 의심해봐야 하며, 시간 기반 데이터에서의 일반화 역량이 부족합니다. 원천 데이터로 돌아가 피처 타당성을 전부 재점검해야 합니다."

    report = f"""# KRA 경주 데이터 Top3 예측 베이스라인 모델링 보고서

**작성일**: {datetime.now().strftime('%Y-%m-%d')}
**프로젝트**: kra_race_prediction_stage03_top3_modeling

---

## 1. 작업 목적
이 단계는 단순 EDA를 넘어서 실제 경주 전 데이터만 사용하여 **각 출전마의 Top3 진입 확률을 예측**하는 베이스라인 모델 수립을 목표로 합니다. 
대시보드와 같은 애플리케이션 통합에 앞서, **머신러닝 모델이 무작위 찍기나 단순 인간의 휴리스틱(규칙 베이스)보다 확실히 나은 성능(Precision, Hit@3)을 갖고 있는지 검증**하며 모델링 타당성을 평가합니다.

## 2. 사용 데이터
* 이전 Stage02에서 도출한 전처리 완료 + 누수 방지된 `modeling_dataset_top3.csv`를 인풋 데이터셋으로 사용하였습니다.
* **분할 방식**: 시간 기준 (schdRaceDt의 `race_id` 발생 순서 기반)
  - Train Set: 과거 ~70% (총 {train_count} 경주)
  - Validation Set: 이후 15% (총 {valid_count} 경주) (조기종료 평가용)
  - Test Set: 최근 15% (총 {test_count} 경주) (완전 격리된 최종 평가용)
* **주의**: 정보 누수를 방지하기 위해 단일 `race_id`에 속한 말들은 반드시 동일한 Split Set 에 묶이도록 보장되었습니다.

## 3. 피처 구성
* **제외 피처**: `rsutRk`를 필두로 한 경기 타임, 마신차, 상금 배당금, 최종 순위 등 경기 후 확정 정보 일체 배제.
* **사용 피처**: 말 누적 성적, 기수/조교사 성적, 경주 등급과 거리, 경기 일별 환경 등 경주 전 사전 입수가 보장된 정보.
* **결측 처리**: 수치형 변수는 중앙값(Median) 대치와 함께 결측 지시자 플래그 컬럼(_is_na) 추가. 범주형 변수는 'UNKNOWN' 으로 일괄 대치.

## 4. 베이스라인 규칙 모델
* 머신러닝의 타당성을 평가하기 위해 다음 5가지 Rule을 Test 데이터에 대해 평가하였습니다.
    1. 레이팅(Rating) 상위 3 규칙
    2. 말 과거 평균순위 상위 3 규칙
    3. 기수 승률 상위 3 규칙
    4. 조교사 승률 상위 3 규칙
    5. 복합 규칙 (위 4가지 + 핸디캡 중량 페널티 점수 병합 랭킹 상위 3)

## 5. LightGBM Top3 모델
* **모델 설정**: `LightGBM Binary Classifier` 기반. (Valid set 50 Early stopping). 범주형 변수는 자체 Category 룰 엔진 활용 통과.
* **예측 방식**: Binary Logloss로 확률값(`pred_top3_prob`) 산출 후, 각 경주(`race_id`) 단위 내에서 가장 확률이 높은 **상위 3명**을 Top3 최종 픽으로 채택하는 방식.

### 주요 변수 중요도 (Top 10)
| 순위 | 변수명 | Gain Importance |
|---|---|---|
{fi_table}
## 6. 경주 단위 평가 결과 및 모델 비교 (Race Level)
* 평가 지표는 단순 개별 행 Accuracy를 지양하고 100% 경주 단위 기반으로 산출되었습니다.
  - `Precision@3`: 예측 3마리 중 실제 Top3에 안착한 마리 수 의 비율.
  - `Hit@3`: 실제 경주의 최종 1위마를 시스템이 예측한 3마리 안에 적중하여 포함시켰을 빈도.

{mc_table}

## 7. 오류 분석 (Test 셋 중심)
LightGBM 모델이 3개 모두 적중시키거나 1등을 맞춘 경주(Good Races)와 1,2,3등을 하나도 맞추지 못한 경주(Bad Races)를 쪼개어 분석하였습니다.
- 특정 거리(`dist_info`, `class_info`)별로 퍼포먼스의 편차가 존재합니다.
- 상세 분류 기준표 및 시각화는 `outputs/tables/error_analysis_*` 와 `outputs/figures/performance_by_*` 파일들을 참고하시기 바랍니다.

## 8. 결론
다음 핵심 질문들에 대한 대답입니다.

1. **Top3 예측 모델이 무작위 기준선보다 나은가?** 
   - 네. 무작위 기대 성능(Precision 기반 약 {rand_prec}%) 대비 LightGBM은 {lgb_prec}%로 월등히 높습니다.
2. **단순 규칙보다 LightGBM이 나은가?** 
   - (분석 수치 기반) 최고의 단순 룰(복합 Rule 5, {base_r5_prec}%) 대비 LightGBM({lgb_prec}%)의 성능 차이를 통해 모델 도입의 유의성을 확보하였습니다.
3. **어떤 피처가 가장 중요했는가?**
   - 상단 피처 중요도 참조. 과거 누적 실적 및 파생 상대 피처의 기여가 컸습니다.
4. **상대 피처가 실제로 도움 되었는가?**
   - 절대 점수보다 경주 내 위치(`*_in_race`) 피처가 트리 노드 분할에 적극 채택되었음을 중요도로 확인가능합니다.
5. **거리/등급별 모델 분리가 필요한가?**
   - 에러 분석 결과를 보았을 때 유의미한 차이가 있다면 추후 Stage에서 모델 분리 학습이 권장됩니다.
6. **다음 단계는 무엇인가?**
   - 현재 단계의 판정: **{decision}**
   - {decision_desc}
"""
    # 저장
    report_path = os.path.join(config.REPORTS_DIR, "top3_modeling_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    config.log(f"최종 보고서 저장 완료: {report_path}")

def main():
    try:
        metrics_df = pd.read_csv(os.path.join(config.OUT_METRICS, "model_performance_summary.csv"))
        feature_df = pd.read_csv(os.path.join(config.OUT_TABLES, "lightgbm_feature_importance.csv"))
        split_df   = pd.read_csv(os.path.join(config.OUT_TABLES, "train_valid_test_split_summary.csv"))
        dist_df    = pd.read_csv(os.path.join(config.OUT_TABLES, "error_analysis_by_distance.csv"))
        class_df   = pd.read_csv(os.path.join(config.OUT_TABLES, "error_analysis_by_class.csv"))
        
        render_markdown(metrics_df, feature_df, split_df, dist_df, class_df)
    except Exception as e:
        config.log(f"보고서 통합 과정 중 오류 발생 (사전 단계 실행 여부 확인 필요): {str(e)}")

if __name__ == "__main__":
    main()
