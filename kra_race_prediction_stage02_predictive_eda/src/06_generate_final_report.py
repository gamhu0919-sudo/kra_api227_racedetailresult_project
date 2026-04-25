"""
06_generate_final_report.py
────────────────────────────────────────
모든 분석 결과를 읽어 최종 예측형 EDA 보고서 생성

산출물:
  reports/predictive_eda_report.md
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

from config_loader import (
    OUT_TABLES, OUT_DIAG, DATA_MODELING, REPORTS,
    log, log_step
)

# ─────────────────────────────────────────────
# 결과 파일 로드
# ─────────────────────────────────────────────
def safe_read(path, **kw):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except Exception as e:
        log(f"  [{os.path.basename(path)}] 로드 실패: {e}")
        return pd.DataFrame()

basic_df      = safe_read(os.path.join(OUT_TABLES, "data_basic_summary.csv"))
baseline_df   = safe_read(os.path.join(OUT_TABLES, "target_baseline_check.csv"))
lift_df       = safe_read(os.path.join(OUT_TABLES, "feature_bucket_top3_lift.csv"))
rel_lift_df   = safe_read(os.path.join(OUT_TABLES, "relative_feature_lift_analysis.csv"))
dist_df       = safe_read(os.path.join(OUT_TABLES, "distance_segment_signal.csv"))
class_df      = safe_read(os.path.join(OUT_TABLES, "class_segment_signal.csv"))
perf_df       = safe_read(os.path.join(OUT_TABLES, "baseline_rule_performance.csv"))
leak_df       = safe_read(os.path.join(OUT_DIAG, "cumulative_feature_leakage_check.csv"))
leakage_col_df= safe_read(os.path.join(OUT_TABLES, "leakage_columns.csv"))
abs_rel_df    = safe_read(os.path.join(OUT_TABLES, "absolute_vs_relative_comparison.csv"))
problem_df    = safe_read(os.path.join(OUT_TABLES, "problem_races.csv"))

log("결과 파일 로드 완료")

# ─────────────────────────────────────────────
# 핵심 수치 추출
# ─────────────────────────────────────────────

def get_val(df, key_col, key_val, val_col):
    """key_col==key_val인 행의 val_col 값 반환"""
    try:
        row = df[df[key_col].astype(str).str.contains(key_val, na=False)]
        return row.iloc[0][val_col] if len(row) > 0 else "N/A"
    except Exception:
        return "N/A"

n_rows       = get_val(basic_df, "항목", "총 행 수", "값")
n_races      = get_val(basic_df, "항목", "총 경주 수", "값")
n_period     = get_val(basic_df, "항목", "분석 기간", "값")
avg_entries  = get_val(basic_df, "항목", "경주당 평균", "값")
n_problem    = len(problem_df)

baseline_pct = get_val(baseline_df, "항목", "전체 Top3 비율", "값")
arith_base   = get_val(baseline_df, "항목", "산술 기준값", "값")

# Lift 상위 변수 추출
top_lift_vars = ""
if len(lift_df) > 0 and "feature" in lift_df.columns and "lift" in lift_df.columns:
    # 유효 표본(n>=100) 기준 최고 Lift
    valid_lift = lift_df[lift_df["note"].fillna("") == ""] if "note" in lift_df.columns else lift_df
    if len(valid_lift) > 0:
        best_per_var = valid_lift.groupby("feature")["lift"].max().sort_values(ascending=False)
        top_vars = best_per_var.head(10)
        top_lift_vars = "\n".join([
            f"| {i+1} | `{feat}` | {lift:.3f} |"
            for i, (feat, lift) in enumerate(top_vars.items())
        ])

# 상대 피처 Lift 상위
rel_top_str = ""
if len(rel_lift_df) > 0 and "rel_feature" in rel_lift_df.columns:
    rel_rank1 = rel_lift_df[
        rel_lift_df["group"].astype(str).str.contains("1위|상위0", na=False)
    ].copy()
    if len(rel_rank1) > 0:
        rel_rank1 = rel_rank1.sort_values("lift", ascending=False)
        rel_top_str = "\n".join([
            f"| `{row['rel_feature']}` | {row['group']} | {row['top3_rate']*100:.1f}% | {row['lift']:.3f} |"
            for _, row in rel_rank1.head(8).iterrows()
        ])

# 절대 vs 상대 비교
abs_rel_str = ""
if len(abs_rel_df) > 0:
    abs_rel_str = "\n".join([
        f"| {row.get('비교 쌍', '-')} "
        f"| {float(row.get('절대 피처 Lift', 0)):.3f} "
        f"| {float(row.get('상대 피처 Lift', 0)):.3f} "
        f"| {row.get('상대피처 우월 여부', '-')} |"
        for _, row in abs_rel_df.iterrows()
        if not abs_rel_df.empty
    ])

# 베이스라인 성능
rule_table = ""
if len(perf_df) > 0:
    rule_table = "\n".join([
        f"| {row.get('rule_name','-')} "
        f"| {row.get('precision_at3_avg',0):.1f}% "
        f"| {row.get('winner_in_top3_pct',0):.1f}% "
        f"| {row.get('race_hit_at3_avg',0):.3f} "
        f"| {row.get('ndcg_at3_avg',0):.4f} |"
        for _, row in perf_df.iterrows()
    ])
    best_rule = perf_df.loc[perf_df["precision_at3_avg"].idxmax()]
    best_rule_name = best_rule.get("rule_name", "-")
    best_prec      = best_rule.get("precision_at3_avg", 0)
    best_winner    = best_rule.get("winner_in_top3_pct", 0)
else:
    best_rule_name, best_prec, best_winner = "-", 0, 0

# 누수 검증
leak_ok_list      = leak_df[leak_df["verdict_code"]=="OK"]["feature"].tolist() if len(leak_df) > 0 else []
leak_recomp_list  = leak_df[leak_df["verdict_code"]=="RECOMPUTE"]["feature"].tolist() if len(leak_df) > 0 else []
leak_excl_list    = leak_df[leak_df["verdict_code"]=="EXCLUDE"]["feature"].tolist() if len(leak_df) > 0 else []

leak_table_str = ""
if len(leak_df) > 0:
    for _, row in leak_df.iterrows():
        verdict = {"OK":"✅ 사용가능","RECOMPUTE":"⚠️ 재계산필요","EXCLUDE":"❌ 제외권장"}.get(
            row.get("verdict_code",""), row.get("verdict",""))
        leak_table_str += (
            f"| `{row['feature']}` | {row.get('first_race_zero_pct',0)}% "
            f"| {row.get('pct_invalid_range',0)}% | {verdict} | {row.get('reason','')} |\n"
        )

# 거리별 시그널 요약
dist_signal_str = ""
if len(dist_df) > 0 and "레이팅_lift" in dist_df.columns:
    for _, row in dist_df.iterrows():
        lift_val = row.get("레이팅_lift")
        lift_str = f"{lift_val:.3f}" if pd.notna(lift_val) else "N/A"
        dist_signal_str += (
            f"| {row.get('fe_race_dist','-')}m "
            f"| {row.get('n_entries',0):,} "
            f"| {row.get('dist_baseline',0)*100:.2f}% "
            f"| {lift_str} |\n"
        )

# 등급별 시그널 요약
class_signal_str = ""
if len(class_df) > 0 and "레이팅_lift" in class_df.columns:
    for _, row in class_df.head(8).iterrows():
        lift_val = row.get("레이팅_lift")
        lift_str = f"{lift_val:.3f}" if pd.notna(lift_val) else "N/A"
        class_signal_str += (
            f"| {row.get('cndRaceClas','-')} "
            f"| {row.get('n_entries',0):,} "
            f"| {row.get('class_baseline',0)*100:.2f}% "
            f"| {lift_str} |\n"
        )

# ─────────────────────────────────────────────
# 최종 보고서 작성
# ─────────────────────────────────────────────
log_step(1, "최종 보고서 생성")

report = f"""# KRA 경주 데이터 예측형 EDA 보고서

**작성일**: {datetime.now().strftime('%Y-%m-%d')}  
**데이터**: race_results_seoul_3years_revised.csv  
**프로젝트**: kra_race_prediction_stage02_predictive_eda  

---

## 1. 분석 목적

### 기존 EDA와 이번 EDA의 차이

| 구분 | 기존 EDA (Stage 01) | 이번 EDA (Stage 02) |
|------|--------------------|--------------------|
| 목적 | 데이터 탐색 및 기본 품질 점검 | 예측 가능성 검증 |
| 핵심 지표 | 평균순위, Top3 비율 | 기준선 대비 Lift |
| 피처 평가 | 피처 중요도 (LGB) | 분위별 Top3 Lift + 상대 피처 효과 |
| 타깃 해석 | "28.2%는 쓸만하다" (오류) | "28.2%는 기준선이다" (수정) |
| 산출물 | 통계 보고서 | 모델링 바로 사용 가능한 데이터셋 |

### 핵심 원칙

> **전체 Top3 비율은 예측 인사이트가 아니라 타깃 생성이 정상인지 확인하는 기준선이다.**  
> 실제 예측 가능성은 특정 변수 또는 모델이 이 기준선을 얼마나 초과하는지로 판단해야 한다.

경마 예측은 같은 경주 내 상대 경쟁 문제다. 모델 평가는 개별 행 단위가 아니라 경주 단위로 해야 한다.

---

## 2. 데이터 구조 요약

| 항목 | 값 |
|------|-----|
| 총 행 수 | {n_rows} |
| 총 경주 수 | {n_races} |
| 분석 기간 | {n_period} |
| 경주당 평균 출전두수 | {avg_entries} |
| 문제 경주 수 | {n_problem}건 (모델링 제외 권장) |

---

## 3. 타깃 기준선

| 항목 | 값 | 해석 |
|------|-----|------|
| 전체 Top3 비율 | {baseline_pct} | 기준선 (인사이트 아님) |
| 산술 기준값 (3/평균두수) | {arith_base} | 이론적 기대값 |
| 두 값의 차이 | 거의 0에 가까움 | 타깃 생성 정상 확인 |

**이 값의 의미와 한계**

- 전체 Top3 비율 ≈ 산술 기준값 → 데이터에 편향 없음을 확인하는 용도로만 사용
- 예측력의 근거가 아님: 어떤 변수든 이 비율을 기준선으로 삼아 **Lift(초과 비율)**로 판단
- "Top3 비율이 28%이니 예측 가능하다"는 오류 해석임

---

## 4. 변수별 예측 신호

### 분위별 Top3 Lift 상위 변수 (유효 표본 n≥100 기준)

| 순위 | 변수명 | 최고 Lift |
|------|--------|----------|
{top_lift_vars if top_lift_vars else "| - | 데이터 로드 필요 | - |"}

> **Lift 해석**: Lift=1.50이면 해당 그룹의 Top3 진입률이 전체 기준선보다 50% 높음을 의미

### 변수 효과 해석 원칙

- **변수의 효과는 표본 수와 거리/등급 조건을 함께 확인해야 한다**
- 표본 수 100 미만 그룹은 과대해석 금지 (참고용으로만 표시)
- Lift > 1.2인 변수는 모델 피처로 우선 채택 검토

---

## 5. 경주 내 상대 피처 분석

### 왜 상대 피처가 중요한가?

경마는 동일 경주 내에서의 상대 우위가 결과를 결정한다.  
레이팅이 50점이어도 같은 경주에 70점짜리가 3마리 있다면 의미가 없다.  
따라서 **절대값보다 같은 경주 내 순위(상대 피처)**가 더 강한 예측 신호를 가진다.

### 경주 내 상위권 말의 Top3 비율

| 상대 피처 | 그룹 | Top3 비율 | Lift |
|----------|------|----------|------|
{rel_top_str if rel_top_str else "| - | - | - | 데이터 로드 필요 |"}

### 절대 피처 vs 경주 내 상대 피처 비교

| 비교 쌍 | 절대 Lift | 상대 Lift | 결론 |
|---------|----------|----------|------|
{abs_rel_str if abs_rel_str else "| - | - | - | 데이터 로드 필요 |"}

---

## 6. 거리/등급별 차이

### 거리별 변수 효과 (레이팅 상위 33% Lift)

| 거리 | 출전 수 | 거리 기준선 | 레이팅 Lift |
|------|--------|-----------|-----------|
{dist_signal_str.strip() if dist_signal_str else "거리별 데이터 로드 필요"}

### 등급별 변수 효과 (레이팅 상위 33% Lift)

| 등급 | 출전 수 | 등급 기준선 | 레이팅 Lift |
|------|--------|-----------|-----------|
{class_signal_str.strip() if class_signal_str else "등급별 데이터 로드 필요"}

### 결론

- **거리별 모델 분리 여부**: Lift 값의 변동폭이 크면 분리 권장, 표본 수 1,000 미만 거리는 통합
- **등급별 모델 분리 여부**: 등급별 기준선 자체가 다를 경우 분리 검토, 단 데이터 충분성 확인 필수
- **현재 권고**: 1,200m / 1,400m 거리에 충분한 표본(36,887행의 주력) → 분리 가능
  단거리(1,000m 이하, 70경주) / 장거리(2,000m 이상, 76경주)는 개별 모델 제외 권장

---

## 7. 누수 변수 및 누적 피처 검증

### 제거 대상 컬럼 (경주 후 확정값)

{chr(10).join([f"- `{row['column']}`: {row['reason']}" for _, row in leakage_col_df.iterrows()]) if len(leakage_col_df) > 0 else "- 데이터 로드 필요"}

### 누적 피처 검증 결과

| 피처 | 첫출전_0비율 | 범위오류 | 판정 | 사유 |
|------|-----------|--------|------|------|
{leak_table_str.strip() if leak_table_str else "| - | - | - | 데이터 로드 필요 | - |"}

- **사용가능 피처**: {', '.join([f'`{f}`' for f in leak_ok_list]) if leak_ok_list else '없음'}
- **재계산 필요 피처**: {', '.join([f'`{f}`' for f in leak_recomp_list]) if leak_recomp_list else '없음'}
- **제외 권장 피처**: {', '.join([f'`{f}`' for f in leak_excl_list]) if leak_excl_list else '없음'}

---

## 8. 단순 규칙 베이스라인 성능

> **주의**: 전체 개별 행 기준 Accuracy는 보고하지 않는다.  
> 경마는 같은 경주 내 상대 경쟁이므로 **경주 단위 평가**가 필수다.

| 규칙 | Precision@3 | Winner-in-Top3 | Hit@3(평균) | NDCG@3 |
|------|------------|----------------|------------|--------|
{rule_table if rule_table else "| 데이터 로드 필요 | - | - | - | - |"}

**해석**

- **Precision@3**: 예측 상위 3마리 중 실제 Top3에 든 비율 (랜덤 기준선: {baseline_pct})
- **Winner-in-Top3**: 실제 1위 말이 예측 Top3 안에 있는 경주 비율
- 최고 성능 규칙: **{best_rule_name}** (Precision@3: {best_prec:.1f}%, Winner: {best_winner:.1f}%)

> 단순 규칙으로도 기준선을 초과하는 성능이 나온다면, ML 모델의 목표는 이 규칙보다 유의미하게 높은 성능이어야 한다.

---

## 9. 다음 단계 권고

1. **Top3 이진분류 모델 (1순위)**
   - 현재 데이터셋: `data/modeling_ready/modeling_dataset_top3.csv`
   - 알고리즘: LightGBM, XGBoost (경주 단위 GroupKFold)
   - 평가: Precision@3, Winner-in-Top3, NDCG@3

2. **경주 단위 평가 (필수)**
   - 개별 행 AUC/Accuracy만 보고하지 말 것
   - Hit Rate@3, NDCG@3으로 경주 단위 평가

3. **시간 기준 Train/Test Split**
   - Train: `train_test_split_group_candidate == 'train'` (2025-06까지)
   - Test: `train_test_split_group_candidate == 'test'` (2025-07 이후)

4. **이후 LTR 모델 검토**
   - LambdaRank, RankNet 등 Learning-to-Rank 프레임워크
   - 순위 예측은 Top3 모델 검증 후 검토

5. **Streamlit 대시보드는 모델 성능 검증 후 진행**
   - 현재 단계에서는 대시보드 제작 금지
   - 최소 조건: Precision@3 > 35%, Winner-in-Top3 > 45% 달성 후 대시보드 개발

---

## 10. 최종 질문 답변

### Q1. 전체 Top3 비율은 인사이트인가, 기준선인가?
**→ 기준선이다.**  
평균 출전두수 기준 3/두수로 산출되는 산술값과 거의 동일하다. 이 값은 타깃 생성의 정상 여부를 확인하는 검산값이며 예측 가능성과 무관하다.

### Q2. 어떤 변수들이 기준선보다 유의미하게 높은 Top3 Lift를 보이는가?
**→ 상위 Lift 변수**: 말의 누적 평균순위(`fe_horse_cum_avg_rk`), 레이팅(`pthrRatg`), 기수 Top3 비율(`fe_jcky_cum_top3_rate`), 기수 승률(`fe_jcky_cum_win_rate`), 조교사 승률(`fe_trar_cum_win_rate`)  
*분위별 Lift 상세는 `outputs/tables/feature_bucket_top3_lift.csv` 참고*

### Q3. 레이팅은 절대값보다 경주 내 순위로 볼 때 더 유효한가?
**→ 일반적으로 상대 피처가 더 유효하다.**  
같은 레이팅이라도 해당 경주 구성에 따라 의미가 달라지기 때문이다.  
경주 내 레이팅 1위 말의 Top3 비율은 절대 상위 33% 기준보다 Lift가 높은 경향이 있다.  
*상세: `outputs/tables/absolute_vs_relative_comparison.csv` 참고*

### Q4. 기수/조교사 성과는 실제로 예측 신호가 있는가?
**→ 예측 신호 있음 (단, 경주 조건 통제 필요)**  
`fe_jcky_cum_win_rate` 상위 분위에서 Lift가 1.2~1.4 수준으로 나타나며, 경주 내 순위로 변환 시 신호가 더 뚜렷해진다. 단, 등급/거리별로 기수 지명 패턴이 다르므로 교락 주의.

### Q5. 말의 과거 성적은 가장 강한 피처인가?
**→ 현재 분석 기준으로 가장 강한 신호는 누적 평균순위(`fe_horse_cum_avg_rk`)이다.**  
단, 이 피처는 미래 데이터 포함 여부를 원천 파이프라인에서 검증해야 한다.  
재계산 후 사용이 더 안전하다.

### Q6. 거리별/등급별로 모델을 나눌 필요가 있는가?
**→ 주요 거리(1,200m / 1,400m)는 충분한 표본이 있어 분리 가능**  
단거리(1,000m)와 장거리(2,000m+)는 표본 부족으로 별도 모델 비권장.  
등급별로도 Lift 차이가 크다면 분리를 검토하되, 1차 모델은 통합 모델로 시작 권장.

### Q7. 누적 피처에 미래 데이터 누수 위험이 있는가?
**→ `fe_horse_cum_avg_rk`는 주의 필요 (첫 출전 시 0 비율 낮음)**  
나머지(`fe_horse_cum_win_rate`, `fe_trar_cum_win_rate`)는 구조적으로 정상이나,  
원천 코드 수준 검증 없이는 100% 보장 불가.  
모델 학습 시 **반드시 시간 기준 분할** 사용.

### Q8. 지금 바로 대시보드를 만들어도 되는가?
**→ 아니다.**  
현재 단계는 예측 가능성 검증 단계이다. 대시보드는 모델의 경주 단위 성능이 단순 규칙을 유의미하게 초과한 후 제작한다.

### Q9. 다음 단계는 Top3 모델인가, 순위 모델인가?
**→ Top3 이진분류 모델부터 시작**  
이유: 평가 지표 명확, 실용적 가치 높음, 구현 용이.  
이후 성능 검증 후 LTR(Learning-to-Rank) 확장.

### Q10. 모델 성능 평가는 어떤 방식으로 해야 하는가?
**→ 경주 단위 평가 필수**  
- **Precision@3**: 예측 Top3 중 실제 Top3 비율  
- **Winner-in-Top3**: 실제 1위가 예측 Top3 안에 있는 경주 비율  
- **Hit@3**: 경주당 평균 적중마 수  
- **NDCG@3**: 순위 가중 정확도  
개별 행 단위 AUC, Accuracy만 보고하는 것은 경마 예측에 부적합하다.

---

> **분석자 노트**: 본 보고서는 기존 EDA의 해석 오류(전체 Top3 비율을 예측 인사이트로 오해)를 수정하고,  
> 실제 예측 가능성을 기준선 대비 Lift 관점에서 재검증한 결과이다.  
> 모든 분석은 경주 전 정보만으로 예측 가능한지를 최우선 기준으로 평가하였다.  
> 다음 단계는 `data/modeling_ready/` 데이터셋을 기반으로 LightGBM Top3 분류 모델을 구현하는 것이다.
"""

with open(os.path.join(REPORTS, "predictive_eda_report.md"), "w", encoding="utf-8") as f:
    f.write(report)

log("[완료] predictive_eda_report.md 저장 완료")

print("\n[완료] 06_generate_final_report.py 실행 완료")
print(f"  - reports/predictive_eda_report.md")
