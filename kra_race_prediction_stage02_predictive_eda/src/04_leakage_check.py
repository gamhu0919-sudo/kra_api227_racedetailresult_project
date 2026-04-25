"""
04_leakage_check.py
────────────────────────────────────────
누적 피처 미래 데이터 누수 검증

검증 대상:
  - fe_horse_cum_win_rate
  - fe_horse_cum_avg_rk
  - fe_jcky_cum_win_rate
  - fe_jcky_cum_top3_rate
  - fe_trar_cum_win_rate

검증 방법:
  1. 시간순 정렬 후 말/기수/조교사별 첫 출전 확인
  2. 첫 출전 시 누적값이 0인지 확인 (이전 이력 없이 시작)
  3. 당일 경주 결과가 누적값에 포함됐는지 의심 패턴 탐지
  4. 시간 흐름에 따라 누적값이 단조 증가/수렴하는지 확인
  5. 각 피처의 신뢰도 판정: "사용가능" / "재계산필요" / "제외권장"

산출물:
  outputs/diagnostics/cumulative_feature_leakage_check.csv
  reports/cumulative_feature_leakage_review.md
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config_loader import (
    DATA_PROCESSED, OUT_DIAG, REPORTS, log, log_step
)

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
processed_path = os.path.join(DATA_PROCESSED, "race_data_preprocessed.csv")
df = pd.read_csv(processed_path, encoding="utf-8-sig", low_memory=False)
df["schdRaceDt"] = pd.to_datetime(df["schdRaceDt"], errors="coerce")

# 시간순 정렬 (누수 검증은 전체 데이터 대상)
df_sorted = df.sort_values(["schdRaceDt", "race_id"]).reset_index(drop=True)
log(f"데이터 로드: {len(df_sorted):,}행 / 기간: {df_sorted['schdRaceDt'].min().date()} ~ {df_sorted['schdRaceDt'].max().date()}")

# ─────────────────────────────────────────────
# 검증 대상 정의
# ─────────────────────────────────────────────
CUM_FEATURES = [
    {
        "col":        "fe_horse_cum_win_rate",
        "entity_col": "pthrHrno",
        "entity_nm":  "말",
        "result_col": "target_is_top3",  # is_top3를 win으로 처리
        "direction":  "높을수록 좋음",
        "valid_range": (0.0, 1.0),
        "first_should_be_zero": True,
    },
    {
        "col":        "fe_horse_cum_avg_rk",
        "entity_col": "pthrHrno",
        "entity_nm":  "말",
        "result_col": "target_rank",
        "direction":  "낮을수록 좋음 (평균순위 낮음=좋음)",
        "valid_range": (1.0, 30.0),
        "first_should_be_zero": False,  # 평균순위는 0이 아님
    },
    {
        "col":        "fe_jcky_cum_win_rate",
        "entity_col": "hrmJckyId",
        "entity_nm":  "기수",
        "result_col": "target_is_top3",
        "direction":  "높을수록 좋음",
        "valid_range": (0.0, 1.0),
        "first_should_be_zero": True,
    },
    {
        "col":        "fe_jcky_cum_top3_rate",
        "entity_col": "hrmJckyId",
        "entity_nm":  "기수",
        "result_col": "target_is_top3",
        "direction":  "높을수록 좋음",
        "valid_range": (0.0, 1.0),
        "first_should_be_zero": True,
    },
    {
        "col":        "fe_trar_cum_win_rate",
        "entity_col": "hrmTrarId",
        "entity_nm":  "조교사",
        "result_col": "target_is_top3",
        "direction":  "높을수록 좋음",
        "valid_range": (0.0, 1.0),
        "first_should_be_zero": True,
    },
]

# ─────────────────────────────────────────────
# Step 1: 각 피처별 검증 수행
# ─────────────────────────────────────────────
log_step(1, "누적 피처별 누수 검증 수행")

check_results = []

for feat in CUM_FEATURES:
    col        = feat["col"]
    entity_col = feat["entity_col"]
    entity_nm  = feat["entity_nm"]
    vmin, vmax = feat["valid_range"]
    first_zero = feat["first_should_be_zero"]

    if col not in df_sorted.columns:
        log(f"  [{col}] 컬럼 없음 — 스킵")
        continue

    log(f"\n  [{col}] 검증 시작")

    series = df_sorted[col].dropna()
    n_total = len(series)

    # 1. 범위 오류
    n_invalid_range = ((series < vmin) | (series > vmax)).sum()
    pct_invalid = round(n_invalid_range / n_total * 100, 2) if n_total > 0 else 0

    # 2. 첫 출전 검증
    first_race = df_sorted.groupby(entity_col).first().reset_index()
    first_vals = first_race[col].dropna()

    if first_zero:
        n_first_zero = (first_vals == 0).sum()
        n_first_total = len(first_vals)
        first_zero_pct = round(n_first_zero / n_first_total * 100, 2) if n_first_total > 0 else 0
    else:
        # 평균순위 등은 첫 경주에서 0이 아니어야 정상 (이전 실적 없으면 NaN이거나 별도 값)
        n_first_zero = (first_vals == 0).sum()
        n_first_total = len(first_vals)
        first_zero_pct = round(n_first_zero / n_first_total * 100, 2) if n_first_total > 0 else 0

    # 3. 시간적 변동성 (entity별 std 평균 — 고정값이면 누수 의심)
    entity_std = df_sorted.groupby(entity_col)[col].std().dropna()
    mean_std = round(entity_std.mean(), 4)
    pct_zero_std = round((entity_std == 0).sum() / len(entity_std) * 100, 2) if len(entity_std) > 0 else 0

    # 4. 현재 경주 결과 포함 의심 검증
    #    방법: 말별 누적승률이 해당 경주 직후와 비교해 이미 포함된 값인지
    #    - 말 A의 경주 i에서 cum_win_rate가,
    #      경주 i의 실제 결과를 포함한 값인지 의심 (direct test)
    #    - 간이 검증: 첫 경주에서 win=1인 말의 cum_win_rate가 이미 > 0이면 의심
    leakage_suspect_count = 0
    if first_zero and col.endswith("win_rate"):
        # 첫 경주에서 이긴 말 → 그 경주의 cum_win_rate > 0이면 현재 경주 포함 의심
        df_first = df_sorted.groupby(entity_col, group_keys=False).apply(
            lambda g: g.iloc[[0]]
        ).reset_index(drop=True)
        if "target_is_top3" in df_first.columns:
            first_winner = df_first[df_first["target_is_top3"] == 1]
            suspect = (first_winner[col] > 0).sum()
            leakage_suspect_count = int(suspect)

    # 5. 최종 판定
    if pct_invalid > 5:
        verdict     = "제외권장"
        verdict_en  = "EXCLUDE"
        reason      = f"범위 오류 {pct_invalid}% 초과"
    elif first_zero and first_zero_pct < 50:
        verdict     = "재계산필요"
        verdict_en  = "RECOMPUTE"
        reason      = f"첫출전 0 비율 {first_zero_pct}% — 이전 경주 포함 의심"
    elif leakage_suspect_count > 10:
        verdict     = "재계산필요"
        verdict_en  = "RECOMPUTE"
        reason      = f"첫 출전 당일 결과 포함 의심 {leakage_suspect_count}건"
    elif pct_zero_std > 30:
        verdict     = "재계산필요"
        verdict_en  = "RECOMPUTE"
        reason      = f"시간 변동없는 엔티티 {pct_zero_std}% — 고정값 의심"
    else:
        verdict     = "사용가능"
        verdict_en  = "OK"
        reason      = "구조적 누수 미탐지"

    row = {
        "feature":               col,
        "entity":                entity_nm,
        "n_total_records":       n_total,
        "n_invalid_range":       n_invalid_range,
        "pct_invalid_range":     pct_invalid,
        "n_entities":            n_first_total,
        "first_race_zero_count": n_first_zero,
        "first_race_zero_pct":   first_zero_pct,
        "entity_std_mean":       mean_std,
        "pct_entity_zero_std":   pct_zero_std,
        "leakage_suspect_count": leakage_suspect_count,
        "verdict":               verdict,
        "verdict_code":          verdict_en,
        "reason":                reason,
    }
    check_results.append(row)

    log(f"    범위오류: {pct_invalid}%")
    log(f"    첫출전 0비율: {first_zero_pct}%")
    log(f"    변동성(std): {mean_std} / 고정비율: {pct_zero_std}%")
    log(f"    누수의심건수: {leakage_suspect_count}")
    log(f"    판정: [{verdict}] — {reason}")

check_df = pd.DataFrame(check_results)
check_df.to_csv(
    os.path.join(OUT_DIAG, "cumulative_feature_leakage_check.csv"),
    index=False, encoding="utf-8-sig"
)
log(f"\n  cumulative_feature_leakage_check.csv 저장 완료")


# ─────────────────────────────────────────────
# Step 2: 누수 검증 보고서 작성
# ─────────────────────────────────────────────
log_step(2, "누수 검증 보고서 작성")

ok_list       = check_df[check_df["verdict_code"] == "OK"]["feature"].tolist()
recomp_list   = check_df[check_df["verdict_code"] == "RECOMPUTE"]["feature"].tolist()
exclude_list  = check_df[check_df["verdict_code"] == "EXCLUDE"]["feature"].tolist()

def make_table_row(row):
    verdict_label = {
        "OK":        "✅ 사용가능",
        "RECOMPUTE": "⚠️ 재계산필요",
        "EXCLUDE":   "❌ 제외권장",
    }.get(row["verdict_code"], row["verdict"])
    return (
        f"| `{row['feature']}` "
        f"| {row['entity']} "
        f"| {row['first_race_zero_pct']}% "
        f"| {row['pct_invalid_range']}% "
        f"| {row['entity_std_mean']} "
        f"| {row['leakage_suspect_count']} "
        f"| {verdict_label} "
        f"| {row['reason']} |"
    )

table_rows = "\n".join([make_table_row(r) for _, r in check_df.iterrows()])

feature_details = ""
for _, row in check_df.iterrows():
    verdict_label = {"OK": "사용가능", "RECOMPUTE": "재계산필요", "EXCLUDE": "제외권장"}.get(
        row["verdict_code"], row["verdict"]
    )
    feature_details += f"""
### `{row['feature']}` — 판정: [{verdict_label}]

| 검증 항목 | 결과 |
|----------|------|
| 대상 엔티티 | {row['entity']} |
| 총 레코드 수 | {row['n_total_records']:,} |
| 범위 오류 비율 | {row['pct_invalid_range']}% |
| 엔티티 수 | {row['n_entities']:,} |
| 첫 출전 시 값=0 비율 | {row['first_race_zero_pct']}% |
| 엔티티별 평균 변동성(std) | {row['entity_std_mean']} |
| 시간 변동 없는 엔티티 비율 | {row['pct_entity_zero_std']}% |
| 누수 의심 건수 | {row['leakage_suspect_count']} |
| **판정** | **{verdict_label}** |
| **사유** | {row['reason']} |

"""

report_md = f"""# 누적 피처 미래 데이터 누수 검증 보고서

**작성일**: 2026-04-25  
**데이터**: race_results_seoul_3years_revised.csv  
**목적**: 경주 전 정보를 기반으로 한 누적 피처의 미래 데이터 누수 여부 검증

---

## 1. 검증 원칙

누적 피처(예: `fe_horse_cum_win_rate`)는 **해당 경주 이전까지의 과거 기록**만 반영해야 한다.  
만약 현재 경주의 결과가 누적값에 포함되어 있다면, 이는 **미래 데이터 누수(Data Leakage)**이다.

### 검증 기준

1. **첫 출전 시 값 = 0 여부**: 이전 이력이 없는 첫 경주에서 누적값이 0이어야 정상
2. **범위 오류**: 유효 범위(0~1 등)를 벗어나면 계산 오류 의심
3. **시간 변동성**: 같은 말/기수/조교사가 경기를 거칠수록 값이 변해야 함 (고정값=의심)
4. **당일 결과 포함 의심**: 첫 경주에서 Top3인 말의 누적 승률이 이미 >0이면 현재 경주 포함 의심

---

## 2. 검증 결과 요약

| 피처 | 엔티티 | 첫출전_0비율 | 범위오류% | 변동성(std) | 누수의심 | 판정 | 사유 |
|------|--------|------------|---------|------------|--------|------|------|
{table_rows}

---

## 3. 피처별 상세 검증 결과

{feature_details}

---

## 4. 결론

### 그대로 사용 가능한 누적 피처
{chr(10).join([f"- `{f}`" for f in ok_list]) if ok_list else "- (없음)"}

### 재계산 후 사용해야 하는 누적 피처
{chr(10).join([f"- `{f}`" for f in recomp_list]) if recomp_list else "- (없음)"}

> **재계산 방법**: 각 경주 날짜 기준으로, 해당 경주 이전까지의 과거 경기 결과만 집계하여  
> 누적 승률 = (과거 Top3 수) / (과거 출전 수) 로 재산출

### 현재 단계에서 제외해야 하는 누적 피처
{chr(10).join([f"- `{f}`" for f in exclude_list]) if exclude_list else "- (없음)"}

---

## 5. 권고 사항

1. **모델 학습 시 시간 기준 분할(Time-based Split) 필수**
   - 훈련 기간: 예) 2023-01 ~ 2025-06
   - 테스트 기간: 예) 2025-07 ~ 2026-04

2. **재계산 필요 피처 처리 방법**
   - 각 경주 `race_id`에 대해, `schdRaceDt` 이전 날짜의 기록만 집계
   - `cumcount()` / `shift()` 패턴으로 안전하게 재계산
   - 단, 이 작업은 별도 파이프라인에서 수행 권장

3. **현재 사용 권장 피처**
   - 재계산 없이 안전하게 사용 가능한 피처만 모델링 데이터셋에 포함
   - 재계산 필요 피처는 `_RECOMPUTE_NEEDED` 접미사로 표시하여 추후 교체 예정임을 명시

4. **데이터 누수 방어 원칙**
   - 경마 예측 모델에서 누수는 다른 도메인보다 훨씬 치명적
   - 배당률, 레이스 타임, 마신차 등 경주 후 확정 정보는 반드시 제거
   - 누적 피처는 "해당 경주 당일을 포함하지 않는" 방식으로만 계산해야 함

---

> **분석자 노트**: 현재 데이터셋의 누적 피처 계산 방식을 원천 파이프라인 코드 수준에서  
> 반드시 검토할 것을 권장한다. 구조적 검증만으로는 당일 결과 포함 여부를 100% 확인할 수 없다.
"""

with open(os.path.join(REPORTS, "cumulative_feature_leakage_review.md"), "w", encoding="utf-8") as f:
    f.write(report_md)
log("  cumulative_feature_leakage_review.md 저장 완료")

print("\n[완료] 04_leakage_check.py 실행 완료")
print(f"  - cumulative_feature_leakage_check.csv")
print(f"  - cumulative_feature_leakage_review.md")
print(f"  - 사용가능: {ok_list}")
print(f"  - 재계산필요: {recomp_list}")
print(f"  - 제외권장: {exclude_list}")
