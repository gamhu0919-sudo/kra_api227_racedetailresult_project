# -*- coding: utf-8 -*-
"""
preprocess.py - 데이터 전처리 파이프라인
실행: python src/preprocess.py

산출물:
  data/processed/df_clean.csv     - 정상완주(ord<90) 전처리 완료 데이터
  data/processed/df_all.csv       - 전체(특수코드 포함) 전처리 완료 데이터
  reports/GPT/outputs/tables/08_merge_stats.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    MAIN_CSV, RACE_META_CSV, DATA_PROCESSED, REPORTS_TABLES,
    LOGS_DIR, ANA_COLS_PATTERN, LEAKAGE_COLS, UNDEFINED_COLS,
)
from utils import get_logger

logger = get_logger("preprocess", LOGS_DIR)


# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("전처리 파이프라인 시작")
    logger.info("=" * 60)

    # 1. 주 데이터 로드
    df = load_main_data()

    # 2. 4_경주기록정보.csv 병합
    df = merge_race_meta(df)

    # 3. 타입 정리
    df = fix_dtypes(df)

    # 4. wgHr 체중 컬럼 파싱
    df = parse_weight(df)

    # 5. hrTool 장구 파싱
    df = parse_tool(df)

    # 6. 불필요 컬럼 제거
    df = drop_useless_cols(df)

    # 7. 결측 처리
    df = handle_missing(df)

    # 8. 파생 인코딩
    df = encode_categoricals(df)

    # 9. 정상완주 vs 전체 분리
    df_all   = df.copy()
    df_clean = df[df["ord"] < 90].copy()

    logger.info(f"\n전체 데이터: {df_all.shape}")
    logger.info(f"정상완주 데이터(ord<90): {df_clean.shape}")

    # 저장
    df_all.to_csv(DATA_PROCESSED / "df_all.csv",
                  index=False, encoding="utf-8-sig")
    df_clean.to_csv(DATA_PROCESSED / "df_clean.csv",
                    index=False, encoding="utf-8-sig")
    logger.info(f"저장 완료: {DATA_PROCESSED / 'df_all.csv'}")
    logger.info(f"저장 완료: {DATA_PROCESSED / 'df_clean.csv'}")

    logger.info("=" * 60)
    logger.info("전처리 완료")
    logger.info("=" * 60)
    return df_all, df_clean


# ── 1. 주 데이터 로드 ─────────────────────────────────────────────────────
def load_main_data() -> pd.DataFrame:
    logger.info(f"\n[1] 주 데이터 로드: {MAIN_CSV}")
    df = pd.read_csv(MAIN_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  로드 완료: {df.shape}")

    # ord를 숫자로 강제 변환
    df["ord"] = pd.to_numeric(df["ord"], errors="coerce")
    logger.info(f"  ord NaN 발생 건수: {df['ord'].isnull().sum()}")

    return df


# ── 2. 4_경주기록정보.csv 병합 ────────────────────────────────────────────
def merge_race_meta(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"\n[2] 경주 메타 정보 병합: {RACE_META_CSV}")

    if not RACE_META_CSV.exists():
        logger.warning(f"  파일 없음: {RACE_META_CSV} - 병합 건너뜀")
        return df

    meta = pd.read_csv(RACE_META_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  메타 파일 Shape: {meta.shape}")

    # 병합 키: meet + rcDate + rcNo + chulNo
    merge_keys = ["meet", "rcDate", "rcNo", "chulNo"]
    meta_cols = merge_keys + ["rcDist", "weather", "track", "rcName", "rank", "prizeCond"]
    # 존재하는 컬럼만 사용
    avail_meta_cols = [c for c in meta_cols if c in meta.columns]
    logger.info(f"  사용 가능 메타 컬럼: {avail_meta_cols}")

    meta_sub = meta[avail_meta_cols].drop_duplicates(subset=merge_keys)

    # meet 컬럼 타입 맞추기
    df["meet"]   = df["meet"].astype(str).str.strip()
    meta_sub = meta_sub.copy()
    meta_sub["meet"] = meta_sub["meet"].astype(str).str.strip()

    before = len(df)
    df = df.merge(meta_sub, on=merge_keys, how="left", suffixes=("", "_meta"))
    after = len(df)

    # 병합 성공률
    for col in ["rcDist", "weather", "track"]:
        if col in df.columns:
            hit = df[col].notna().sum()
            logger.info(f"  {col} 병합 성공률: {hit/len(df)*100:.1f}% ({hit}/{len(df)})")

    if before != after:
        logger.warning(f"  병합 후 행 수 변화: {before} -> {after} (중복 확인 필요)")

    # 병합 통계 저장
    merge_stats = []
    for col in ["rcDist", "weather", "track", "rcName", "rank", "prizeCond"]:
        if col in df.columns:
            hit = df[col].notna().sum()
            merge_stats.append({
                "컬럼": col,
                "병합성공건수": hit,
                "전체건수": len(df),
                "성공률(%)": round(hit / len(df) * 100, 2),
                "결측건수": df[col].isnull().sum(),
            })
    if merge_stats:
        pd.DataFrame(merge_stats).to_csv(
            REPORTS_TABLES / "08_merge_stats.csv",
            index=False, encoding="utf-8-sig"
        )
        logger.info(f"  병합 통계 저장: {REPORTS_TABLES / '08_merge_stats.csv'}")

    return df


# ── 3. 타입 정리 ──────────────────────────────────────────────────────────
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n[3] 타입 정리")

    # 날짜형
    df["rcDate_dt"] = pd.to_datetime(df["rcDate"].astype(str), format="%Y%m%d", errors="coerce")

    # ID 컬럼은 str로
    for col in ["jkNo", "trNo", "hrNo", "owNo", "race_id", "entry_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # rcNo, chulNo, age는 int로
    for col in ["rcNo", "chulNo", "age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 수치형 처럼 보이는 컬럼 변환
    for col in ["wgBudam", "wgJk", "winOdds", "plcOdds", "rating", "hrRating",
                "rankRise", "implied_rank_from_win_odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("  타입 정리 완료")
    return df


# ── 4. 체중 파싱 ──────────────────────────────────────────────────────────
def parse_weight(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n[4] wgHr 체중 파싱")

    if "wgHr" not in df.columns:
        logger.warning("  wgHr 컬럼 없음 - 건너뜀")
        return df

    # 이미 파싱된 컬럼이 있으면 그것 사용
    if "horse_weight_current" in df.columns and "horse_weight_delta" in df.columns:
        existing_valid = df["horse_weight_current"].notna().sum()
        logger.info(f"  기존 horse_weight_current 유효값: {existing_valid}건 - 재파싱")

    # 패턴: "502(-2)" or "502(+3)" or "502" or "502(0)"
    pattern = re.compile(r"^(\d+)\s*\(([+-]?\d+)\)?$")

    parsed_cur  = []
    parsed_delt = []
    fail_count  = 0

    for val in df["wgHr"]:
        val_str = str(val).strip() if pd.notna(val) else ""
        m = pattern.match(val_str)
        if m:
            parsed_cur.append(int(m.group(1)))
            parsed_delt.append(int(m.group(2)))
        else:
            # 숫자만 있는 경우
            if val_str.isdigit():
                parsed_cur.append(int(val_str))
                parsed_delt.append(np.nan)
            else:
                parsed_cur.append(np.nan)
                parsed_delt.append(np.nan)
                if val_str:
                    fail_count += 1

    df["horse_weight_current"] = parsed_cur
    df["horse_weight_delta"]   = parsed_delt

    logger.info(f"  파싱 성공: {pd.Series(parsed_cur).notna().sum()}건")
    logger.info(f"  파싱 실패: {fail_count}건")
    return df


# ── 5. 장구 파싱 ──────────────────────────────────────────────────────────
def parse_tool(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n[5] hrTool 장구 파싱")

    if "hrTool" not in df.columns:
        logger.warning("  hrTool 컬럼 없음 - 건너뜀")
        return df

    # 장구 사용 여부 (값이 있고 '-'가 아닌 경우)
    df["has_tool"] = df["hrTool"].apply(
        lambda x: 0 if (pd.isna(x) or str(x).strip() in ["", "-", "nan"]) else 1
    ).astype(int)

    # 장구 개수 (쉼표로 구분된 항목 수)
    def count_tools(x):
        if pd.isna(x) or str(x).strip() in ["", "-", "nan"]:
            return 0
        items = [t.strip() for t in str(x).replace("/", ",").split(",") if t.strip()]
        return len(items)

    df["tool_count"] = df["hrTool"].apply(count_tools)

    # 주요 장구 포함 여부
    MAJOR_TOOLS = ["혀끈", "망사", "차양", "귀마개", "깔때기"]
    for tool in MAJOR_TOOLS:
        col = f"tool_{tool}"
        df[col] = df["hrTool"].apply(
            lambda x: 1 if (pd.notna(x) and tool in str(x)) else 0
        ).astype(int)

    logger.info(f"  장구 사용 비율: {df['has_tool'].mean()*100:.1f}%")
    logger.info(f"  평균 장구 개수: {df['tool_count'].mean():.2f}")
    return df


# ── 6. 불필요 컬럼 제거 ──────────────────────────────────────────────────
def drop_useless_cols(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n[6] 불필요 컬럼 제거")

    # _ana 계열 (100% 결측)
    ana_cols = [c for c in df.columns if ANA_COLS_PATTERN in c]
    logger.info(f"  _ana 계열 제거: {len(ana_cols)}개")

    # 100% 결측 컬럼 (ana 외)
    full_missing = [c for c in df.columns
                    if df[c].isnull().all() and c not in ana_cols]
    logger.info(f"  기타 100% 결측 제거: {full_missing}")

    drop_targets = list(set(ana_cols + full_missing))
    df = df.drop(columns=[c for c in drop_targets if c in df.columns])
    logger.info(f"  제거 후 컬럼 수: {df.shape[1]}")
    return df


# ── 7. 결측 처리 ──────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n[7] 결측 처리")

    # 수치형: 중앙값으로 대체 (단, is_missing 플래그 먼저 생성)
    flag_cols = ["horse_weight_current", "horse_weight_delta",
                 "winOdds", "plcOdds", "rating", "hrRating", "wgBudam", "wgJk"]

    for col in flag_cols:
        if col in df.columns and df[col].isnull().any():
            df[f"is_missing_{col}"] = df[col].isnull().astype(int)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: NaN -> 중앙값({median_val:.2f}) 대체, 플래그 생성")

    # 범주형: 'Unknown' 대체
    cat_fill = ["sex", "name", "track_condition_text"]
    for col in cat_fill:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # weather, track, rank (병합 후 추가될 컬럼)
    for col in ["weather", "track", "rank", "prizeCond", "rcName"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


# ── 8. 범주형 인코딩 ─────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n[8] 범주형 인코딩")

    # 주로상태 (track)
    if "track" in df.columns:
        track_map = {"良": 0, "稍": 1, "重": 2, "不": 3, "Unknown": -1}
        df["track_enc"] = df["track"].map(track_map).fillna(-1).astype(int)
        logger.info(f"  track 인코딩 완료: {df['track'].value_counts().to_dict()}")

    # 날씨 (weather)
    if "weather" in df.columns:
        weather_map = {"맑음": 0, "흐림": 1, "비": 2, "눈": 3, "Unknown": -1}
        # 값이 없으면 원본 그대로 map (빈값 -1)
        df["weather_enc"] = df["weather"].map(weather_map)
        if df["weather_enc"].isnull().any():
            # 미매핑 값들 순서대로 번호 부여
            unique_vals = df["weather"].unique()
            for i, v in enumerate(unique_vals):
                if v not in weather_map:
                    weather_map[v] = 4 + i
            df["weather_enc"] = df["weather"].map(weather_map).fillna(-1).astype(int)
        else:
            df["weather_enc"] = df["weather_enc"].astype(int)
        logger.info("  weather 인코딩 완료")

    # 거리군 (rcDist)
    if "rcDist" in df.columns:
        df["rcDist"] = pd.to_numeric(df["rcDist"], errors="coerce")
        def dist_group(d):
            if pd.isna(d): return "Unknown"
            if d <= 1200: return "단거리"
            elif d <= 1800: return "중거리"
            else: return "장거리"
        df["dist_group"] = df["rcDist"].apply(dist_group)
        logger.info("  dist_group 생성 완료")

    # 등급 (rank)
    if "rank" in df.columns:
        rank_vals = df["rank"].dropna().unique()
        rank_map  = {v: i for i, v in enumerate(sorted(rank_vals))}
        df["rank_enc"] = df["rank"].map(rank_map).fillna(-1).astype(int)
        logger.info(f"  rank 인코딩 완료 ({len(rank_map)}개 등급)")

    # prizeCond 인코딩
    if "prizeCond" in df.columns:
        priz_vals = df["prizeCond"].dropna().unique()
        priz_map  = {v: i for i, v in enumerate(sorted(priz_vals))}
        df["prizeCond_enc"] = df["prizeCond"].map(priz_map).fillna(-1).astype(int)
        logger.info("  prizeCond 인코딩 완료")

    # 성별 인코딩
    if "sex" in df.columns:
        sex_map = {"수": 0, "암": 1, "거": 2, "Unknown": -1}
        df["sex_enc"] = df["sex"].map(sex_map).fillna(-1).astype(int)
        logger.info(f"  sex 인코딩: {df['sex'].value_counts().to_dict()}")

    # 출전번호 상대 위치 (race 내)
    if "chulNo" in df.columns:
        df["field_size"] = df.groupby("race_id")["chulNo"].transform("count")
        df["chulNo_relative"] = df["chulNo"] / df["field_size"]

        # 게이트 구간 (내측/중앙/외측)
        def gate_zone(row):
            if pd.isna(row["chulNo"]) or pd.isna(row["field_size"]):
                return "Unknown"
            rel = row["chulNo"] / row["field_size"]
            if rel <= 0.33: return "내측"
            elif rel <= 0.67: return "중앙"
            else: return "외측"
        df["gate_zone"] = df.apply(gate_zone, axis=1)
        logger.info("  chulNo_relative, gate_zone 생성 완료")

    logger.info(f"  최종 컬럼 수: {df.shape[1]}")
    return df


if __name__ == "__main__":
    main()
