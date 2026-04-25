# -*- coding: utf-8 -*-
"""
데이터 탐색 스크립트 - 구현 계획 수립을 위한 기초 분석
"""
import pandas as pd
import sys
import os

BASE = r'c:\Users\kang_\Desktop\NewKyungma\kra_api227_racedetailresult_project'

print("=" * 60)
print("1. merged_team_source.csv 기본 정보")
print("=" * 60)
dfm = pd.read_csv(
    os.path.join(BASE, 'data', 'processed', 'merged_team_source.csv'),
    encoding='utf-8-sig', low_memory=False
)
print(f"Shape: {dfm.shape}")
print(f"rcDate 범위: {dfm['rcDate'].min()} ~ {dfm['rcDate'].max()}")
print(f"race_id 유니크: {dfm['race_id'].nunique()}")
print(f"hrNo 유니크: {dfm['hrNo'].nunique()}")
print(f"jkNo 유니크: {dfm['jkNo'].nunique()}")
print(f"trNo 유니크: {dfm['trNo'].nunique()}")
print()

print("=== meet 분포 ===")
print(dfm['meet'].value_counts())
print()

print("=== ord 분포 ===")
print(dfm['ord'].value_counts().sort_index())
print()

needed_cols = ['rcDist', 'weather', 'track', 'rcName', 'rank', 'prizeCond']
existing = set(dfm.columns)
print("=== 주요 메타컬럼 존재 여부 ===")
for c in needed_cols:
    status = "존재" if c in existing else "없음"
    print(f"  {c}: {status}")
print()

print("=" * 60)
print("2. 4_경주기록정보.csv 기본 정보")
print("=" * 60)
df4 = pd.read_csv(
    os.path.join(BASE, 'data', 'team_source', '4_경주기록정보.csv'),
    encoding='utf-8-sig', low_memory=False
)
print(f"Shape: {df4.shape}")
print(f"rcDate 범위: {df4['rcDate'].min()} ~ {df4['rcDate'].max()}")
print(f"meet 분포:")
print(df4['meet'].value_counts())
print()
print(f"4번 파일 주요컬럼: {[c for c in needed_cols if c in df4.columns]}")
print()

print("=" * 60)
print("3. 6_경주성적정보.csv 기본 정보")
print("=" * 60)
df6 = pd.read_csv(
    os.path.join(BASE, 'data', 'team_source', '6_경주성적정보.csv'),
    encoding='utf-8-sig', low_memory=False
)
print(f"Shape: {df6.shape}")
print(f"rcDate 범위: {df6['rcDate'].min()} ~ {df6['rcDate'].max()}")
print(f"meet 분포:")
print(df6['meet'].value_counts())
print()
print(f"6번 파일 주요컬럼: {[c for c in needed_cols if c in df6.columns]}")
print()

print("=" * 60)
print("4. 1_경주성적정보 기본 정보")
print("=" * 60)
df1 = pd.read_csv(
    os.path.join(BASE, 'data', 'team_source', '1_경주성적정보(15063979).csv'),
    encoding='utf-8-sig', low_memory=False
)
print(f"Shape: {df1.shape}")
print(f"컬럼: {list(df1.columns)}")
print(f"rcDate 범위: {df1['rcDate'].min()} ~ {df1['rcDate'].max()}" if 'rcDate' in df1.columns else "rcDate 없음")
print()

print("=" * 60)
print("5. 결측률 상위 30 (merged_team_source)")
print("=" * 60)
missing = (dfm.isnull().sum() / len(dfm) * 100).sort_values(ascending=False)
print(missing.head(30).to_string())
print()

print("=" * 60)
print("6. ord >= 90 특수코드 분포")
print("=" * 60)
special = dfm[dfm['ord'] >= 90]['ord'].value_counts().sort_index()
print(special)
print(f"특수코드 비중: {len(dfm[dfm['ord'] >= 90]) / len(dfm) * 100:.2f}%")
print()

print("=" * 60)
print("7. 경주당 출전두수 분포")
print("=" * 60)
field_size = dfm.groupby('race_id')['chulNo'].count()
print(field_size.describe())
print()

print("=== 기수 jkNo dtype ===")
print(dfm['jkNo'].dtype, dfm['jkNo'].head(5).tolist())
print("=== 조교사 trNo dtype ===")
print(dfm['trNo'].dtype, dfm['trNo'].head(5).tolist())
print()

print("완료!")
