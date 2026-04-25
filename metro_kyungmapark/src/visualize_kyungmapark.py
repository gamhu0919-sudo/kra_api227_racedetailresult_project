import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import os
from pathlib import Path

# 설정
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_FILE = BASE_DIR.parent / "data" / "raw" / "서울시 지하철 호선별 역별 유_무임 승하차 인원 정보.csv"
OUTPUT_IMG = BASE_DIR / "images" / "kyungmapark_full_analysis.png"

def visualize():
    # 1. 데이터 로드
    if not RAW_DATA_FILE.exists():
        print(f"Error: File not found at {RAW_DATA_FILE}")
        return

    df = None
    # 서울시 데이터는 보통 cp949 인코딩임
    try:
        df = pd.read_csv(RAW_DATA_FILE, encoding='cp949')
        print("CSV loaded with cp949.")
    except Exception:
        df = pd.read_csv(RAW_DATA_FILE, encoding='utf-8-sig')
        print("CSV loaded with utf-8-sig.")

    # 2. 컬럼 정확한 인덱스 기반 할당
    # 0:년월, 1:호선, 2:역명, 3:유임승차, 4:무임승차, 5:유임하차, 6:무임하차, 7:작업일
    col_ym = df.columns[0]
    col_line = df.columns[1]
    col_stn = df.columns[2]
    col_u_in = df.columns[3]
    col_f_in = df.columns[4]
    col_u_out = df.columns[5]
    col_f_out = df.columns[6]

    # 3. 경마공원역 필터링
    # 데이터 행에서 직접 필터링 (컬럼 이름 깨짐과 무관)
    df_kj = df[df[col_stn].astype(str).str.contains('경마공원', na=False)].copy()
    
    if df_kj.empty:
        print("Error: 경마공원역 데이터를 찾을 수 없습니다.")
        # 디버깅: 역명 샘플 출력
        print("First 20 unique station names in data:")
        print(df[col_stn].unique()[:20])
        return

    # 4. 데이터 가공
    # 날짜 변환
    df_kj[col_ym] = pd.to_datetime(df_kj[col_ym].astype(str), format='%Y%m', errors='coerce')
    df_kj = df_kj.dropna(subset=[col_ym])
    df_kj = df_kj.sort_values(col_ym)

    # 지표 계산
    df_kj['총승차'] = df_kj[col_u_in] + df_kj[col_f_in]
    df_kj['총하차'] = df_kj[col_u_out] + df_kj[col_f_out]
    
    # 5. 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.35)

    # 그래프 1: 승하차 인원 합계 추이
    ax1.plot(df_kj[col_ym], df_kj['총승차'], label='총 승차', color='#0077b6', linewidth=2.5, marker='o', markersize=4)
    ax1.plot(df_kj[col_ym], df_kj['총하차'], label='총 하차', color='#e63946', linewidth=2.5, marker='s', markersize=4)
    ax1.set_title(f"경마공원역 연도별 승하차 인원 추이 ({df_kj[col_ym].dt.year.min()} - {df_kj[col_ym].dt.year.max()})", 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylabel("인원 수 (명)", fontsize=13)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(axis='both', which='major', labelsize=11)

    # 그래프 2: 유임/무임 승차 인원 비율 (누적 그래프)
    ax2.stackplot(df_kj[col_ym], 
                  df_kj[col_u_in], 
                  df_kj[col_f_in], 
                  labels=['유임승차', '무임승차'], 
                  colors=['#2a9d8f', '#e76f51'], alpha=0.8)
    ax2.set_title("유임 vs 무임 승차 인원 구성 (누적)", fontsize=18, fontweight='bold', pad=20)
    ax2.set_ylabel("인원 수 (명)", fontsize=13)
    ax2.set_xlabel("기간 (년월)", fontsize=13)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.tick_params(axis='both', which='major', labelsize=11)

    # 이미지 저장
    plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=200)
    
    # 결과 출력
    years = sorted(df_kj[col_ym].dt.year.unique())
    print(f"\n[성공] 시각화 완료")
    print(f"- 데이터 기간: {df_kj[col_ym].dt.date.min()} ~ {df_kj[col_ym].dt.date.max()}")
    print(f"- 데이터 연도 수: {len(years)}년분")
    print(f"- 누적 승차량: {df_kj['총승차'].sum():,.0f}명")
    print(f"- 저장된 이미지: {OUTPUT_IMG}")

if __name__ == "__main__":
    visualize()
