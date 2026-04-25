import pandas as pd
import os

file_path = r'kra_api227_racedetailresult_project\data\raw\서울시 지하철 호선별 역별 유_무임 승하차 인원 정보.csv'

def check_data():
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 인코딩 시도
    encodings = ['cp949', 'utf-8-sig', 'utf-8']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"Success loading with {enc} encoding.")
            break
        except:
            continue

    if df is None:
        print("Failed to load CSV with common encodings.")
        return

    print("\n[Columns]")
    print(df.columns.tolist())

    print("\n[Sample Data]")
    print(df.head(3))

    # 경마공원역 필터링
    target_stn = '경마공원'
    filtered = df[df['지하철역'].str.contains(target_stn, na=False)]
    
    if not filtered.empty:
        print(f"\n[Found data for {target_stn}]")
        print(f"Total rows: {len(filtered)}")
        print(filtered[['사용년월', '호선명', '지하철역']].drop_duplicates().sort_values('사용년월').head(10))
    else:
        print(f"\n[No data found for {target_stn}]")
        # 역명 후보군 출력 (혹시 다른 이름일지 확인)
        print("\n[Station Name Suggestions (first 20 unique names)]")
        print(df['지하철역'].unique()[:20])

if __name__ == "__main__":
    check_data()
