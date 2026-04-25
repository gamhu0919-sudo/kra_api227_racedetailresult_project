import os
import requests
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

# .env 파일 로드
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 설정
BASE_URL = "https://apis.data.go.kr/B553766/psgr/getStnPsgr"
SERVICE_KEY = os.getenv("PUBLIC_DATA_SERVICE_KEY")

# 폴더 경로 설정 (상대 경로 기준)
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOG_DIR = ROOT_DIR / "logs"

# 로그 함수
def log_event(event_type, message, status_code=None, payload=None):
    log_file = LOG_DIR / "fetch_log.jsonl"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "message": message,
        "status_code": status_code,
        "payload": payload
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def fetch_data_for_date(target_date, stn_nm, line_nm=None):
    """
    특정 날짜와 역명으로 API 호출
    """
    params = {
        "serviceKey": SERVICE_KEY,
        "pasngYmd": target_date,
        "stnNm": stn_nm,
        "dataType": "JSON",
        "numOfRows": 1000,
        "pageNo": 1
    }
    if line_nm:
        params["lineNm"] = line_nm

    # 인증키 마스킹 처리된 로그용 페이로드
    masked_params = params.copy()
    masked_params["serviceKey"] = "REDACTED"

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        
        # 공공데이터포털은 HTTP 200이면서 본문에 XML 에러가 오는 경우가 많음
        content_type = response.headers.get('Content-Type', '')
        if 'json' not in content_type.lower():
            if '<returnAuthMsg>' in response.text or '<cmmMsgHeader>' in response.text:
                log_event("API_ERROR", f"API 인증 또는 시스템 오류 발생 (XML 응답)", response.status_code, {"response_text": response.text[:500]})
                return None

        data = response.json()
        
        # 응답 구조 확인 및 저장
        save_path = RAW_DATA_DIR / f"{target_date}_{stn_nm}_{line_nm if line_nm else 'all'}_raw.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return data

    except Exception as e:
        log_event("CONNECTION_ERROR", str(e), payload=masked_params)
        return None

def main():
    if not SERVICE_KEY:
        print("Error: PUBLIC_DATA_SERVICE_KEY not found in .env")
        return

    # 1. 수집 대상 날짜 생성 (최근 7일)
    today = datetime.now()
    date_list = [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
    
    # 2. 역명 시도 목록
    attempts = [
        {"stnNm": "경마공원", "lineNm": "4호선"},
        {"stnNm": "경마공원역", "lineNm": "4호선"},
        {"stnNm": "경마공원", "lineNm": None}
    ]

    all_collected_data = []

    for target_date in date_list:
        print(f"[{target_date}] 수집 시작...")
        success = False
        
        for attempt in attempts:
            stn = attempt["stnNm"]
            line = attempt["lineNm"]
            print(f"  - 조회 시도: {stn} ({line if line else '모든호선'})")
            
            result = fetch_data_for_date(target_date, stn, line)
            
            if result and result.get("response", {}).get("header", {}).get("resultCode") == "00":
                items_wrapper = result.get("response", {}).get("body", {}).get("items", {})
                if isinstance(items_wrapper, dict):
                    item_data = items_wrapper.get("item", [])
                else:
                    item_data = []

                # 리스트 또는 단건(dict) 처리
                if isinstance(item_data, dict):
                    item_list = [item_data]
                elif isinstance(item_data, list):
                    item_list = item_data
                else:
                    item_list = []
                
                if item_list:
                    # 원본 데이터에 수집일자 추가
                    for item in item_list:
                        item["fetchDate"] = target_date
                    
                    all_collected_data.extend(item_list)
                    log_event("SUCCESS", f"{target_date} {stn} {len(item_list)}건 수집 완료", 200)
                    print(f"    => {len(item_list)}건 수집 성공!")
                    success = True
                    break # 해당 날짜 성공 시 다음 시도 건너뜀
                else:
                    log_event("EMPTY", f"{target_date} {stn} 결과 없음 (item list is empty)", 200)
            else:
                msg = result.get("response", {}).get("header", {}).get("resultMsg") if result else "No Response"
                log_event("API_FAIL", f"{target_date} {stn} 실패: {msg}", 200)
            
            time.sleep(0.5) # API 매너

    # 3. 데이터 정제 및 통합
    if all_collected_data:
        df = pd.DataFrame(all_collected_data)
        
        # 컬럼명 매핑 (API 응답 기준 예상 필드 -> 한글)
        # 실제 필드명이 다를 수 있으므로 방어적으로 매핑
        mapping = {
            "pasngYmd": "수송일자",
            "pasngTm": "수송시간",
            "lineNm": "호선명",
            "stnNm": "역명",
            "stnNo": "역번호",
            "stnCd": "역코드",
            "pasngCnt": "승하차인원",
            "tmPasngCnt": "시간대별승하차인원",
            "tkpSectCd": "교통카드구분코드",
            "tkpUserSectCd": "교통카드사용자구분코드",
            "fetchDate": "원본조회일"
        }
        df = df.rename(columns=mapping)
        
        # 파일 저장
        csv_file = PROCESSED_DATA_DIR / "kyungmapark_stn_psgr_last7days.csv"
        parquet_file = PROCESSED_DATA_DIR / "kyungmapark_stn_psgr_last7days.parquet"
        
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        df.to_parquet(parquet_file, index=False)
        
        print(f"\n[최종 결과] 총 {len(df)}건의 데이터가 저장되었습니다.")
        print(f"CSV: {csv_file}")
        print(f"Parquet: {parquet_file}")
    else:
        print("\n[최종 결과] 수집된 데이터가 없습니다. (경마공원역이 서울교통공사 관할이 아닐 가능성이 높습니다.)")

if __name__ == "__main__":
    main()
