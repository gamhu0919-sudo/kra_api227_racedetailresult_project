import os
import time
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
from dotenv import load_dotenv

# 1. .env 파일에서 인증 정보 불러오기 
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HEADERS = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET
}

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 기준: 오늘부터 1년 전까지
TODAY = datetime.now()
ONE_YEAR_AGO = TODAY - timedelta(days=365)

def remove_tags(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'<[^>]+>', '', text)

def fetch_datalab_trend(keywords):
    print("[데이터랩] 검색어 트렌드 수집 시작...")
    url = "https://openapi.naver.com/v1/datalab/search"
    
    keyword_groups = []
    for kw in keywords:
        keyword_groups.append({
            "groupName": kw,
            "keywords": [kw]
        })
        
    body = {
        "startDate": ONE_YEAR_AGO.strftime("%Y-%m-%d"),
        "endDate": TODAY.strftime("%Y-%m-%d"),
        "timeUnit": "week",
        "keywordGroups": keyword_groups
    }
    
    response = requests.post(url, headers=HEADERS, json=body)
    
    if response.status_code == 200:
        data = response.json()
        results = []
        for group in data.get('results', []):
            group_name = group['title']
            for d in group['data']:
                results.append({
                    "기기그룹": group_name,
                    "수집기간": d['period'],
                    "상대적검색량(ratio)": d['ratio']
                })
        
        df = pd.DataFrame(results)
        if not df.empty:
            save_path = os.path.join(DATA_DIR, "datalab_trend_1year_horse.csv")
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f" -> datalab_trend_1year_horse.csv 저장 완료! ({len(df)}건)")
        else:
            print(" -> 데이터랩 결과가 존재하지 않습니다.")
    else:
        print("데이터랩 API Error:", response.status_code, response.text)

def search_api(api_type, keywords, sort_opt="date"):
    print(f"[{api_type.upper()}] 검색 결과 최대 1000건 수집 시작...")
    url = f"https://openapi.naver.com/v1/search/{api_type}.json"
    
    all_results = []
    
    for kw in keywords:
        print(f" - 타겟 검색어: {kw}")
        for start_idx in range(1, 1001, 100):
            params = {
                "query": kw,
                "display": 100,
                "start": start_idx,
                "sort": sort_opt
            }
            
            res = requests.get(url, headers=HEADERS, params=params)
            
            if res.status_code == 200:
                items = res.json().get('items', [])
                if not items:
                    break 
                
                for item in items:
                    item['search_keyword'] = kw 
                    
                    if 'title' in item:
                        item['title'] = remove_tags(item['title'])
                    if 'description' in item:
                        item['description'] = remove_tags(item['description'])
                    
                    include_item = True
                    item_date = None
                    
                    if api_type == 'blog': 
                        if 'postdate' in item and item['postdate']:
                            try:
                                item_date = datetime.strptime(item['postdate'], "%Y%m%d")
                            except ValueError:
                                pass
                    elif api_type == 'news': 
                        if 'pubDate' in item and item['pubDate']:
                            try:
                                item_date = parser.parse(item['pubDate']).replace(tzinfo=None)
                            except Exception:
                                pass
                    
                    if item_date and item_date < ONE_YEAR_AGO:
                        include_item = False
                        
                    if include_item:
                        all_results.append(item)
                        
                time.sleep(0.1) 
            else:
                print(f"   ! API 접속 에러 발생 ({kw}, start={start_idx}):", res.status_code)
                break
                
    df = pd.DataFrame(all_results)
    if not df.empty:
        save_path = os.path.join(DATA_DIR, f"search_1year_{api_type}_horse.csv")
        df.to_csv(save_path, index=False, encoding='utf-8-sig') 
        print(f" -> search_1year_{api_type}_horse.csv 저장 완료! ({len(df)}건 수집됨)")
    else:
        print(f" -> {api_type} 분야에서 수집된 유효 데이터가 없습니다.")

if __name__ == "__main__":
    if not CLIENT_ID or not CLIENT_SECRET or CLIENT_ID == "your_client_id_here":
        print("\n[!] 치명적 오류: NAVER API 설정이 되어있지 않습니다.")
        exit(1)
        
    targets = ["경마"]
    print(f"수집 타겟: {targets}")
    
    fetch_datalab_trend(targets)
    search_api("news", targets, sort_opt="date")
    search_api("blog", targets, sort_opt="date")
    search_api("cafearticle", targets, sort_opt="date")
    search_api("shop", targets, sort_opt="sim")
    
    print("🎉 '경마' 데이터 수집 완료.")
