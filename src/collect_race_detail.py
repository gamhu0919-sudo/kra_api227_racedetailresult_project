# kra_api227_racedetailresult_project/src/collect_race_detail.py

import os
import time
from pathlib import Path
from datetime import datetime

from config import (
    API_BASE_URL, SERVICE_KEY, TARGET_MEETS, MAX_RC_NO, 
    COLLECT_START_DATE, COLLECT_END_DATE, NUM_OF_ROWS, API_DELAY,
    DIR_DATA_RAW_XML, DIR_LOGS
)
from utils import CustomLogger, ensure_dir, generate_date_list
from api_client import KRA_APIClient
from xml_parser import KRA_XMLParser

def run_collection():
    # 1. 초기화
    logger = CustomLogger("collect_race_detail", DIR_LOGS).get_logger()
    client = KRA_APIClient(API_BASE_URL, SERVICE_KEY, logger)
    parser = KRA_XMLParser(logger)
    ensure_dir(DIR_DATA_RAW_XML)
    
    dates = generate_date_list(COLLECT_START_DATE, COLLECT_END_DATE)
    logger.info(f"수집 시작일: {COLLECT_START_DATE}")
    logger.info(f"수집 종료일: {COLLECT_END_DATE}")
    logger.info(f"총 예상 일수: {len(dates)}일")
    
    stats = {"success": 0, "fail": 0, "empty": 0, "files_saved": 0}
    
    # 2. 날짜별 루프
    for rc_date in dates:
        logger.info(f"--- Date: {rc_date} 수집 시작 ---")
        
        # 3. 경마장별 루프 (1: 서울, 2: 제주, 3: 부경)
        for meet in TARGET_MEETS:
            empty_streak = 0  # 연속적으로 경주가 없는 경우 방지
            
            # 4. 경주번호별 루프 (1 ~ MAX_RC_NO)
            for rc_no in range(1, MAX_RC_NO + 1):
                # API 호출
                xml_res = client.fetch_xml(meet, rc_date, rc_no, page_no=1, num_rows=NUM_OF_ROWS)
                
                if not xml_res:
                    stats["fail"] += 1
                    logger.error(f"  [Fail] meet={meet}, date={rc_date}, no={rc_no}")
                    continue
                
                # XML 파싱 (아이템 존재 여부 확인)
                items = parser.parse_items(xml_res)
                
                if not items:
                    stats["empty"] += 1
                    empty_streak += 1
                    logger.debug(f"  [Empty] meet={meet}, date={rc_date}, no={rc_no}")
                    
                    # 3회 연속 빈 응답이면 해당 날짜/경마장 종료
                    if empty_streak >= 3:
                        logger.info(f"  [Stop] 3회 연속 빈 응답. meet={meet} 종료.")
                        break
                    continue
                
                # 아이템 있으면 저장
                empty_streak = 0
                file_name = f"meet{meet}_{rc_date}_rc{rc_no:02d}_page1.xml"
                file_path = DIR_DATA_RAW_XML / file_name
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(xml_res)
                
                stats["success"] += 1
                stats["files_saved"] += 1
                logger.info(f"  [Success] Saved: {file_name} ({len(items)} items)")
                
                # API 딜레이
                time.sleep(API_DELAY)
                
    # 요약 출력
    logger.info("==========================================")
    logger.info("수집 작업이 완료되었습니다.")
    logger.info(f"  성공 건수: {stats['success']}")
    logger.info(f"  실패 건수: {stats['fail']}")
    logger.info(f"  빈 응답 건수: {stats['empty']}")
    logger.info(f"  저장된 XML 파일 수: {stats['files_saved']}")
    logger.info("==========================================")

if __name__ == "__main__":
    run_collection()
