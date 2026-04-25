# kra_api227_racedetailresult_project/src/sample_test.py

import sys
from pathlib import Path

# 모듈 인입을 위해 경로 설정
sys.path.append(str(Path(__file__).resolve().parent))

from config import API_BASE_URL, SERVICE_KEY, DIR_LOGS
from api_client import KRA_APIClient
from xml_parser import KRA_XMLParser
from utils import CustomLogger

def run_sample_test():
    """단일 API 호출 및 파싱 테스트"""
    logger = CustomLogger("sample_test", DIR_LOGS).get_logger()
    client = KRA_APIClient(API_BASE_URL, SERVICE_KEY, logger)
    parser = KRA_XMLParser(logger)
    
    # 샘플 파라미터 (2025년 3월 1일 서울 1경주)
    meet = 1
    rc_date = "20250301"
    rc_no = 1
    
    logger.info("=== 샘플 테스트 시작 ===")
    xml_data = client.test_call(meet, rc_date, rc_no)
    
    if xml_data:
        items = parser.parse_items(xml_data)
        df = parser.to_dataframe(items)
        
        logger.info(f"파싱 성공! 추출된 데이터 수: {len(df)}건")
        if not df.empty:
            logger.info("추출된 컬럼 목록: " + ", ".join(df.columns))
            logger.info("데이터 샘플 (상위 3개):\n" + df.head(3).to_string())
            
        logger.info("=== 샘플 테스트 종료 (성공) ===")
    else:
        logger.error("샘플 테스트 종료 (실패)")

if __name__ == "__main__":
    run_sample_test()
