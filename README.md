# KRA API227 경주별상세성적표 분석 프로젝트

본 프로젝트는 한국마사회에서 제공하는 오픈 API(API227, `racedetailresult`)를 활용하여 경주별 출전마의 상세 기록 데이터를 수집하고 분석하는 시스템입니다.

## 1. 프로젝트 개요
- **분석 대상 기간**: 2025년 3월 ~ 2026년 3월 (13개월)
- **주요 목표**:
    - XML 기반 대량 데이터 수집 및 원본 보존
    - 정형 데이터 변환 및 품질 검증
    - 기수, 조교사, 마필별 성과 분석 및 EDA
- **특징**: 재현 가능한 파이프라인, 객체 지향적 API 클라이언트 설계, 전문가급 문서화

## 2. 폴더 구조
```text
kra_api227_racedetailresult_project/
├─ docs/                 # API 명세, 변수 사전, 전처리 규칙, EDA 계획
├─ data/
│  ├─ raw/xml/           # 수집된 원본 XML 파일
│  ├─ interim/           # 파싱 완료된 중간 CSV
│  └─ processed/         # 전처리 완료된 최종 분석용 CSV
├─ src/                  # 수집, 파싱, 검증, 전처리, EDA 코드
├─ images/               # EDA 결과 시각화 차트
├─ logs/                 # 실행 로그
└─ README.md
```

## 3. 실행 프로세스
1. **의존성 설치**: `pip install -r requirements.txt`
2. **데이터 수집**: `python src/collect_race_detail.py` (2025.03~2026.03 기간 자동 순회)
3. **데이터 전처리**: (추후 통합 스크립트 작성 예정)
   - 파싱: `src/xml_parser.py`
   - 검증: `src/validate_dataset.py`
   - 정제: `src/preprocess_race_detail.py`
4. **시각화 및 보고**: `src/eda_race_detail.py` 및 `src/build_report_assets.py` 실행

## 4. 데이터 적재 원칙
- **Raw (XML)**: API의 원본 응답을 그대로 저장하여 데이터 유실 방지
- **Interim (CSV)**: XML을 정형 테이블로 변환한 형태 (원본값 보존)
- **Processed (CSV)**: 타입 변환, 코드 정규화, 파생변수 생성이 완료된 최종 분석용 데이터

## 5. 주의사항
- **API 키**: `src/config.py`에 올바른 서비스키가 설정되어 있어야 합니다.
- **수집 속도**: 과도한 호출 방지를 위해 `API_DELAY`가 설정되어 있으며, 1년치 데이터 수집 시 상당한 시간이 소요될 수 있습니다.
- **가상환경**: `uv` 또는 `venv`를 활용하여 독립적인 환경에서 실행하는 것을 권장합니다.

---
**분석 대상 기간 엄수**: 모든 데이터 분석 및 리포트는 2025년 3월부터 2026년 3월까지의 데이터만을 대상으로 수행합니다.
