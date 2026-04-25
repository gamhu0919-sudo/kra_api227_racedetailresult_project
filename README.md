# KRA 경주마 Top3 예측 플랫폼

본 프로젝트는 한국마사회(KRA) 상세 성적 데이터를 기반으로 인공지능 모델을 학습시켜, 다가올 경주의 출전마가 Top3(1, 2, 3위) 내에 진입할 확률을 예측하는 플랫폼입니다.

## 🚀 프로젝트 진행 단계

1.  **Stage01: 데이터 수집**
    - XML 기반 대량 데이터 수집 및 정형화 변환
2.  **Stage02: 예측형 EDA**
    - 피처 중요도 분석 및 타겟 변수 설계
3.  **Stage03: Top3 모델링**
    - LightGBM 기반 바이너리 분류 모델 학습 및 검증
4.  **Stage04: Streamlit 대시보드**
    - 분석 결과 시각화 및 경주별 예측 리포트 대시보드 구축
5.  **Stage05: 다음 경주 입력 예측 파이프라인 (현재 단계)**
    - 미래 경주 정보 입력 시 전처리 및 추론 자동화 자동화
    - **현재 상태**: 파이프라인 안정화(Hardening) 및 대시보드 연동 완료
6.  **Stage06: 자동 수집 및 운영형 플랫폼 (예정)**
    - 실시간 API 연동 및 자동 예측 업데이트 실현

## 🛠️ 설치 및 실행

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 추론 파이프라인 실행 (Stage 05)
사용자가 미래 경주 데이터를 기입한 후 아래 명령을 통해 예측을 수행합니다.
```bash
python run_inference.py
```

### 대시보드 실행 (Stage 04)
```bash
streamlit run streamlit_app.py
```

## 📂 폴더 구조
- `kra_race_prediction_stage03_top3_modeling/`: 학습 모델 및 학습 데이터
- `kra_race_prediction_stage04_streamlit_dashboard/`: 대시보드 소스 (utils, 레이아웃)
- `kra_race_prediction_stage05_inference_pipeline/`: 추론 파이프라인 (입력, 검증, 예측)
- `streamlit_app.py`: 통합 대시보드 메인 엔트리
- `run_inference.py`: 추론 자동화 마스터 스크립트

---
**현재 단계**: Stage 05 다음 경주 입력 예측 파이프라인 안정화
