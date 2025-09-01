# 🚀 통합 알파 팩터 포트폴리오 시스템

모든 모듈을 하나로 통합한 완전한 퀀트 투자 솔루션

## 📋 시스템 개요

이 통합 시스템은 기존의 모든 개별 모듈들을 하나의 완전한 플랫폼으로 통합합니다:

- **🔢 Alpha Factor Library**: 60+ 종류의 팩터 생성
- **📊 Z-Score 분석**: 표준화된 팩터 분석
- **⚖️ 포트폴리오 최적화**: 평균-분산, 리스크 패리티
- **🧪 백테스팅 엔진**: 완전한 성과 검증
- **🗄️ 데이터베이스**: 결과 저장 및 관리
- **🌐 통합 UI**: 직관적인 웹 인터페이스

## 🏗️ 아키텍처

```
통합 알파 시스템/
├── 📁 core/
│   ├── unified_alpha_system.py    # 메인 통합 클래스
│   ├── alpha_factor_library.py    # 팩터 생성 엔진
│   ├── zscore.py                  # Z-Score 분석 엔진
│   ├── backtesting_engine.py      # 백테스팅 엔진
│   ├── portfolio_optimizer.py     # 포트폴리오 최적화
│   └── database.py                # 데이터베이스 관리
├── 📁 ui/
│   └── unified_app.py             # 통합 Streamlit UI
└── 📁 utils/
    └── run_unified_system.py      # 시스템 런처
```

## 🚀 빠른 시작

### 1️⃣ 설치

```bash
pip install streamlit pandas numpy plotly yfinance scikit-learn scipy cvxpy xgboost
```

### 2️⃣ 실행

```bash
python run_unified_system.py
```

또는 직접:

```bash
streamlit run unified_app.py
```

### 3️⃣ 접속

브라우저에서 `http://localhost:8501` 접속

## 💡 주요 기능

### 🔢 통합 팩터 분석
- **기술적 팩터**: 모멘텀, 평균회귀, 변동성, 거래량
- **고급 기술적**: RSI, 볼린저밴드, Z-Score, 상관계수  
- **머신러닝**: Random Forest, PCA, XGBoost
- **리스크**: 베타, 하방리스크
- **수학/통계**: 칼만필터, Hurst지수, 웨이블릿
- **Z-Score 표준화**: 모든 팩터를 동일한 척도로 변환

### 📈 포트폴리오 구성
- **동일가중**: 선택된 종목들을 동일 비중으로
- **팩터가중**: Z-Score에 비례하여 가중
- **순위가중**: 팩터 순위에 따른 가중
- **롱-숏**: 상위 종목 롱, 하위 종목 숏

### 🧪 백테스팅
- **성과 지표**: 수익률, 샤프비율, 최대손실, VaR
- **벤치마크 비교**: SPY, QQQ 등과 비교
- **거래비용 반영**: 실제적인 수익률 계산
- **리밸런싱**: 일간, 주간, 월간, 분기간

### ⚠️ 리스크 관리
- **실시간 알림**: 극단적 Z-Score 감지
- **포트폴리오 모니터링**: 변동성, 집중도 추적
- **손실 한계**: 최대손실 임계값 설정

## 🎯 사용 방법

### 1단계: 투자 유니버스 설정
사이드바에서 투자 대상 선택:
- 미국 대형주 (AAPL, MSFT, GOOGL...)
- 한국 대형주 (삼성전자, SK하이닉스...)  
- 글로벌 혼합
- 섹터 ETF
- 커스텀 (직접 입력)

### 2단계: 분석 파라미터 설정
- **분석 기간**: 1년 ~ 5년
- **Z-Score 임계값**: 매매 신호 기준
- **포트폴리오 설정**: 최대 비중, 거래비용
- **리밸런싱 주기**: 일간/주간/월간/분기

### 3단계: 분석 실행
**🏠 대시보드 탭**에서 "🚀 완전 분석 실행" 버튼 클릭

### 4단계: 결과 분석
각 탭에서 결과 확인:
- **🔢 팩터 분석**: Z-Score 히트맵, 상관관계
- **📈 포트폴리오**: 최적 비중, 리스크 분석  
- **📊 백테스팅**: 성과 차트, 지표 요약

## 📊 UI 구성

### 🏠 대시보드
- 시스템 상태 모니터링
- 빠른 실행 버튼
- 최근 분석 결과 요약

### 🔢 팩터 분석  
- Z-Score 히트맵
- 팩터 시계열 차트
- 상관관계 분석
- 팩터 통계 요약

### 📈 포트폴리오
- 최적화 설정
- 포트폴리오 구성 차트
- 리스크 분석 지표
- 종목별 비중 테이블

### 📊 백테스팅
- 누적 수익률 차트
- 벤치마크 비교
- 성과 지표 대시보드
- 상세 통계 테이블

### ⚙️ 시스템 관리
- 데이터베이스 현황
- 백업 및 최적화
- 저장된 팩터 목록
- 백테스팅 기록

## 🔧 고급 설정

### Z-Score 분석 설정
```python
config = UnifiedSystemConfig(
    enable_zscore=True,              # Z-Score 활성화
    zscore_threshold_high=1.5,       # 매수 임계값
    zscore_threshold_low=-1.5,       # 매도 임계값
)
```

### 포트폴리오 제약조건
```python
constraints = OptimizationConstraints(
    max_weight=0.15,                 # 최대 15% 비중
    min_weight=0.01,                 # 최소 1% 비중
    long_only=True,                  # 롱 온리
    leverage=1.0                     # 레버리지 없음
)
```

### 백테스팅 설정
```python
backtest_config = BacktestConfig(
    rebalance_frequency='monthly',   # 월간 리밸런싱
    transaction_cost=0.001,          # 0.1% 거래비용
    slippage=0.0005                  # 0.05% 슬리피지
)
```

## 🎨 커스터마이징

### 새로운 팩터 추가
1. `alpha_factor_library.py`에 팩터 클래스 추가
2. `get_available_factors()` 메서드에 등록
3. UI에서 자동으로 사용 가능

### 새로운 최적화 방법 추가  
1. `portfolio_optimizer.py`에 Optimizer 클래스 상속
2. `optimize()` 메서드 구현
3. `PortfolioOptimizer`에 등록

### UI 커스터마이징
1. `unified_app.py`에서 탭 수정
2. Plotly 차트 스타일 변경
3. Streamlit 컴포넌트 추가

## 📈 성능 최적화

### 캐싱 활용
```python
@st.cache_data(ttl=1800)  # 30분 캐시
def load_market_data(tickers, period):
    # 데이터 로딩 로직
```

### 병렬 처리
```python
# 여러 팩터 동시 계산
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(calculate_factor, factor) 
              for factor in factors]
```

### 데이터베이스 최적화
```python
# 정기적 최적화
system.database.vacuum_database()
system.database.cleanup_old_data(30)
```

## 🔍 문제 해결

### 일반적 문제들

**1. 데이터 로딩 실패**
```bash
# Yahoo Finance 연결 문제
pip install --upgrade yfinance
```

**2. 최적화 실패**
```bash
# CVXPY 문제
pip install --upgrade cvxpy
```

**3. 메모리 부족**
- 분석 기간 단축
- 종목 수 줄이기
- 캐시 클리어

**4. UI 반응 느림**
- 데이터 캐시 활용
- 백그라운드 계산
- 세션 상태 최적화

### 로그 확인
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 개발 로드맵

### 단기 (1개월)
- [ ] 실시간 데이터 피드 추가
- [ ] 모바일 UI 최적화
- [ ] 성능 지표 확장

### 중기 (3개월)
- [ ] 딥러닝 팩터 추가
- [ ] 포트폴리오 귀속분석
- [ ] 위험관리 강화

### 장기 (6개월)
- [ ] 멀티에셋 지원
- [ ] API 서비스 제공
- [ ] 클라우드 배포

## 🤝 기여 방법

1. Fork 저장소
2. 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 📞 지원

- 📧 이메일: [support@example.com]
- 📞 전화: [02-1234-5678]
- 💬 채팅: [Slack/Discord 링크]

---

## 🎉 마무리

이제 모든 기능이 통합된 완전한 퀀트 투자 플랫폼을 사용할 수 있습니다!

**핵심 워크플로우**:
1. 🚀 `python run_unified_system.py` 실행
2. 📊 투자 유니버스 및 설정 선택  
3. 🔄 완전 분석 실행
4. 📈 결과 분석 및 포트폴리오 구성
5. 💾 결과 저장 및 모니터링

**Happy Quant Trading! 📈💰**