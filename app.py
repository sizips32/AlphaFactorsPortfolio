"""
Alpha Factor Generator - 메인 Streamlit 웹앱
금융투자 포트폴리오를 위한 종합적인 알파 팩터 생성 도구

작성자: AI Assistant
작성일: 2025년 7월 23일
버전: 2.0 (데이터베이스 연동)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import yfinance as yf
from typing import Dict, List, Optional, Any
import logging
import json

# 로컬 모듈 import
from database import DatabaseManager
from alpha_factor_library import (
    FactorEngine, FactorCategory, FactorValidator,
    TechnicalFactors, FundamentalFactors, MachineLearningFactors, RiskFactors,
    EnhancedFactorLibrary,
    ensure_numeric_dataframe, ensure_numeric_series
)
from backtesting_engine import BacktestEngine, BacktestConfig, BacktestVisualizer
from portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# 페이지 설정
st.set_page_config(
    page_title="Alpha Factor Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사용자 정의 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .factor-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if 'factor_engine' not in st.session_state:
    st.session_state.factor_engine = FactorEngine()

if 'enhanced_factor_library' not in st.session_state:
    st.session_state.enhanced_factor_library = EnhancedFactorLibrary()

if 'portfolio_optimizer' not in st.session_state:
    st.session_state.portfolio_optimizer = PortfolioOptimizer()

if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

if 'calculated_factors' not in st.session_state:
    st.session_state.calculated_factors = {}

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

# 데이터 로딩 함수
@st.cache_data
def load_market_data(symbols: List[str], period: str = "2y") -> Optional[Dict[str, pd.DataFrame]]:
    """시장 데이터 로딩 (캐시 우선)"""
    try:
        data = {'prices': pd.DataFrame(), 'volumes': pd.DataFrame(), 'returns': pd.DataFrame(), 'market_caps': pd.DataFrame()}
        
        # 각 심볼에 대해 캐시 확인 후 로드
        for symbol in symbols:
            # 캐시에서 먼저 확인
            cached_data = st.session_state.db_manager.get_cached_market_data(symbol)
            
            if cached_data is not None and len(cached_data) > 100:  # 충분한 데이터가 있으면
                symbol_data = cached_data
            else:
                # Yahoo Finance에서 새로 로드
                ticker = yf.Ticker(symbol)
                symbol_data = ticker.history(period=period)
                
                if not symbol_data.empty:
                    # 데이터베이스에 캐시
                    st.session_state.db_manager.cache_market_data(symbol_data, symbol)
            
            if not symbol_data.empty:
                data['prices'][symbol] = symbol_data['Close']
                data['volumes'][symbol] = symbol_data['Volume']
        
        if data['prices'].empty:
            st.error("유효한 데이터를 불러올 수 없습니다. 종목 심볼을 확인해주세요.")
            return None
        
        # 데이터 타입 확인 및 변환
        data['prices'] = ensure_numeric_dataframe(data['prices'].dropna())
        data['volumes'] = ensure_numeric_dataframe(data['volumes'].dropna())
        data['returns'] = ensure_numeric_dataframe(data['prices'].pct_change().dropna())
        data['market_caps'] = ensure_numeric_dataframe(data['prices'] * data['volumes'])
        
        # 최종 검증: 데이터가 비어있지 않은지 확인
        if data['prices'].empty or len(data['prices'].index) == 0:
            st.error("로드된 데이터가 비어있습니다. 종목 심볼과 기간을 확인해주세요.")
            return None
        
        return data
        
    except Exception as e:
        st.error(f"데이터 로딩 오류: {str(e)}")
        return None

# 안전한 팩터 계산 함수
def safe_calculate_factor(factor_type: str, factor_name: str, data: Dict, **kwargs):
    """안전한 팩터 계산 및 데이터베이스 저장"""
    try:
        engine = st.session_state.factor_engine
        factor_values = engine.calculate_factor(factor_type, factor_name, data, **kwargs)
        
        # 데이터 타입 검증
        if isinstance(factor_values, pd.DataFrame):
            factor_values = ensure_numeric_dataframe(factor_values)
        elif isinstance(factor_values, pd.Series):
            factor_values = ensure_numeric_series(factor_values)
        
        # 유효성 검증
        if factor_values is None or factor_values.empty:
            st.error("팩터 계산 결과가 비어있습니다.")
            return None
        
        # 데이터베이스에 저장
        factor_full_name = f"{factor_type}_{factor_name}"
        description = f"{factor_type} 카테고리의 {factor_name} 팩터"
        
        # 팩터 정의 저장
        factor_id = st.session_state.db_manager.save_factor_definition(
            name=factor_full_name,
            category=factor_type,
            description=description,
            parameters=kwargs
        )
        
        # 팩터 값 저장
        if isinstance(factor_values, pd.Series):
            # Series를 DataFrame으로 변환
            factor_df = pd.DataFrame({factor_values.name or 'factor': factor_values})
        else:
            factor_df = factor_values
        
        # 무한값과 NaN 제거
        factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
        factor_df = factor_df.dropna(how='all')
        
        st.session_state.db_manager.save_factor_values(factor_id, factor_df)
        
        return factor_values, factor_id
        
    except Exception as e:
        st.error(f"팩터 계산 오류: {str(e)}")
        return None, None

# 사이드바 렌더링
def render_sidebar():
    """사이드바 렌더링"""
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.title("🎛️ 설정")
    
    # 데이터 설정
    st.sidebar.subheader("📊 데이터 설정")
    
    # 기본 종목 리스트
    default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
    
    symbols_input = st.sidebar.text_area(
        "종목 심볼 (쉼표로 구분)",
        value=', '.join(default_symbols),
        help="예: AAPL, GOOGL, MSFT"
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    period = st.sidebar.selectbox(
        "데이터 기간",
        options=['1y', '2y', '3y', '5y'],
        index=1,
        help="Yahoo Finance에서 가져올 데이터 기간"
    )
    
    if st.sidebar.button("📥 데이터 로드", type="primary"):
        with st.spinner("데이터를 로딩하는 중..."):
            data = load_market_data(symbols, period)
            if data is not None:
                st.session_state.sample_data = data
                st.sidebar.success(f"{len(symbols)}개 종목 데이터 로드 완료!")
            else:
                st.sidebar.error("데이터 로딩에 실패했습니다.")
    
    # 현재 로드된 데이터 정보
    if st.session_state.sample_data is not None:
        data = st.session_state.sample_data
        if not data['prices'].empty and len(data['prices'].index) > 0:
            st.sidebar.info(f"""
            **로드된 데이터:**
            - 종목 수: {len(data['prices'].columns)}
            - 기간: {data['prices'].index[0].strftime('%Y-%m-%d')} ~ {data['prices'].index[-1].strftime('%Y-%m-%d')}
            - 총 일수: {len(data['prices'])}
            """)
        else:
            st.sidebar.warning("데이터가 비어있습니다. 다시 로드해주세요.")
    
    # 저장된 팩터 목록
    st.sidebar.subheader("💾 저장된 팩터")
    factor_list = st.session_state.db_manager.get_factor_list()
    
    if factor_list:
        for factor in factor_list[:5]:  # 최근 5개만 표시
            st.sidebar.text(f"• {factor['name']}")
        
        if len(factor_list) > 5:
            st.sidebar.text(f"... 외 {len(factor_list) - 5}개")
    else:
        st.sidebar.text("저장된 팩터가 없습니다.")
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# 대시보드 페이지
def render_dashboard():
    """대시보드 페이지"""
    st.markdown('<h1 class="main-header">📈 Alpha Factor Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Alpha Factor 생성 플랫폼에 오신 것을 환영합니다!
    
    이 플랫폼은 금융투자 포트폴리오를 위한 종합적인 알파 팩터 생성 도구입니다.
    다양한 팩터 카테고리와 머신러닝 기법을 활용하여 수익성 있는 투자 전략을 개발할 수 있습니다.
    """)
    
    # 기능 소개
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="factor-card">
            <h3>🔧 팩터 생성</h3>
            <p>기술적 분석, 펀더멘털 분석, 머신러닝 등 다양한 방법으로 알파 팩터를 생성합니다.</p>
            <ul>
                <li>모멘텀 & 평균회귀</li>
                <li>변동성 & 거래량</li>
                <li>밸류에이션 & 수익성</li>
                <li>ML/DL 기반 팩터</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="factor-card">
            <h3>📊 백테스팅</h3>
            <p>생성된 팩터의 과거 성과를 검증하고 다양한 지표로 평가합니다.</p>
            <ul>
                <li>수익률 & 리스크 분석</li>
                <li>드로우다운 분석</li>
                <li>정보 계수 (IC) 분석</li>
                <li>거래비용 고려</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="factor-card">
            <h3>⚖️ 포트폴리오 최적화</h3>
            <p>다양한 최적화 기법으로 최적의 포트폴리오를 구성합니다.</p>
            <ul>
                <li>평균-분산 최적화</li>
                <li>리스크 패리티</li>
                <li>블랙-리터만</li>
                <li>계층적 리스크 패리티</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 빠른 시작 가이드
    st.markdown("### 🚀 빠른 시작 가이드")
    
    with st.expander("1단계: 데이터 로드", expanded=True):
        st.markdown("""
        1. 왼쪽 사이드바에서 분석하고 싶은 종목들을 입력하세요
        2. 데이터 기간을 선택하세요 (1년~5년)
        3. "데이터 로드" 버튼을 클릭하세요
        """)
    
    with st.expander("2단계: 팩터 생성"):
        st.markdown("""
        1. "팩터 생성" 페이지로 이동하세요
        2. 원하는 팩터 카테고리를 선택하세요
        3. 파라미터를 조정하고 "팩터 계산" 버튼을 클릭하세요
        4. 실시간으로 팩터 값과 미리보기를 확인하세요
        """)
    
    with st.expander("3단계: 백테스팅"):
        st.markdown("""
        1. "백테스팅" 페이지로 이동하세요
        2. 백테스팅 설정을 조정하세요 (기간, 리밸런싱 주기 등)
        3. "백테스팅 실행" 버튼을 클릭하세요
        4. 성과 분석 결과를 확인하세요
        """)
    
    # 현재 상태 표시
    if st.session_state.sample_data is not None:
        data = st.session_state.sample_data
        
        # 데이터가 유효한지 확인
        if not data['prices'].empty and len(data['prices'].index) > 0:
            st.success("✅ 데이터가 로드되었습니다. 팩터 생성을 시작할 수 있습니다!")
            
            # 간단한 데이터 미리보기
            st.markdown("### 📊 로드된 데이터 미리보기")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**최근 가격 데이터**")
                st.dataframe(data['prices'].tail())
            
            with col2:
                st.markdown("**기본 통계**")
                try:
                    stats_df = pd.DataFrame({
                        '평균 수익률': data['returns'].mean() * 252,
                        '연간 변동성': data['returns'].std() * np.sqrt(252),
                        '최근 가격': data['prices'].iloc[-1]
                    })
                    st.dataframe(stats_df)
                except Exception:
                    st.warning("통계 계산 중 오류가 발생했습니다.")
        else:
            st.warning("로드된 데이터가 비어있습니다. 다시 로드해주세요.")
    else:
        st.info("ℹ️ 먼저 사이드바에서 데이터를 로드해주세요.")

# 팩터 생성 페이지
def render_factor_generator():
    """팩터 생성 페이지"""
    st.title("🔧 팩터 생성기")
    
    if st.session_state.sample_data is None:
        st.warning("⚠️ 먼저 사이드바에서 데이터를 로드해주세요.")
        return
    
    data = st.session_state.sample_data
    
    # 데이터 유효성 검증
    if data['prices'].empty or len(data['prices'].index) == 0:
        st.error("로드된 데이터가 비어있습니다. 사이드바에서 데이터를 다시 로드해주세요.")
        return
    
    # 팩터 카테고리 선택
    st.subheader("📂 팩터 카테고리 선택")
    
    category_options = {
        "기술적 분석 (기본)": "technical",
        "기술적 분석 (고급)": "advanced_technical",
        "펀더멘털 분석 (기본)": "fundamental",
        "펀더멘털 분석 (고급)": "advanced_fundamental",
        "머신러닝": "machine_learning",
        "리스크": "risk",
        "고급 수학/통계": "advanced_math",
        "딥러닝": "deep_learning"
    }
    
    selected_category = st.selectbox(
        "팩터 카테고리",
        options=list(category_options.keys()),
        help="생성하고 싶은 팩터의 카테고리를 선택하세요"
    )
    
    category = category_options[selected_category]
    
    # 카테고리별 팩터 생성 인터페이스
    if category == 'technical':
        render_technical_factors(data)
    elif category == 'fundamental':
        render_fundamental_factors(data)
    elif category == 'machine_learning':
        render_ml_factors(data)
    elif category == 'risk':
        render_risk_factors(data)
    elif category in ['advanced_technical', 'advanced_fundamental', 'advanced_math', 'deep_learning']:
        render_enhanced_factors(data, category)

def render_technical_factors(data):
    """기술적 팩터 생성 인터페이스"""
    st.subheader("📈 기술적 분석 팩터")
    
    factor_type = st.selectbox(
        "팩터 유형",
        ["모멘텀", "평균회귀", "변동성", "거래량"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if factor_type == "모멘텀":
            st.markdown("#### 모멘텀 팩터 설정")
            
            window = st.slider("모멘텀 기간 (일)", 5, 252, 20)
            method = st.radio(
                "모멘텀 유형",
                ["simple", "log", "risk_adjusted"],
                format_func=lambda x: {
                    "simple": "단순 모멘텀",
                    "log": "로그 모멘텀", 
                    "risk_adjusted": "리스크 조정 모멘텀"
                }[x]
            )
            
            if st.button("모멘텀 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중..."):
                    result = safe_calculate_factor(
                        'technical', 'momentum', data, 
                        window=window, method=method
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['momentum'] = factor_values
                        st.success(f"모멘텀 팩터 계산 완료! (ID: {factor_id})")
        
        elif factor_type == "평균회귀":
            st.markdown("#### 평균회귀 팩터 설정")
            
            window = st.slider("계산 기간 (일)", 5, 100, 20)
            method = st.radio(
                "평균회귀 유형",
                ["zscore", "bollinger", "rsi"],
                format_func=lambda x: {
                    "zscore": "Z-Score",
                    "bollinger": "볼린저 밴드",
                    "rsi": "RSI"
                }[x]
            )
            
            if st.button("평균회귀 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중..."):
                    result = safe_calculate_factor(
                        'technical', 'mean_reversion', data, 
                        window=window, method=method
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['mean_reversion'] = factor_values
                        st.success(f"평균회귀 팩터 계산 완료! (ID: {factor_id})")
        
        elif factor_type == "변동성":
            st.markdown("#### 변동성 팩터 설정")
            
            window = st.slider("변동성 계산 기간 (일)", 10, 252, 20)
            method = st.radio(
                "변동성 유형",
                ["realized", "garch"],
                format_func=lambda x: {
                    "realized": "실현 변동성",
                    "garch": "GARCH 변동성"
                }[x]
            )
            
            if st.button("변동성 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중..."):
                    result = safe_calculate_factor(
                        'technical', 'volatility', data, 
                        window=window, method=method
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['volatility'] = factor_values
                        st.success(f"변동성 팩터 계산 완료! (ID: {factor_id})")
        
        elif factor_type == "거래량":
            st.markdown("#### 거래량 팩터 설정")
            
            method = st.radio(
                "거래량 팩터 유형",
                ["obv", "vroc", "volume_ratio"],
                format_func=lambda x: {
                    "obv": "On-Balance Volume",
                    "vroc": "Volume Rate of Change",
                    "volume_ratio": "거래량 비율"
                }[x]
            )
            
            if st.button("거래량 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중..."):
                    result = safe_calculate_factor(
                        'technical', 'volume', data, 
                        method=method
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['volume'] = factor_values
                        st.success(f"거래량 팩터 계산 완료! (ID: {factor_id})")
    
    with col2:
        st.markdown("#### 팩터 미리보기")
        
        # 최근 계산된 팩터가 있으면 표시
        if st.session_state.calculated_factors:
            latest_factor_key = list(st.session_state.calculated_factors.keys())[-1]
            latest_factor = st.session_state.calculated_factors[latest_factor_key]
            
            if isinstance(latest_factor, pd.DataFrame):
                st.dataframe(latest_factor.tail().round(4))
                
                # 간단한 차트
                fig = px.line(latest_factor.tail(50), 
                             title=f"{latest_factor_key} 팩터 (최근 50일)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(latest_factor.tail().round(4))

def render_fundamental_factors(data):
    """펀더멘털 팩터 생성 인터페이스"""
    st.subheader("💰 펀더멘털 분석 팩터")
    
    factor_type = st.selectbox(
        "팩터 유형",
        ["밸류에이션", "수익성"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if factor_type == "밸류에이션":
            st.markdown("#### 밸류에이션 팩터 설정")
            
            method = st.radio(
                "밸류에이션 유형",
                ["pbr", "relative_price"],
                format_func=lambda x: {
                    "pbr": "Price-to-Book Ratio (근사치)",
                    "relative_price": "상대적 가격"
                }[x]
            )
            
            if st.button("밸류에이션 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중..."):
                    result = safe_calculate_factor(
                        'fundamental', 'valuation', data, 
                        method=method
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['valuation'] = factor_values
                        st.success(f"밸류에이션 팩터 계산 완료! (ID: {factor_id})")
        
        elif factor_type == "수익성":
            st.markdown("#### 수익성 팩터 설정")
            
            window = st.slider("수익성 계산 기간 (일)", 30, 756, 252)
            
            if st.button("수익성 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중..."):
                    result = safe_calculate_factor(
                        'fundamental', 'profitability', data, 
                        window=window
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['profitability'] = factor_values
                        st.success(f"수익성 팩터 계산 완료! (ID: {factor_id})")
    
    with col2:
        st.markdown("#### 팩터 미리보기")
        
        if st.session_state.calculated_factors:
            latest_factor_key = list(st.session_state.calculated_factors.keys())[-1]
            latest_factor = st.session_state.calculated_factors[latest_factor_key]
            
            if isinstance(latest_factor, pd.DataFrame):
                st.dataframe(latest_factor.tail().round(4))

def render_ml_factors(data):
    """머신러닝 팩터 생성 인터페이스"""
    st.subheader("🤖 머신러닝 기반 팩터")
    
    factor_type = st.selectbox(
        "팩터 유형",
        ["Random Forest", "PCA"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if factor_type == "Random Forest":
            st.markdown("#### Random Forest 팩터 설정")
            
            window = st.slider("학습 윈도우 (일)", 100, 500, 252)
            n_estimators = st.slider("트리 개수", 50, 200, 100)
            
            if st.button("Random Forest 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중... (시간이 걸릴 수 있습니다)"):
                    result = safe_calculate_factor(
                        'machine_learning', 'random_forest', data, 
                        window=window, n_estimators=n_estimators
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['random_forest'] = factor_values
                        st.success(f"Random Forest 팩터 계산 완료! (ID: {factor_id})")
        
        elif factor_type == "PCA":
            st.markdown("#### PCA 팩터 설정")
            
            n_components = st.slider("주성분 개수", 2, 10, 5)
            
            if st.button("PCA 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중..."):
                    result = safe_calculate_factor(
                        'machine_learning', 'pca', data, 
                        n_components=n_components
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['pca'] = factor_values
                        st.success(f"PCA 팩터 계산 완료! (ID: {factor_id})")
    
    with col2:
        st.markdown("#### 팩터 미리보기")
        
        if st.session_state.calculated_factors:
            latest_factor_key = list(st.session_state.calculated_factors.keys())[-1]
            latest_factor = st.session_state.calculated_factors[latest_factor_key]
            
            if isinstance(latest_factor, pd.DataFrame):
                st.dataframe(latest_factor.tail().round(4))

def render_risk_factors(data):
    """리스크 팩터 생성 인터페이스"""
    st.subheader("⚠️ 리스크 팩터")
    
    factor_type = st.selectbox(
        "팩터 유형",
        ["베타", "하방 리스크"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if factor_type == "베타":
            st.markdown("#### 베타 팩터 설정")
            
            window = st.slider("베타 계산 기간 (일)", 60, 504, 252)
            
            if st.button("베타 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중..."):
                    result = safe_calculate_factor(
                        'risk', 'beta', data, 
                        window=window
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['beta'] = factor_values
                        st.success(f"베타 팩터 계산 완료! (ID: {factor_id})")
        
        elif factor_type == "하방 리스크":
            st.markdown("#### 하방 리스크 팩터 설정")
            
            window = st.slider("리스크 계산 기간 (일)", 60, 504, 252)
            
            if st.button("하방 리스크 팩터 계산", type="primary"):
                with st.spinner("팩터 계산 중..."):
                    result = safe_calculate_factor(
                        'risk', 'downside_risk', data, 
                        window=window
                    )
                    
                    if result[0] is not None:
                        factor_values, factor_id = result
                        st.session_state.calculated_factors['downside_risk'] = factor_values
                        st.success(f"하방 리스크 팩터 계산 완료! (ID: {factor_id})")
    
    with col2:
        st.markdown("#### 팩터 미리보기")
        
        if st.session_state.calculated_factors:
            latest_factor_key = list(st.session_state.calculated_factors.keys())[-1]
            latest_factor = st.session_state.calculated_factors[latest_factor_key]
            
            if isinstance(latest_factor, pd.DataFrame):
                st.dataframe(latest_factor.tail().round(4))

# 백테스팅 페이지
def render_backtesting():
    """백테스팅 페이지"""
    st.title("📊 백테스팅")
    
    if st.session_state.sample_data is None:
        st.warning("⚠️ 먼저 시장 데이터를 로드해주세요.")
        return
    
    data = st.session_state.sample_data
    
    # 데이터 유효성 검증
    if data['prices'].empty or len(data['prices'].index) == 0:
        st.error("로드된 데이터가 비어있습니다. 사이드바에서 데이터를 다시 로드해주세요.")
        return
    
    if not st.session_state.calculated_factors:
        st.warning("⚠️ 먼저 팩터를 생성해주세요.")
        return
    
    # 백테스팅 설정
    st.subheader("⚙️ 백테스팅 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 날짜 범위 설정
        try:
            start_date = st.date_input(
                "시작 날짜",
                value=data['prices'].index[0].date(),
                min_value=data['prices'].index[0].date(),
                max_value=data['prices'].index[-1].date()
            )
            
            end_date = st.date_input(
                "종료 날짜",
                value=data['prices'].index[-1].date(),
                min_value=data['prices'].index[0].date(),
                max_value=data['prices'].index[-1].date()
            )
        except (IndexError, AttributeError):
            st.error("날짜 설정 중 오류가 발생했습니다. 데이터를 다시 로드해주세요.")
            return
        
        rebalance_frequency = st.selectbox(
            "리밸런싱 주기",
            options=['daily', 'weekly', 'monthly', 'quarterly'],
            index=2,
            format_func=lambda x: {
                'daily': '일별',
                'weekly': '주별',
                'monthly': '월별',
                'quarterly': '분기별'
            }[x]
        )
    
    with col2:
        transaction_cost = st.slider("거래비용 (%)", 0.0, 1.0, 0.1, 0.01)
        n_assets = st.slider("포트폴리오 종목 수", 5, min(20, len(data['prices'].columns)), 10)
        
        portfolio_method = st.selectbox(
            "포트폴리오 구성 방법",
            options=['equal_weight', 'factor_weight', 'rank_weight'],
            format_func=lambda x: {
                'equal_weight': '동일 가중',
                'factor_weight': '팩터 점수 가중',
                'rank_weight': '순위 가중'
            }[x]
        )
    
    # 팩터 선택
    st.subheader("📈 백테스팅할 팩터 선택")
    
    selected_factor = st.selectbox(
        "팩터 선택",
        options=list(st.session_state.calculated_factors.keys()),
        help="백테스팅을 수행할 팩터를 선택하세요"
    )
    
    # 백테스팅 실행
    if st.button("🚀 백테스팅 실행", type="primary"):
        with st.spinner("백테스팅을 실행하는 중..."):
            try:
                # 백테스팅 설정
                config = BacktestConfig(
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.min.time()),
                    rebalance_frequency=rebalance_frequency,
                    transaction_cost=transaction_cost / 100,
                )
                
                # 백테스팅 엔진 생성
                engine = BacktestEngine(config)
                
                # 팩터 점수 준비
                factor_values = st.session_state.calculated_factors[selected_factor]
                
                if isinstance(factor_values, pd.Series):
                    # Series를 DataFrame으로 변환
                    factor_df = pd.DataFrame({col: factor_values for col in data['prices'].columns})
                else:
                    factor_df = factor_values
                
                # 백테스팅 실행
                results = engine.run_backtest(
                    factor_scores=factor_df,
                    returns=data['returns'],
                    portfolio_method=portfolio_method,
                    n_assets=n_assets
                )
                
                st.session_state.backtest_results = results
                
                # 백테스팅 설정을 데이터베이스에 저장
                config_dict = {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'rebalance_frequency': rebalance_frequency,
                    'transaction_cost': transaction_cost,
                    'n_assets': n_assets,
                    'portfolio_method': portfolio_method
                }
                
                # 결과 저장 (팩터 ID 필요)
                # 여기서는 간단히 0으로 설정, 실제로는 팩터 ID를 추적해야 함
                st.session_state.db_manager.save_backtest_results(
                    factor_id=1,  # 실제 팩터 ID로 교체 필요
                    config=config_dict,
                    performance_metrics=results.performance_metrics,
                    portfolio_returns=results.portfolio_returns
                )
                
                st.success("백테스팅이 완료되었습니다!")
                
            except Exception as e:
                st.error(f"백테스팅 오류: {str(e)}")
    
    # 백테스팅 결과 표시
    if st.session_state.backtest_results is not None:
        st.subheader("📊 백테스팅 결과")
        
        results = st.session_state.backtest_results
        
        # 성과 지표 표
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = results.performance_metrics
        
        with col1:
            st.metric("총 수익률", f"{metrics.get('Total Return', 0):.2%}")
            st.metric("연간 수익률", f"{metrics.get('Annualized Return', 0):.2%}")
        
        with col2:
            st.metric("샤프 비율", f"{metrics.get('Sharpe Ratio', 0):.3f}")
            st.metric("소르티노 비율", f"{metrics.get('Sortino Ratio', 0):.3f}")
        
        with col3:
            st.metric("최대 손실", f"{metrics.get('Max Drawdown', 0):.2%}")
            st.metric("변동성", f"{metrics.get('Volatility', 0):.2%}")
        
        with col4:
            st.metric("승률", f"{metrics.get('Win Rate', 0):.2%}")
            st.metric("칼마 비율", f"{metrics.get('Calmar Ratio', 0):.3f}")
        
        # 성과 차트
        st.subheader("📈 성과 차트")
        
        try:
            fig = BacktestVisualizer.plot_performance(results)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"차트 생성 오류: {str(e)}")
        
        # 팩터 분석
        if results.factor_analysis:
            st.subheader("🔍 팩터 분석")
            
            factor_metrics = results.factor_analysis
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("정보 계수 (IC)", f"{factor_metrics.get('Information Coefficient', 0):.4f}")
                st.metric("IC 정보 비율", f"{factor_metrics.get('IC Information Ratio', 0):.3f}")
            
            with col2:
                st.metric("IC 표준편차", f"{factor_metrics.get('IC Standard Deviation', 0):.4f}")
                st.metric("IC 적중률", f"{factor_metrics.get('IC Hit Rate', 0):.2%}")

# 포트폴리오 최적화 페이지
def render_portfolio_optimization():
    """포트폴리오 최적화 페이지"""
    st.title("⚖️ 포트폴리오 최적화")
    
    if st.session_state.sample_data is None:
        st.warning("⚠️ 먼저 시장 데이터를 로드해주세요.")
        return
    
    data = st.session_state.sample_data
    
    # 데이터 유효성 검증
    if data['prices'].empty or len(data['prices'].index) == 0:
        st.error("로드된 데이터가 비어있습니다. 사이드바에서 데이터를 다시 로드해주세요.")
        return
    
    # 최적화 설정
    st.subheader("⚙️ 최적화 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_method = st.selectbox(
            "최적화 방법",
            options=['mean_variance', 'risk_parity'],
            format_func=lambda x: {
                'mean_variance': '평균-분산 최적화',
                'risk_parity': '리스크 패리티'
            }[x]
        )
        
        covariance_method = st.selectbox(
            "공분산 추정 방법",
            options=['sample', 'exponential', 'ledoit_wolf'],
            format_func=lambda x: {
                'sample': '표본 공분산',
                'exponential': '지수가중 공분산',
                'ledoit_wolf': 'Ledoit-Wolf 추정'
            }[x]
        )
    
    with col2:
        max_weight = st.slider("최대 종목 비중 (%)", 5, 50, 15) / 100
        min_weight = st.slider("최소 종목 비중 (%)", 0, 10, 1) / 100
        
        if optimization_method == 'mean_variance':
            risk_aversion = st.slider("리스크 회피도", 0.1, 10.0, 1.0, 0.1)
    
    # 사용할 팩터 선택
    st.subheader("📈 사용할 팩터 선택 (선택사항)")
    
    use_factor = st.checkbox("팩터 점수를 기대수익률로 사용", value=False)
    selected_factor = None
    
    if use_factor and st.session_state.calculated_factors:
        selected_factor = st.selectbox(
            "팩터 선택",
            options=list(st.session_state.calculated_factors.keys()),
            help="최적화에 사용할 팩터를 선택하세요"
        )
    
    # 최적화 실행
    if st.button("🎯 포트폴리오 최적화", type="primary"):
        with st.spinner("포트폴리오 최적화를 실행하는 중..."):
            try:
                # 제약조건 설정
                constraints = OptimizationConstraints(
                    max_weight=max_weight,
                    min_weight=min_weight,
                    long_only=True
                )
                
                # 팩터 점수 준비
                factor_scores = None
                if use_factor and selected_factor:
                    factor_values = st.session_state.calculated_factors[selected_factor]
                    if isinstance(factor_values, pd.DataFrame):
                        factor_scores = factor_values
                    else:
                        # Series를 DataFrame으로 변환
                        factor_scores = pd.DataFrame({col: factor_values for col in data['prices'].columns})
                
                # 최적화 실행
                kwargs = {}
                if optimization_method == 'mean_variance':
                    kwargs['risk_aversion'] = risk_aversion
                
                result = st.session_state.portfolio_optimizer.optimize_portfolio(
                    returns=data['returns'],
                    factor_scores=factor_scores,
                    method=optimization_method,
                    covariance_method=covariance_method,
                    constraints=constraints,
                    **kwargs
                )
                
                st.session_state.optimization_results = result
                
                # 결과를 데이터베이스에 저장
                factor_ids = [1] if use_factor else []  # 실제 팩터 ID로 교체 필요
                st.session_state.db_manager.save_optimization_results(
                    factor_ids=factor_ids,
                    method=optimization_method,
                    weights=result.weights,
                    expected_return=result.expected_return,
                    expected_volatility=result.expected_volatility,
                    sharpe_ratio=result.sharpe_ratio
                )
                
                st.success("포트폴리오 최적화가 완료되었습니다!")
                
            except Exception as e:
                st.error(f"최적화 오류: {str(e)}")
    
    # 최적화 결과 표시
    if st.session_state.optimization_results is not None:
        st.subheader("🎯 최적화 결과")
        
        result = st.session_state.optimization_results
        
        # 성과 지표
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("기대 수익률", f"{result.expected_return:.2%}")
        
        with col2:
            st.metric("기대 변동성", f"{result.expected_volatility:.2%}")
        
        with col3:
            st.metric("샤프 비율", f"{result.sharpe_ratio:.3f}")
        
        # 포트폴리오 구성
        st.subheader("📊 포트폴리오 구성")
        
        # 가중치 테이블
        weights_df = pd.DataFrame({
            '종목': result.weights.index,
            '비중 (%)': result.weights.values * 100
        }).sort_values('비중 (%)', ascending=False)
        
        weights_df = weights_df[weights_df['비중 (%)'] > 0.01]  # 0.01% 이상만 표시
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(weights_df, use_container_width=True)
        
        with col2:
            # 파이 차트
            fig = px.pie(weights_df, values='비중 (%)', names='종목', 
                        title="포트폴리오 구성")
            st.plotly_chart(fig, use_container_width=True)
        
        # 효율적 경계선 (평균-분산 최적화인 경우)
        if optimization_method == 'mean_variance':
            st.subheader("📈 효율적 경계선")
            
            with st.spinner("효율적 경계선을 생성하는 중..."):
                try:
                    risks, returns = st.session_state.portfolio_optimizer.create_efficient_frontier(
                        data['returns']
                    )
                    
                    if risks and returns:
                        fig = go.Figure()
                        
                        # 효율적 경계선
                        fig.add_trace(go.Scatter(
                            x=risks, y=returns,
                            mode='lines+markers',
                            name='효율적 경계선',
                            line=dict(color='blue')
                        ))
                        
                        # 최적 포트폴리오 점
                        fig.add_trace(go.Scatter(
                            x=[result.expected_volatility], 
                            y=[result.expected_return],
                            mode='markers',
                            name='최적 포트폴리오',
                            marker=dict(color='red', size=10)
                        ))
                        
                        fig.update_layout(
                            title="효율적 경계선",
                            xaxis_title="리스크 (변동성)",
                            yaxis_title="기대 수익률",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"효율적 경계선 생성 오류: {str(e)}")

# 데이터 관리 페이지
def render_data_management():
    """데이터베이스 관리 페이지 - 강화된 기능"""
    st.title("📊 데이터베이스 관리")
    st.write("저장된 팩터, 백테스팅 결과 및 포트폴리오 데이터를 관리하고 모니터링합니다.")
    
    # 데이터베이스 통계
    st.subheader("📈 데이터베이스 현황")
    
    db_stats = st.session_state.db_manager.get_database_stats()
    
    if db_stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("팩터 정의", db_stats.get('factor_definitions_count', 0))
            st.metric("팩터 값", f"{db_stats.get('factor_values_count', 0):,}")
        
        with col2:
            st.metric("백테스팅 결과", db_stats.get('backtest_results_count', 0))
            st.metric("최적화 결과", db_stats.get('optimization_results_count', 0))
        
        with col3:
            st.metric("시장 데이터", f"{db_stats.get('market_data_count', 0):,}")
            st.metric("DB 크기", f"{db_stats.get('database_size_mb', 0)} MB")
        
        with col4:
            if db_stats.get('last_factor_created'):
                st.metric("최근 팩터 생성", db_stats['last_factor_created'][:10])
            if db_stats.get('last_backtest'):
                st.metric("최근 백테스팅", db_stats['last_backtest'][:10])
    
    # 저장된 팩터 목록
    st.subheader("📈 저장된 팩터")
    
    factor_list = st.session_state.db_manager.get_factor_list()
    
    if factor_list:
        factors_df = pd.DataFrame(factor_list)
        factors_df['created_at'] = pd.to_datetime(factors_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(factors_df, use_container_width=True)
        
        # 팩터 로드
        selected_factor_id = st.selectbox(
            "로드할 팩터 선택",
            options=[f["id"] for f in factor_list],
            format_func=lambda x: next(f["name"] for f in factor_list if f["id"] == x)
        )
        
        if st.button("팩터 로드"):
            factor_data = st.session_state.db_manager.load_factor_values(selected_factor_id)
            if factor_data is not None:
                st.session_state.calculated_factors[f'loaded_factor_{selected_factor_id}'] = factor_data
                st.success("팩터를 성공적으로 로드했습니다!")
            else:
                st.error("팩터 로드에 실패했습니다.")
    else:
        st.info("저장된 팩터가 없습니다.")
    
    # 백테스팅 기록
    st.subheader("📊 백테스팅 기록")
    
    backtest_history = st.session_state.db_manager.get_backtest_history()
    
    if backtest_history:
        history_df = pd.DataFrame(backtest_history)
        history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        # 성과 지표 추출 (예시)
        for idx, row in history_df.iterrows():
            metrics = row.get('performance_metrics', {})
            history_df.at[idx, 'Total Return'] = f"{metrics.get('Total Return', 0):.2%}"
            history_df.at[idx, 'Sharpe Ratio'] = f"{metrics.get('Sharpe Ratio', 0):.3f}"
        
        display_columns = ['factor_name', 'Total Return', 'Sharpe Ratio', 'created_at']
        st.dataframe(history_df[display_columns], use_container_width=True)
    else:
        st.info("백테스팅 기록이 없습니다.")
    
    # 데이터 내보내기
    st.subheader("📤 데이터 내보내기")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_table = st.selectbox(
            "내보낼 테이블",
            ["factor_definitions", "factor_values", "backtest_results", 
             "optimization_results", "market_data"]
        )
    
    with col2:
        export_format = st.selectbox("파일 형식", ["csv", "json"])
    
    with col3:
        if st.button("데이터 내보내기"):
            try:
                filename = st.session_state.db_manager.export_data(export_table, export_format)
                if filename:
                    st.success(f"데이터를 {filename}로 내보냈습니다.")
                    # 다운로드 링크 생성
                    with open(filename, 'rb') as f:
                        st.download_button(
                            label=f"📥 {filename} 다운로드",
                            data=f.read(),
                            file_name=filename,
                            mime='text/csv' if export_format == 'csv' else 'application/json'
                        )
            except Exception as e:
                st.error(f"데이터 내보내기 오류: {str(e)}")
    
    # 데이터베이스 관리
    st.subheader("🔧 데이터베이스 관리")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("데이터베이스 최적화", help="VACUUM을 실행하여 데이터베이스를 최적화합니다"):
            try:
                st.session_state.db_manager.vacuum_database()
                st.success("데이터베이스가 최적화되었습니다.")
            except Exception as e:
                st.error(f"최적화 오류: {str(e)}")
    
    with col2:
        if st.button("데이터베이스 백업", help="현재 데이터베이스를 백업합니다"):
            try:
                backup_file = st.session_state.db_manager.backup_database()
                if backup_file:
                    st.success(f"백업 완료: {backup_file}")
                    # 백업 파일 다운로드
                    with open(backup_file, 'rb') as f:
                        st.download_button(
                            label=f"📥 {backup_file} 다운로드",
                            data=f.read(),
                            file_name=backup_file,
                            mime='application/octet-stream'
                        )
            except Exception as e:
                st.error(f"백업 오류: {str(e)}")
    
    with col3:
        cleanup_days = st.number_input("정리할 데이터 기간 (일)", min_value=1, max_value=365, value=30)
        if st.button("오래된 데이터 정리"):
            try:
                st.session_state.db_manager.cleanup_old_data(cleanup_days)
                st.success(f"{cleanup_days}일 이전 데이터를 정리했습니다.")
                st.rerun()
            except Exception as e:
                st.error(f"데이터 정리 오류: {str(e)}")
    
    # 직접 SQL 쿼리 실행 (고급 사용자용)
    st.subheader("⚡ 고급 쿼리")
    
    with st.expander("SQL 쿼리 실행 (고급 사용자용)", expanded=False):
        st.warning("⚠️ 주의: 직접 SQL 쿼리를 실행하면 데이터가 손상될 수 있습니다.")
        
        sql_query = st.text_area(
            "SQL 쿼리 입력",
            placeholder="SELECT * FROM factor_definitions LIMIT 10;",
            help="SELECT 문만 실행 가능합니다."
        )
        
        if st.button("쿼리 실행") and sql_query.strip():
            if sql_query.strip().upper().startswith('SELECT'):
                try:
                    import sqlite3
                    with sqlite3.connect(st.session_state.db_manager.db_path) as conn:
                        result_df = pd.read_sql_query(sql_query, conn)
                        st.dataframe(result_df, use_container_width=True)
                except Exception as e:
                    st.error(f"쿼리 실행 오류: {str(e)}")
            else:
                st.error("보안상 SELECT 쿼리만 실행할 수 있습니다.")

def render_enhanced_factors(data, category):
    """향상된 팩터 인터페이스"""
    st.subheader(f"🔬 {category.replace('_', ' ').title()} 팩터")
    
    # 사용 가능한 팩터 목록 가져오기
    enhanced_library = st.session_state.enhanced_factor_library
    available_factors = enhanced_library.get_available_factors()
    
    if category not in available_factors:
        st.error(f"지원하지 않는 카테고리: {category}")
        return
    
    factor_options = available_factors[category]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_factor = st.selectbox(
            "팩터 선택",
            options=factor_options,
            format_func=lambda x: enhanced_library.get_factor_description(category, x).split(' - ')[0]
        )
        
        # 팩터 설명 표시
        factor_description = enhanced_library.get_factor_description(category, selected_factor)
        st.info(f"📖 **팩터 설명**: {factor_description}")
        
        # 카테고리별 특수 파라미터
        kwargs = {}
        
        if category == 'advanced_math':
            if selected_factor == 'kalman_filter':
                kwargs['observation_noise'] = st.slider("관측 노이즈", 0.01, 1.0, 0.1)
                kwargs['process_noise'] = st.slider("프로세스 노이즈", 0.001, 0.1, 0.01)
            elif selected_factor == 'regime_detection':
                kwargs['n_components'] = st.slider("체제 수", 2, 5, 2)
            elif selected_factor == 'hurst_exponent':
                kwargs['min_window'] = st.slider("최소 윈도우", 10, 100, 20)
                kwargs['max_window'] = st.slider("최대 윈도우", 50, 300, 100)
            elif selected_factor == 'wavelet':
                wavelet_type = st.selectbox("웨이블릿 타입", ['db4', 'db8', 'haar', 'coif2'])
                kwargs['wavelet'] = wavelet_type
                kwargs['levels'] = st.slider("분해 레벨", 1, 5, 3)
            elif selected_factor == 'isolation_forest':
                kwargs['contamination'] = st.slider("이상치 비율", 0.01, 0.5, 0.1)
        
        elif category == 'deep_learning':
            if selected_factor == 'lstm':
                kwargs['lookback'] = st.slider("룩백 기간", 10, 100, 60)
                kwargs['lstm_units'] = st.slider("LSTM 유닛 수", 16, 128, 50)
                kwargs['epochs'] = st.slider("학습 에포크", 10, 100, 50)
            elif selected_factor == 'attention':
                kwargs['sequence_length'] = st.slider("시퀀스 길이", 10, 100, 30)
            elif selected_factor == 'ensemble':
                kwargs['n_models'] = st.slider("모델 수", 3, 10, 5)
        
        # 공통 파라미터
        kwargs['window'] = st.slider("윈도우 크기", 5, 100, 20)
        
    with col2:
        st.markdown("### 📊 팩터 미리보기")
        
        if st.button("🔄 팩터 계산", type="primary"):
            try:
                with st.spinner(f"{selected_factor} 팩터를 계산 중..."):
                    # 향상된 팩터 라이브러리로 계산
                    factor_result = enhanced_library.calculate_factor(
                        category, selected_factor, data, **kwargs
                    )
                    
                    if factor_result is not None and not factor_result.empty:
                        # 결과 저장
                        factor_key = f"{category}_{selected_factor}"
                        st.session_state.calculated_factors[factor_key] = factor_result
                        
                        # 데이터베이스에 저장
                        factor_id = st.session_state.db_manager.save_factor_definition(
                            name=factor_key,
                            category=category,
                            description=factor_description,
                            parameters=kwargs
                        )
                        
                        if factor_id:
                            # 무한값과 NaN 제거
                            clean_result = factor_result.copy()
                            if isinstance(clean_result, pd.DataFrame):
                                clean_result = clean_result.replace([np.inf, -np.inf], np.nan)
                                clean_result = clean_result.dropna(how='all')
                            elif isinstance(clean_result, pd.Series):
                                clean_result = clean_result.replace([np.inf, -np.inf], np.nan)
                                clean_result = clean_result.dropna()
                            
                            st.session_state.db_manager.save_factor_values(
                                factor_id, clean_result
                            )
                        
                        st.success(f"✅ {selected_factor} 팩터가 계산되었습니다!")
                        
                        # 미리보기 표시
                        if isinstance(factor_result, pd.DataFrame):
                            st.line_chart(factor_result.tail(50))
                            
                            # 기본 통계
                            st.write("**기본 통계**")
                            stats = factor_result.describe()
                            st.dataframe(stats.tail(3))  # count, mean, std만 표시
                        else:
                            st.line_chart(factor_result.tail(50))
                            
                    else:
                        st.error("팩터 계산에 실패했습니다.")
                        
            except Exception as e:
                st.error(f"팩터 계산 오류: {str(e)}")
    
    # 최신 팩터 결과 표시 (이미 계산된 경우)
    factor_key = f"{category}_{selected_factor}"
    if factor_key in st.session_state.calculated_factors:
        st.subheader("📈 최신 계산 결과")
        
        latest_result = st.session_state.calculated_factors[factor_key]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**최근 팩터 값**")
            if isinstance(latest_result, pd.DataFrame):
                st.dataframe(latest_result.tail(10))
            else:
                st.dataframe(pd.DataFrame(latest_result.tail(10)))
        
        with col2:
            st.write("**팩터 성과 통계**")
            if isinstance(latest_result, pd.DataFrame):
                # 각 종목별 통계
                stats_df = pd.DataFrame({
                    '평균': latest_result.mean(),
                    '표준편차': latest_result.std(),
                    '최대값': latest_result.max(),
                    '최소값': latest_result.min()
                })
                st.dataframe(stats_df)
            else:
                stats = latest_result.describe()
                st.dataframe(pd.DataFrame(stats).tail(4))

# 메인 앱
def main():
    """메인 애플리케이션"""
    render_sidebar()
    
    # 페이지 네비게이션
    page = st.sidebar.selectbox(
        "페이지 선택",
        ["대시보드", "팩터 생성", "백테스팅", "포트폴리오 최적화", "데이터 관리"],
        help="원하는 기능의 페이지를 선택하세요"
    )
    
    # 페이지 렌더링
    if page == "대시보드":
        render_dashboard()
    elif page == "팩터 생성":
        render_factor_generator()
    elif page == "백테스팅":
        render_backtesting()
    elif page == "포트폴리오 최적화":
        render_portfolio_optimization()
    elif page == "데이터 관리":
        render_data_management()

if __name__ == "__main__":
    main()
