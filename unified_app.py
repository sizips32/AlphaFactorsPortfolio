"""
통합 알파 팩터 포트폴리오 시스템 - 메인 UI
모든 기능을 하나로 통합한 Streamlit 애플리케이션

작성자: AI Assistant
작성일: 2025년 1월
버전: 1.0 (통합 UI)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List
import warnings

from unified_alpha_system import UnifiedAlphaSystem, UnifiedSystemConfig

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="🚀 통합 알파 팩터 시스템",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 타이틀
st.title("🚀 통합 알파 팩터 포트폴리오 시스템")
st.markdown("""
### 완전한 퀀트 투자 솔루션
**팩터 분석 + Z-Score 표준화 + 포트폴리오 최적화 + 백테스팅 + 리스크 관리**를 하나로 통합
""")

# 사이드바 설정
st.sidebar.header("🔧 시스템 설정")

# 투자 유니버스 선택
st.sidebar.subheader("📊 투자 유니버스")
universe_options = {
    "미국 대형주": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"],
    "한국 대형주": ["005930.KS", "000660.KS", "035420.KS", "051910.KS", "035720.KS", "068270.KS", "207940.KS", "005380.KS", "006400.KS", "003670.KS"],
    "글로벌 혼합": ["AAPL", "MSFT", "005930.KS", "GOOGL", "AMZN", "000660.KS", "TSLA", "035420.KS", "META", "051910.KS"],
    "섹터 ETF": ["XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE"],
    "커스텀": []
}

selected_universe = st.sidebar.selectbox("투자 유니버스 선택", list(universe_options.keys()))

if selected_universe == "커스텀":
    custom_input = st.sidebar.text_area(
        "티커 입력 (쉼표로 구분)",
        placeholder="예: AAPL, MSFT, GOOGL"
    )
    if custom_input:
        tickers = [ticker.strip().upper() for ticker in custom_input.split(",")]
    else:
        tickers = universe_options["미국 대형주"]
else:
    tickers = universe_options[selected_universe]

# 분석 설정
st.sidebar.subheader("⚙️ 분석 파라미터")

# 기간 설정
analysis_period = st.sidebar.selectbox(
    "분석 기간",
    ["1년", "2년", "3년", "5년"],
    index=1
)
period_map = {"1년": "1y", "2년": "2y", "3년": "3y", "5년": "5y"}

# Z-Score 설정
enable_zscore = st.sidebar.checkbox("Z-Score 분석 활성화", value=True)
zscore_threshold = st.sidebar.slider("Z-Score 임계값", 0.5, 3.0, 1.5, 0.1)

# 포트폴리오 설정
st.sidebar.subheader("📈 포트폴리오 설정")
initial_capital = st.sidebar.number_input("초기 자본 (원)", min_value=100000, value=1000000, step=100000)
max_position = st.sidebar.slider("최대 종목 비중 (%)", 5, 30, 15)
transaction_cost = st.sidebar.slider("거래비용 (%)", 0.0, 1.0, 0.1, 0.05)

# 리밸런싱 설정
rebalance_freq = st.sidebar.selectbox(
    "리밸런싱 주기",
    ["daily", "weekly", "monthly", "quarterly"],
    index=2
)

# 고급 기능 설정
st.sidebar.subheader("🔬 고급 분석")
enable_ensemble = st.sidebar.checkbox("앙상블 모델 활성화", value=True)
enable_hedging = st.sidebar.checkbox("동적 헤징 활성화", value=False)
enable_alerts = st.sidebar.checkbox("실시간 알림 활성화", value=True)

# 시스템 설정 생성
config = UnifiedSystemConfig(
    start_date=datetime.now() - timedelta(days=int(period_map[analysis_period][:-1]) * 365),
    end_date=datetime.now(),
    initial_capital=initial_capital,
    max_position_size=max_position / 100,
    transaction_cost=transaction_cost / 100,
    enable_zscore=enable_zscore,
    enable_ensemble=enable_ensemble,
    enable_hedging=enable_hedging,
    zscore_threshold_high=zscore_threshold,
    zscore_threshold_low=-zscore_threshold,
    rebalance_frequency=rebalance_freq
)

# 메인 애플리케이션
def main():
    """메인 애플리케이션"""
    
    # 시스템 초기화
    if 'unified_system' not in st.session_state:
        st.session_state.unified_system = UnifiedAlphaSystem(config)
    
    system = st.session_state.unified_system
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📖 사용 가이드", 
        "🏠 대시보드", 
        "🔢 팩터 분석", 
        "📈 포트폴리오", 
        "📊 백테스팅", 
        "⚙️ 시스템 관리"
    ])
    
    with tab1:
        display_user_guide()
    
    with tab2:
        display_dashboard(system, tickers)
    
    with tab3:
        display_factor_analysis(system, tickers)
    
    with tab4:
        display_portfolio_management(system, tickers)
    
    with tab5:
        display_backtesting(system, tickers)
    
    with tab6:
        display_system_management(system)

def display_dashboard(system: UnifiedAlphaSystem, tickers: List[str]):
    """통합 대시보드"""
    st.header("🏠 통합 분석 대시보드")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📋 분석 개요")
        st.write(f"**분석 대상**: {len(tickers)}개 종목")
        st.write(f"**분석 기간**: {analysis_period}")
        st.write(f"**초기 자본**: {initial_capital:,}원")
        
        # 빠른 실행 버튼
        if st.button("🚀 완전 분석 실행", type="primary", use_container_width=True):
            with st.spinner("분석 실행 중..."):
                results = system.run_complete_analysis(tickers)
                st.session_state.analysis_results = results
                st.rerun()
    
    with col2:
        st.subheader("📊 시스템 상태")
        
        # 데이터베이스 통계
        db_stats = system.database.get_database_stats()
        st.metric("저장된 팩터", db_stats.get('factor_definitions_count', 0))
        st.metric("백테스팅 기록", db_stats.get('backtest_results_count', 0))
        st.metric("DB 크기", f"{db_stats.get('database_size_mb', 0)} MB")
    
    with col3:
        st.subheader("⚡ 빠른 액션")
        
        if st.button("📈 최신 데이터 업데이트"):
            # 캐시 클리어
            st.cache_data.clear()
            st.success("✅ 캐시 클리어 완료")
        
        if st.button("💾 데이터베이스 백업"):
            backup_path = system.database.backup_database()
            if backup_path:
                st.success(f"✅ 백업 완료: {backup_path}")
        
        if st.button("🧹 오래된 데이터 정리"):
            system.database.cleanup_old_data(30)
            st.success("✅ 30일 이전 데이터 정리 완료")
    
    # 최근 분석 결과 표시
    if 'analysis_results' in st.session_state:
        st.divider()
        st.subheader("📊 최근 분석 결과")
        
        results = st.session_state.analysis_results
        
        if 'portfolio_results' in results:
            portfolio = results['portfolio_results']
            
            # 핵심 지표 요약
            col1, col2, col3, col4 = st.columns(4)
            
            if 'optimized_portfolio' in portfolio:
                opt_portfolio = portfolio['optimized_portfolio']
                
                with col1:
                    st.metric(
                        "예상 연수익률",
                        f"{opt_portfolio.get('expected_return', 0) * 100:.2f}%"
                    )
                with col2:
                    st.metric(
                        "예상 변동성",
                        f"{opt_portfolio.get('expected_volatility', 0) * 100:.2f}%"
                    )
                with col3:
                    st.metric(
                        "샤프 비율",
                        f"{opt_portfolio.get('sharpe_ratio', 0):.3f}"
                    )
                with col4:
                    if 'backtest_results' in portfolio:
                        backtest = portfolio['backtest_results']
                        total_return = backtest.get('performance_metrics', {}).get('Total Return', 0)
                        st.metric(
                            "백테스트 수익률",
                            f"{total_return * 100:.2f}%"
                        )

def display_factor_analysis(system: UnifiedAlphaSystem, tickers: List[str]):
    """팩터 분석 탭"""
    st.header("🔢 팩터 분석")
    
    # 팩터 계산 버튼
    if st.button("🔄 팩터 분석 실행"):
        with st.spinner("팩터 분석 중..."):
            # 데이터 로딩
            raw_data = system.load_market_data(tickers, period_map[analysis_period])
            processed_data = system.process_market_data(raw_data)
            
            # 팩터 계산
            factor_results = system.calculate_all_factors(processed_data)
            
            st.session_state.factor_analysis = {
                'data': processed_data,
                'results': factor_results
            }
            st.rerun()
    
    # 팩터 분석 결과 표시
    if 'factor_analysis' in st.session_state:
        factor_data = st.session_state.factor_analysis
        results = factor_data['results']
        
        if results:
            # Z-Score 팩터 히트맵
            if 'zscore_factors' in results and results['zscore_factors']:
                st.subheader("🔥 Z-Score 팩터 히트맵")
                
                zscore_df = pd.DataFrame()
                for factor_type, scores in results['zscore_factors'].items():
                    if isinstance(scores, pd.Series) and not scores.empty:
                        # 최근 30일 데이터만 표시
                        recent_scores = scores.tail(30) if len(scores) > 30 else scores
                        zscore_df[factor_type] = recent_scores
                
                if not zscore_df.empty:
                    fig = px.imshow(
                        zscore_df.T,
                        title="Z-Score 팩터 히트맵 (최근 30일)",
                        color_continuous_scale='RdYlBu_r',
                        aspect='auto',
                        labels={'x': '날짜', 'y': '팩터', 'color': 'Z-Score'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 팩터 통계 요약
            col1, col2 = st.columns(2)
            
            with col1:
                if 'factor_statistics' in results:
                    st.subheader("📊 팩터 통계 요약")
                    
                    stats_df = []
                    for factor_type, stats in results['factor_statistics'].items():
                        stats_df.append({
                            'Factor': factor_type.title(),
                            'Mean': f"{stats.get('mean', 0):.3f}",
                            'Std': f"{stats.get('std', 0):.3f}",
                            'Min': f"{stats.get('min', 0):.3f}",
                            'Max': f"{stats.get('max', 0):.3f}",
                            'Skewness': f"{stats.get('skewness', 0):.3f}"
                        })
                    
                    if stats_df:
                        st.dataframe(pd.DataFrame(stats_df), use_container_width=True)
            
            with col2:
                if 'factor_correlation' in results and not results['factor_correlation'].empty:
                    st.subheader("🔗 팩터 상관관계")
                    
                    fig = px.imshow(
                        results['factor_correlation'],
                        title="팩터 간 상관관계",
                        color_continuous_scale='RdBu_r',
                        aspect='auto'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 개별 팩터 시계열
            if results['zscore_factors']:
                st.subheader("📈 팩터 시계열 분석")
                
                selected_factor = st.selectbox(
                    "표시할 팩터 선택",
                    list(results['zscore_factors'].keys())
                )
                
                if selected_factor in results['zscore_factors']:
                    factor_series = results['zscore_factors'][selected_factor]
                    
                    if isinstance(factor_series, pd.Series) and not factor_series.empty:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=factor_series.index,
                            y=factor_series.values,
                            mode='lines',
                            name=f'{selected_factor.title()} Z-Score',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Z-Score 임계값 라인
                        fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="매수 임계값 (+1.5)")
                        fig.add_hline(y=-1.5, line_dash="dash", line_color="green", annotation_text="매도 임계값 (-1.5)")
                        fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="중립선")
                        
                        fig.update_layout(
                            title=f"{selected_factor.title()} 팩터 Z-Score 추이",
                            xaxis_title="날짜",
                            yaxis_title="Z-Score",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

def display_portfolio_management(system: UnifiedAlphaSystem, tickers: List[str]):
    """포트폴리오 관리 탭"""
    st.header("📈 포트폴리오 관리")
    
    # 포트폴리오 최적화 설정
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ 최적화 설정")
        
        optimization_method = st.selectbox(
            "최적화 방법",
            ["mean_variance", "risk_parity"],
            format_func=lambda x: {"mean_variance": "평균-분산 최적화", "risk_parity": "리스크 패리티"}[x]
        )
        
        covariance_method = st.selectbox(
            "공분산 추정 방법",
            ["sample", "exponential", "ledoit_wolf"],
            format_func=lambda x: {"sample": "표본 공분산", "exponential": "지수가중", "ledoit_wolf": "Ledoit-Wolf"}[x]
        )
        
        max_assets = st.slider("최대 종목 수", 5, min(20, len(tickers)), 10)
    
    with col2:
        st.subheader("📊 현재 설정")
        st.write(f"**투자 대상**: {len(tickers)}개 종목")
        st.write(f"**최적화 방법**: {optimization_method}")
        st.write(f"**공분산 방법**: {covariance_method}")
        st.write(f"**최대 종목 수**: {max_assets}개")
        st.write(f"**최대 비중**: {max_position}%")
        st.write(f"**Z-Score 임계값**: ±{zscore_threshold}")
    
    # 포트폴리오 생성 실행
    if st.button("🎯 포트폴리오 최적화 실행", type="primary"):
        with st.spinner("포트폴리오 최적화 중..."):
            # 데이터 로딩
            raw_data = system.load_market_data(tickers, period_map[analysis_period])
            processed_data = system.process_market_data(raw_data)
            
            # 팩터 계산
            factor_results = system.calculate_all_factors(processed_data)
            
            # 포트폴리오 생성
            portfolio_results = system.create_unified_portfolio(processed_data, factor_results)
            
            st.session_state.portfolio_analysis = {
                'data': processed_data,
                'factors': factor_results,
                'portfolio': portfolio_results
            }
            st.rerun()
    
    # 포트폴리오 결과 표시
    if 'portfolio_analysis' in st.session_state:
        portfolio_data = st.session_state.portfolio_analysis
        portfolio_results = portfolio_data['portfolio']
        
        if portfolio_results:
            st.divider()
            
            # 포트폴리오 구성
            if 'optimized_portfolio' in portfolio_results and 'weights' in portfolio_results['optimized_portfolio']:
                weights = portfolio_results['optimized_portfolio']['weights']
                
                st.subheader("💼 최적화 포트폴리오 구성")
                
                # 포트폴리오 비중 차트
                top_weights = weights.sort_values(ascending=False).head(max_assets)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 파이 차트
                    fig = px.pie(
                        values=top_weights.values * 100,
                        names=top_weights.index,
                        title=f"포트폴리오 구성 (상위 {len(top_weights)}개)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 바 차트
                    fig = px.bar(
                        x=top_weights.values * 100,
                        y=top_weights.index,
                        orientation='h',
                        title="종목별 비중 (%)",
                        labels={'x': '비중 (%)', 'y': '종목'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 포트폴리오 상세 정보
                st.subheader("📋 포트폴리오 상세")
                
                portfolio_detail = pd.DataFrame({
                    'Symbol': top_weights.index,
                    'Weight (%)': (top_weights.values * 100).round(2),
                    'Z-Score': [portfolio_results.get('composite_score', pd.Series()).get(symbol, 0) 
                               for symbol in top_weights.index]
                })
                
                st.dataframe(portfolio_detail, use_container_width=True)
            
            # 리스크 분석 결과
            if 'risk_analysis' in portfolio_results:
                st.subheader("⚠️ 리스크 분석")
                
                risk = portfolio_results['risk_analysis']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("포트폴리오 변동성", f"{risk.get('portfolio_volatility', 0) * 100:.2f}%")
                with col2:
                    st.metric("VaR (95%)", f"{risk.get('var_95', 0) * 100:.2f}%")
                with col3:
                    st.metric("최대 손실", f"{risk.get('max_drawdown', 0) * 100:.2f}%")
                with col4:
                    st.metric("베타", f"{risk.get('beta', 0):.3f}")

def display_factor_analysis(system: UnifiedAlphaSystem, tickers: List[str]):
    """팩터 분석 탭"""
    st.header("🔢 상세 팩터 분석")
    
    # 분석할 팩터 카테고리 선택
    available_categories = system.factor_engine.get_available_factors()
    
    selected_categories = st.multiselect(
        "분석할 팩터 카테고리",
        list(available_categories.keys()),
        default=['technical', 'advanced_technical']
    )
    
    if not selected_categories:
        st.warning("분석할 팩터 카테고리를 선택해주세요.")
        return
    
    if st.button("🔍 상세 팩터 분석 실행"):
        with st.spinner("상세 팩터 분석 중..."):
            # 데이터 로딩
            raw_data = system.load_market_data(tickers, period_map[analysis_period])
            processed_data = system.process_market_data(raw_data)
            
            if processed_data:
                # 선택된 카테고리별 팩터 계산
                detailed_results = {}
                
                for category in selected_categories:
                    st.write(f"**{category} 팩터 분석 중...**")
                    
                    category_factors = {}
                    factor_names = available_categories[category]
                    
                    for factor_name in factor_names:
                        try:
                            factor_data = system.factor_engine.calculate_factor(
                                category, factor_name, processed_data
                            )
                            
                            if not factor_data.empty:
                                category_factors[factor_name] = factor_data
                                
                                # 팩터 검증
                                if isinstance(factor_data, pd.DataFrame):
                                    validation = system.validator.validate_factor_values(
                                        factor_data, None
                                    )
                                    
                                    if not validation['is_valid']:
                                        st.warning(f"⚠️ {factor_name} 팩터 검증 실패: {validation['errors']}")
                        
                        except Exception as e:
                            st.error(f"❌ {factor_name} 팩터 계산 실패: {str(e)}")
                    
                    detailed_results[category] = category_factors
                
                st.session_state.detailed_factor_analysis = detailed_results
                st.success("✅ 상세 팩터 분석 완료")
    
    # 상세 분석 결과 표시
    if 'detailed_factor_analysis' in st.session_state:
        detailed_results = st.session_state.detailed_factor_analysis
        
        for category, factors in detailed_results.items():
            st.subheader(f"📊 {category.title()} 팩터 결과")
            
            if not factors:
                st.info("계산된 팩터가 없습니다.")
                continue
            
            # 팩터별 시각화
            factor_names = list(factors.keys())
            selected_factor = st.selectbox(f"{category} 팩터 선택", factor_names, key=f"select_{category}")
            
            if selected_factor in factors:
                factor_data = factors[selected_factor]
                
                if isinstance(factor_data, pd.DataFrame):
                    # 다중 시계열 차트
                    fig = go.Figure()
                    
                    for i, column in enumerate(factor_data.columns[:5]):  # 상위 5개만 표시
                        fig.add_trace(go.Scatter(
                            x=factor_data.index,
                            y=factor_data[column],
                            mode='lines',
                            line=dict(color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]),
                            name=str(column),
                            opacity=0.7
                        ))
                    
                    fig.update_layout(
                        title=f"{selected_factor.title()} 팩터 시계열",
                        xaxis_title="날짜",
                        yaxis_title="팩터 값",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 팩터 통계
                    st.write("**팩터 통계 정보**")
                    stats_summary = factor_data.describe()
                    st.dataframe(stats_summary.T, use_container_width=True)
                
                elif isinstance(factor_data, pd.Series):
                    # 단일 시계열 차트
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=factor_data.index,
                        y=factor_data.values,
                        mode='lines',
                        name=selected_factor.title(),
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_factor.title()} 팩터 시계열",
                        xaxis_title="날짜", 
                        yaxis_title="팩터 값",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def display_backtesting(system: UnifiedAlphaSystem, tickers: List[str]):
    """백테스팅 탭"""
    st.header("📊 백테스팅 분석")
    
    # 백테스팅 설정
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ 백테스팅 설정")
        
        portfolio_method = st.selectbox(
            "포트폴리오 구성 방법",
            ["equal_weight", "factor_weight", "rank_weight", "long_short"],
            format_func=lambda x: {
                "equal_weight": "동일가중",
                "factor_weight": "팩터 가중", 
                "rank_weight": "순위 가중",
                "long_short": "롱-숏"
            }[x]
        )
        
        n_assets = st.slider("포트폴리오 종목 수", 5, min(15, len(tickers)), 10)
        
        benchmark = st.selectbox(
            "벤치마크",
            ["SPY", "QQQ", "VTI", "KOSPI200", "없음"],
            index=0
        )
    
    with col2:
        st.subheader("📈 수익률 목표")
        
        target_return = st.slider("목표 연수익률 (%)", 5, 30, 15)
        target_volatility = st.slider("목표 변동성 (%)", 5, 25, 12)
        max_drawdown_limit = st.slider("최대 손실 한계 (%)", 5, 30, 15)
        
        # 백테스팅 설정 요약 표시
        portfolio_method_names = {
            'equal_weight': '동일가중', 
            'factor_weight': '팩터가중', 
            'rank_weight': '순위가중', 
            'long_short': '롱-숏'
        }
        
        st.info(f"""
        **선택된 설정:**
        - 포트폴리오 방법: {portfolio_method_names[portfolio_method]}
        - 종목 수: {n_assets}개
        - 목표 수익률: {target_return}%
        - 목표 변동성: {target_volatility}%
        - 최대손실 한계: {max_drawdown_limit}%
        """)
        
        # 설정 검증
        if target_return > target_volatility * 1.5:
            st.warning("⚠️ 목표 수익률이 변동성 대비 너무 높을 수 있습니다.")
    
    # 백테스팅 실행
    if st.button("🚀 종합 백테스팅 실행", type="primary"):
        with st.spinner("백테스팅 실행 중..."):
            try:
                # 전체 분석 실행
                complete_results = system.run_complete_analysis(tickers)
                
                if complete_results and 'portfolio_results' in complete_results:
                    portfolio_results = complete_results['portfolio_results']
                    
                    if 'backtest_results' in portfolio_results:
                        backtest = portfolio_results['backtest_results']
                        
                        st.session_state.backtest_results = {
                            'complete_results': complete_results,
                            'backtest_data': backtest
                        }
                        
                        st.success("✅ 백테스팅 완료")
                        st.rerun()
                    else:
                        st.error("❌ 백테스팅 결과를 찾을 수 없습니다.")
                else:
                    st.error("❌ 분석 실행 실패")
            
            except Exception as e:
                st.error(f"백테스팅 실행 오류: {str(e)}")
    
    # 백테스팅 결과 표시
    if 'backtest_results' in st.session_state:
        backtest_data = st.session_state.backtest_results['backtest_data']
        
        if 'portfolio_returns' in backtest_data and not backtest_data['portfolio_returns'].empty:
            returns = backtest_data['portfolio_returns']
            
            # 누적 수익률 차트
            st.subheader("📈 포트폴리오 성과")
            
            cumulative_returns = (1 + returns).cumprod()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='포트폴리오',
                line=dict(color='blue', width=3)
            ))
            
            # 벤치마크 추가 (선택적)
            if benchmark != "없음":
                try:
                    benchmark_data = system.load_market_data([benchmark], period_map[analysis_period])
                    if benchmark in benchmark_data:
                        benchmark_prices = benchmark_data[benchmark]['Close']
                        benchmark_returns = benchmark_prices.pct_change().dropna()
                        benchmark_cumulative = (1 + benchmark_returns).cumprod()
                        
                        fig.add_trace(go.Scatter(
                            x=benchmark_cumulative.index,
                            y=benchmark_cumulative.values,
                            mode='lines',
                            name=f'벤치마크 ({benchmark})',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                except:
                    pass
            
            fig.update_layout(
                title="포트폴리오 vs 벤치마크 누적 수익률",
                xaxis_title="날짜",
                yaxis_title="누적 수익률",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 성과 지표
            if 'performance_metrics' in backtest_data:
                metrics = backtest_data['performance_metrics']
                
                st.subheader("📊 성과 지표")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("총 수익률", f"{metrics.get('Total Return', 0) * 100:.2f}%")
                    st.metric("연수익률", f"{metrics.get('Annualized Return', 0) * 100:.2f}%")
                
                with col2:
                    st.metric("변동성", f"{metrics.get('Volatility', 0) * 100:.2f}%")
                    st.metric("샤프 비율", f"{metrics.get('Sharpe Ratio', 0):.3f}")
                
                with col3:
                    st.metric("소르티노 비율", f"{metrics.get('Sortino Ratio', 0):.3f}")
                    st.metric("칼마 비율", f"{metrics.get('Calmar Ratio', 0):.3f}")
                
                with col4:
                    st.metric("최대 손실", f"{metrics.get('Max Drawdown', 0) * 100:.2f}%")
                    st.metric("승률", f"{metrics.get('Win Rate', 0) * 100:.1f}%")
                
                # 상세 지표 테이블
                st.subheader("📋 상세 성과 지표")
                
                detailed_metrics = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': [f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in metrics.values()]
                })
                
                st.dataframe(detailed_metrics, use_container_width=True)

def display_system_management(system: UnifiedAlphaSystem):
    """시스템 관리 탭"""
    st.header("⚙️ 시스템 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 데이터베이스 현황")
        
        db_stats = system.database.get_database_stats()
        
        for key, value in db_stats.items():
            if isinstance(value, (int, float)):
                st.metric(key.replace('_', ' ').title(), f"{value:,}")
            else:
                st.write(f"**{key.replace('_', ' ').title()}**: {value}")
    
    with col2:
        st.subheader("🔧 시스템 관리")
        
        if st.button("🗃️ 데이터베이스 백업"):
            backup_path = system.database.backup_database()
            if backup_path:
                st.success(f"✅ 백업 완료: {backup_path}")
        
        if st.button("🧹 데이터 정리"):
            system.database.cleanup_old_data(30)
            st.success("✅ 30일 이전 데이터 정리 완료")
        
        if st.button("⚡ 데이터베이스 최적화"):
            system.database.vacuum_database()
            st.success("✅ 데이터베이스 최적화 완료")
    
    # 저장된 팩터 목록
    st.subheader("📊 저장된 팩터 목록")
    factor_list = system.database.get_factor_list()
    
    if factor_list:
        factor_df = pd.DataFrame(factor_list)
        st.dataframe(factor_df, use_container_width=True)
    else:
        st.info("저장된 팩터가 없습니다.")
    
    # 백테스팅 기록
    st.subheader("📈 백테스팅 기록")
    backtest_history = system.database.get_backtest_history()
    
    if backtest_history:
        # 성과 지표만 표시
        history_df = []
        for record in backtest_history:
            metrics = record.get('performance_metrics', {})
            history_df.append({
                'Date': record.get('created_at', ''),
                'Factor': record.get('factor_name', ''),
                'Total Return': f"{metrics.get('Total Return', 0) * 100:.2f}%",
                'Sharpe Ratio': f"{metrics.get('Sharpe Ratio', 0):.3f}",
                'Max Drawdown': f"{metrics.get('Max Drawdown', 0) * 100:.2f}%"
            })
        
        if history_df:
            st.dataframe(pd.DataFrame(history_df), use_container_width=True)
    else:
        st.info("백테스팅 기록이 없습니다.")

def display_user_guide():
    """사용법과 해석법 가이드"""
    st.header("📖 통합 알파 팩터 시스템 사용 가이드")
    
    # 개요 섹션
    st.markdown("---")
    st.subheader("🎯 시스템 개요")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 무엇을 할 수 있나요?
        - **60+ 알파팩터** 자동 계산 및 분석
        - **Z-Score 표준화**로 팩터 간 비교 가능
        - **포트폴리오 최적화** (평균분산, 리스크패리티)  
        - **백테스팅**으로 전략 성과 검증
        - **실시간 모니터링** 및 리스크 알림
        """)
        
    with col2:
        st.markdown("""
        ### 🎲 핵심 기능
        - **기술적 팩터**: 모멘텀, 평균회귀, 변동성, 거래량
        - **고급 팩터**: RSI, 볼린저밴드, 상관계수, ML 팩터
        - **리스크 팩터**: 베타, VaR, 하방리스크, 집중도
        - **수학/통계**: 칼만필터, Hurst지수, 웨이블릿 변환
        """)
    
    # 단계별 사용법
    st.markdown("---")
    st.subheader("📋 단계별 사용법")
    
    with st.expander("🔧 **1단계: 초기 설정**", expanded=True):
        st.markdown("""
        **📊 투자 유니버스 선택** (좌측 사이드바)
        - **미국 대형주**: AAPL, MSFT, GOOGL 등 안정적 종목
        - **한국 대형주**: 삼성전자, SK하이닉스 등 코스피 대표주
        - **섹터 ETF**: XLF, XLK 등 분산 투자
        - **커스텀**: 직접 티커 입력 (쉼표로 구분)
        
        **⚙️ 분석 파라미터 설정**
        - **분석 기간**: 1년~5년 (더 길수록 안정적)
        - **Z-Score 임계값**: ±1.5 (보수적) ~ ±2.0 (공격적)
        - **최소 거래량**: 일평균 거래대금 기준
        """)
    
    with st.expander("🏠 **2단계: 대시보드에서 완전분석 실행**"):
        st.markdown("""
        **🚀 "완전 분석 실행" 버튼 클릭**
        - 모든 팩터를 자동으로 계산
        - Z-Score로 표준화하여 비교 가능한 형태로 변환
        - 복합 점수 계산으로 종목 순위 결정
        - 포트폴리오 후보 종목 자동 선별
        
        **⏱️ 예상 소요 시간**
        - 10종목 1년: 약 30초
        - 20종목 3년: 약 1-2분
        """)
    
    with st.expander("🔢 **3단계: 팩터 분석 탭 결과 해석**"):
        st.markdown("""
        **📊 Z-Score 히트맵 해석**
        - **빨간색 (양수)**: 평균 대비 높은 값 → 매수 신호 가능성
        - **파란색 (음수)**: 평균 대비 낮은 값 → 매도 신호 가능성
        - **Z-Score > +2**: 매우 강한 매수 신호 (상위 2.5%)
        - **Z-Score < -2**: 매우 강한 매도 신호 (하위 2.5%)
        
        **📈 팩터별 의미**
        - **모멘텀 팩터**: 추세 지속성 (높을수록 상승 추세)
        - **평균회귀 팩터**: 과매수/과매도 (극값에서 반전 기대)
        - **변동성 팩터**: 리스크 수준 (낮을수록 안정적)
        - **거래량 팩터**: 시장 관심도 (높을수록 활발한 거래)
        """)
        
    with st.expander("📈 **4단계: 포트폴리오 구성**"):
        st.markdown("""
        **🎯 포트폴리오 전략 선택**
        - **동일가중**: 선택 종목을 동일 비중으로 (단순, 안정적)
        - **팩터가중**: Z-Score에 비례하여 가중 (신호 강도 반영)
        - **순위가중**: 순위에 따라 가중 (상위 종목 집중)
        - **롱-숏**: 상위 매수, 하위 매도 (마켓뉴트럴)
        
        **⚖️ 리스크 제약조건**
        - **최대 비중**: 개별 종목 집중도 한계 (보통 10-20%)
        - **최소 비중**: 소액 투자 방지 (보통 1-5%)
        - **섹터 제한**: 특정 섹터 과도 집중 방지
        """)
        
    with st.expander("📊 **5단계: 백테스팅 결과 해석**"):
        st.markdown("""
        **📈 성과 지표 해석**
        - **총 수익률**: 기간 전체 누적 수익률
        - **연간 수익률**: 연환산 수익률 (복리 적용)
        - **샤프 비율**: 위험 대비 수익률 (1.0 이상 양호, 2.0 이상 우수)
        - **최대 손실률**: 고점 대비 최대 하락폭 (낮을수록 좋음)
        
        **🎯 벤치마크 비교**
        - **알파**: 벤치마크 초과 수익률 (양수면 우수)
        - **베타**: 시장 대비 민감도 (1.0=시장과 동일)
        - **정보비율**: 추적오차 대비 초과수익률
        - **승률**: 벤치마크 대비 우수한 기간 비율
        """)
    
    # 해석 가이드라인
    st.markdown("---")
    st.subheader("🧭 결과 해석 가이드라인")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🟢 강한 매수 신호
        - **Z-Score > +1.5**
        - **여러 팩터 동시 양수**  
        - **모멘텀 + 거래량 증가**
        - **리스크 팩터 안정**
        
        **⚠️ 주의사항**
        - 과매수 위험 확인
        - 펀더멘털 검토 필요
        """)
        
    with col2:
        st.markdown("""
        ### 🟡 중립/관망
        - **-1.0 < Z-Score < +1.0**
        - **팩터 간 혼재 신호**
        - **횡보 패턴**
        - **불확실성 높음**
        
        **📊 추가 분석 필요**
        - 장기 트렌드 확인
        - 뉴스/이벤트 점검
        """)
        
    with col3:
        st.markdown("""
        ### 🔴 매도 신호
        - **Z-Score < -1.5**
        - **다수 팩터 음수**
        - **변동성 급증**  
        - **거래량 감소**
        
        **🛡️ 리스크 관리**
        - 손절 기준 설정
        - 포지션 크기 축소
        """)
    
    # 전략별 활용법
    st.markdown("---")
    st.subheader("💡 투자 전략별 활용법")
    
    strategy_tabs = st.tabs(["📈 추세추종", "🔄 평균회귀", "⚖️ 균형투자", "🛡️ 리스크관리"])
    
    with strategy_tabs[0]:
        st.markdown("""
        ### 📈 추세추종 전략
        **핵심 팩터**: 모멘텀, RSI, 이동평균
        
        **매수 조건**
        - 모멘텀 Z-Score > +1.0
        - RSI 팩터 > 0 (50선 돌파)
        - 거래량 팩터 양수 (거래량 증가)
        
        **매도 조건**
        - 모멘텀 Z-Score < -0.5 (추세 약화)
        - 변동성 급증 (리스크 증가)
        
        **⚠️ 주의사항**: 추세 전환점에서 손실 위험
        """)
        
    with strategy_tabs[1]:
        st.markdown("""
        ### 🔄 평균회귀 전략
        **핵심 팩터**: 평균회귀, 볼린저밴드, 상관계수
        
        **매수 조건**
        - 평균회귀 Z-Score > +1.5 (과매도)
        - 볼린저 하단 근접
        - 변동성 팩터 높음 (반등 기대)
        
        **매도 조건**
        - 평균 근처 복귀 (목표가 달성)
        - 과매수 구간 진입
        
        **⚠️ 주의사항**: 하락 트렌드에서 낙하는 칼 위험
        """)
        
    with strategy_tabs[2]:
        st.markdown("""
        ### ⚖️ 균형투자 전략
        **핵심 팩터**: 복합점수, 리스크 팩터, 상관관계
        
        **포트폴리오 구성**
        - 상위 30% 종목 선별
        - 리스크 패리티 가중
        - 정기 리밸런싱 (월/분기)
        
        **리스크 관리**
        - 최대 손실률 15% 제한
        - 섹터/종목 분산
        - 상관관계 모니터링
        
        **장점**: 안정적, 장기 성과, 낮은 변동성
        """)
        
    with strategy_tabs[3]:
        st.markdown("""
        ### 🛡️ 리스크관리 전략
        **핵심 팩터**: VaR, 베타, 하방리스크, 집중도
        
        **리스크 신호**
        - VaR Z-Score < -1.0 (손실 위험 증가)
        - 베타 > 1.5 (시장 민감도 과도)
        - 하방리스크 증가 (하락 위험)
        
        **대응 방안**
        - 포지션 크기 축소
        - 헤지 전략 적용
        - 현금 비중 증가
        
        **⚡ 실시간 모니터링**: 임계값 돌파 시 알림
        """)
    
    # FAQ 섹션
    st.markdown("---")
    st.subheader("❓ 자주 묻는 질문")
    
    with st.expander("Q1. Z-Score가 무엇이고 어떻게 해석하나요?"):
        st.markdown("""
        **Z-Score = (현재값 - 평균) / 표준편차**
        
        - **+2.0 이상**: 상위 2.5% (매우 높음) 🔥
        - **+1.0 ~ +2.0**: 상위 16% (높음) 📈  
        - **-1.0 ~ +1.0**: 정상 범위 (68%) ➡️
        - **-1.0 ~ -2.0**: 하위 16% (낮음) 📉
        - **-2.0 이하**: 하위 2.5% (매우 낮음) ❄️
        
        **장점**: 모든 팩터를 동일한 척도로 비교 가능
        """)
        
    with st.expander("Q2. 어떤 팩터가 가장 중요한가요?"):
        st.markdown("""
        **시장 상황별 중요 팩터**
        - **상승장**: 모멘텀, 성장 팩터
        - **하락장**: 밸류, 퀄리티 팩터  
        - **횡보장**: 평균회귀, 변동성 팩터
        - **변동성 장**: 리스크 팩터, 상관관계
        
        **권장**: 복합점수 활용으로 다중 팩터 종합 판단
        """)
        
    with st.expander("Q3. 백테스팅 결과를 어떻게 신뢰해야 하나요?"):
        st.markdown("""
        **신뢰성 검증 포인트**
        - **충분한 데이터**: 최소 2-3년, 권장 5년
        - **다양한 시장**: 상승/하락/횡보 구간 포함
        - **거래비용 반영**: 실제 수익률과 괴리 최소화
        - **표본외 검증**: 최신 구간에서 성과 유지
        
        **⚠️ 한계**: 과거 성과는 미래를 보장하지 않음
        """)
        
    with st.expander("Q4. 실제 투자할 때 주의사항은?"):
        st.markdown("""
        **🛡️ 필수 리스크 관리**
        - **포지션 사이징**: 총 자산의 1-5% 수준
        - **손절 기준**: -10% ~ -15% 수준 설정
        - **분산투자**: 최소 10-15개 종목
        - **정기 점검**: 월 1회 팩터 업데이트
        
        **📊 추가 분석 필요**
        - 펀더멘털 분석 병행
        - 뉴스/이벤트 모니터링
        - 거시경제 환경 고려
        """)
    
    # 사용 팁
    st.markdown("---")
    st.subheader("💎 고급 사용 팁")
    
    tip_col1, tip_col2 = st.columns(2)
    
    with tip_col1:
        st.markdown("""
        ### 🎯 정확도 향상
        - **데이터 품질**: 상장폐지/합병 종목 제외
        - **생존편향**: 현재 존재 종목만 분석 주의  
        - **섹터 균형**: 특정 섹터 과도 집중 방지
        - **시가총액 고려**: 유동성 충분한 종목 선택
        """)
        
    with tip_col2:
        st.markdown("""
        ### ⚡ 성능 최적화
        - **캐시 활용**: 동일 설정 반복 시 속도 향상
        - **배치 처리**: 여러 종목 동시 분석  
        - **선택적 팩터**: 필요 팩터만 계산으로 속도 증가
        - **정기 최적화**: 데이터베이스 정리로 성능 유지
        """)
    
    # 마무리
    st.markdown("---")
    st.info("""
    ### 🎉 성공적인 퀀트 투자를 위한 마지막 조언
    
    1. **체계적 접근**: 감정보다는 데이터에 기반한 판단
    2. **지속적 학습**: 시장은 변화하므로 전략도 진화 필요
    3. **리스크 우선**: 수익보다 손실 방지가 우선
    4. **장기 관점**: 단기 변동성에 흔들리지 않는 인내심
    5. **겸손한 자세**: 시장은 예측 불가능하다는 인식
    
    **Happy Quant Trading! 📈💰**
    """)

# 앱 실행
if __name__ == "__main__":
    main()