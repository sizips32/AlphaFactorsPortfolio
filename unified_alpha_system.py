"""
통합 알파 팩터 포트폴리오 시스템
모든 모듈을 통합한 완전한 투자 분석 및 포트폴리오 관리 시스템

작성자: AI Assistant
작성일: 2025년 1월
버전: 1.0 (통합 버전)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import logging

# 로컬 모듈 imports
from alpha_factor_library import EnhancedFactorLibrary, FactorValidator, ensure_numeric_dataframe, ensure_numeric_series
from backtesting_engine import BacktestEngine, BacktestConfig, BacktestResults
from portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints
from database import DatabaseManager
from zscore import FactorZScoreCalculator

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UnifiedSystemConfig:
    """통합 시스템 설정"""
    # 데이터 설정
    start_date: datetime = datetime.now() - timedelta(days=730)  # 2년
    end_date: datetime = datetime.now()
    
    # 포트폴리오 설정
    initial_capital: float = 1000000  # 100만원
    max_position_size: float = 0.15   # 최대 15%
    min_position_size: float = 0.01   # 최소 1%
    transaction_cost: float = 0.001   # 0.1%
    
    # 팩터 설정
    enable_zscore: bool = True
    enable_ensemble: bool = True
    enable_hedging: bool = True
    
    # Z-Score 임계값
    zscore_threshold_high: float = 1.5
    zscore_threshold_low: float = -1.5
    
    # 리밸런싱
    rebalance_frequency: str = 'monthly'

class UnifiedAlphaSystem:
    """통합 알파 팩터 포트폴리오 시스템"""
    
    def __init__(self, config: UnifiedSystemConfig = None):
        if config is None:
            config = UnifiedSystemConfig()
        
        self.config = config
        
        # 핵심 엔진들 초기화
        self.factor_engine = EnhancedFactorLibrary()
        self.zscore_calculator = FactorZScoreCalculator()
        self.backtest_engine = BacktestEngine(BacktestConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            rebalance_frequency=config.rebalance_frequency,
            transaction_cost=config.transaction_cost
        ))
        self.optimizer = PortfolioOptimizer()
        self.database = DatabaseManager()
        self.validator = FactorValidator()
        
        # 캐시된 데이터
        self.cached_data = {}
        self.cached_factors = {}
        
    def load_market_data(self, tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """시장 데이터 로딩"""
        try:
            data = {}
            
            st.info(f"📊 {len(tickers)}개 종목 데이터 로딩 중...")
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(tickers):
                try:
                    # 캐시 확인
                    cache_key = f"{ticker}_{period}"
                    if cache_key in self.cached_data:
                        data[ticker] = self.cached_data[cache_key]
                        continue
                    
                    # 데이터베이스에서 먼저 확인
                    cached_data = self.database.get_cached_market_data(
                        ticker, 
                        self.config.start_date.strftime('%Y-%m-%d'),
                        self.config.end_date.strftime('%Y-%m-%d')
                    )
                    
                    if cached_data is not None and len(cached_data) > 100:
                        data[ticker] = cached_data
                        self.cached_data[cache_key] = cached_data
                    else:
                        # Yahoo Finance에서 새로 다운로드
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period=period)
                        
                        if len(hist) > 50:
                            hist = ensure_numeric_dataframe(hist)
                            data[ticker] = hist
                            self.cached_data[cache_key] = hist
                            
                            # 데이터베이스에 캐시
                            self.database.cache_market_data(hist, ticker)
                
                except Exception as e:
                    logger.error(f"데이터 로딩 실패 {ticker}: {str(e)}")
                    continue
                
                progress_bar.progress((i + 1) / len(tickers))
            
            progress_bar.empty()
            
            if data:
                st.success(f"✅ {len(data)}개 종목 데이터 로딩 완료")
            else:
                st.error("❌ 데이터 로딩 실패")
            
            return data
            
        except Exception as e:
            logger.error(f"시장 데이터 로딩 오류: {str(e)}")
            st.error(f"데이터 로딩 오류: {str(e)}")
            return {}
    
    def process_market_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """시장 데이터 전처리"""
        try:
            processed_data = {
                'prices': pd.DataFrame(),
                'volumes': pd.DataFrame(),
                'returns': pd.DataFrame()
            }
            
            # 가격 및 거래량 데이터 정리
            for ticker, data in raw_data.items():
                if 'Close' in data.columns:
                    processed_data['prices'][ticker] = data['Close']
                if 'Volume' in data.columns:
                    processed_data['volumes'][ticker] = data['Volume']
            
            # 수익률 계산
            if not processed_data['prices'].empty:
                processed_data['returns'] = processed_data['prices'].pct_change()
            
            # 결측치 제거 및 정렬
            for key in processed_data:
                if not processed_data[key].empty:
                    processed_data[key] = ensure_numeric_dataframe(processed_data[key])
                    processed_data[key] = processed_data[key].dropna(how='all')
                    processed_data[key] = processed_data[key].sort_index()
            
            return processed_data
            
        except Exception as e:
            logger.error(f"데이터 전처리 오류: {str(e)}")
            return {}
    
    def calculate_all_factors(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """모든 팩터 계산 (Z-Score 포함)"""
        try:
            if not data or 'prices' not in data:
                return {}
            
            results = {
                'raw_factors': {},
                'zscore_factors': {},
                'percentile_ranks': {},
                'factor_statistics': {},
                'factor_correlation': pd.DataFrame()
            }
            
            st.info("🔢 팩터 계산 중...")
            progress = st.progress(0)
            
            # 1. 기본 팩터들 계산
            factor_categories = ['technical', 'advanced_technical', 'machine_learning', 'risk']
            factor_types = {
                'technical': ['momentum', 'mean_reversion', 'volatility', 'volume'],
                'advanced_technical': ['rsi', 'bollinger_bands', 'zscore', 'correlation'],
                'machine_learning': ['random_forest', 'pca'],
                'risk': ['beta', 'downside_risk']
            }
            
            total_factors = sum(len(types) for types in factor_types.values())
            current_step = 0
            
            for category in factor_categories:
                results['raw_factors'][category] = {}
                
                for factor_name in factor_types[category]:
                    try:
                        factor_data = self.factor_engine.calculate_factor(
                            category, factor_name, data
                        )
                        
                        if not factor_data.empty:
                            results['raw_factors'][category][factor_name] = factor_data
                        
                    except Exception as e:
                        logger.error(f"팩터 계산 실패 {category}.{factor_name}: {str(e)}")
                    
                    current_step += 1
                    progress.progress(current_step / (total_factors + 3))
            
            # 2. Z-Score 기반 팩터들 계산 (zscore.py 모듈 활용)
            if self.config.enable_zscore:
                zscore_factors = self.calculate_zscore_factors(data)
                results['zscore_factors'] = zscore_factors
                
                # 백분위 순위 계산
                for factor_type, scores in zscore_factors.items():
                    if isinstance(scores, pd.Series) and not scores.empty:
                        results['percentile_ranks'][factor_type] = \
                            self.zscore_calculator.calculate_percentile_rank(scores)
                
                # 팩터 통계 계산
                for factor_type, scores in zscore_factors.items():
                    if isinstance(scores, pd.Series) and not scores.empty:
                        results['factor_statistics'][factor_type] = \
                            self.zscore_calculator.get_factor_statistics(scores)
            
            progress.progress(0.9)
            
            # 3. 팩터 상관관계 계산
            if results['zscore_factors']:
                correlation_data = pd.DataFrame()
                for factor_type, scores in results['zscore_factors'].items():
                    if isinstance(scores, pd.Series) and not scores.empty:
                        correlation_data[factor_type] = scores
                
                if not correlation_data.empty:
                    results['factor_correlation'] = \
                        self.zscore_calculator.calculate_factor_correlation(correlation_data)
            
            progress.progress(1.0)
            progress.empty()
            
            st.success("✅ 팩터 계산 완료")
            return results
            
        except Exception as e:
            logger.error(f"팩터 계산 오류: {str(e)}")
            st.error(f"팩터 계산 오류: {str(e)}")
            return {}
    
    def calculate_zscore_factors(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Z-Score 기반 팩터 계산"""
        try:
            zscore_factors = {}
            
            if 'prices' not in data:
                return zscore_factors
            
            price_data = data['prices']
            volume_data = data.get('volumes')
            
            # Value 팩터
            value_data = self.zscore_calculator.calculate_value_factor(price_data, volume_data)
            if not value_data.empty:
                zscore_factors['value'] = self.zscore_calculator.calculate_factor_zscore(
                    value_data, 'value'
                )
            
            # Quality 팩터
            quality_data = self.zscore_calculator.calculate_quality_factor(price_data, volume_data)
            if not quality_data.empty:
                zscore_factors['quality'] = self.zscore_calculator.calculate_factor_zscore(
                    quality_data, 'quality'
                )
            
            # Momentum 팩터
            momentum_data = self.zscore_calculator.calculate_momentum_factor(price_data)
            if not momentum_data.empty:
                zscore_factors['momentum'] = self.zscore_calculator.calculate_factor_zscore(
                    momentum_data, 'momentum'
                )
            
            return zscore_factors
            
        except Exception as e:
            logger.error(f"Z-Score 팩터 계산 오류: {str(e)}")
            return {}
    
    def create_unified_portfolio(self, data: Dict[str, pd.DataFrame], 
                               factor_results: Dict[str, Any]) -> Dict[str, Any]:
        """통합 포트폴리오 생성"""
        try:
            if not factor_results or 'zscore_factors' not in factor_results:
                return {}
            
            results = {}
            
            # 1. Z-Score 기반 복합 점수 계산
            composite_score = self.calculate_composite_score(factor_results['zscore_factors'])
            results['composite_score'] = composite_score
            
            # 2. 포트폴리오 후보 선별
            portfolio_candidates = self.select_portfolio_candidates(composite_score)
            results['candidates'] = portfolio_candidates
            
            # 3. 포트폴리오 최적화
            if portfolio_candidates['long_positions'] is not None:
                optimized_portfolio = self.optimize_portfolio(
                    data, portfolio_candidates, factor_results
                )
                results['optimized_portfolio'] = optimized_portfolio
            
            # 4. 리스크 분석
            risk_analysis = self.analyze_portfolio_risk(
                data, results.get('optimized_portfolio', {})
            )
            results['risk_analysis'] = risk_analysis
            
            # 5. 백테스팅
            if 'optimized_portfolio' in results:
                backtest_results = self.run_comprehensive_backtest(
                    data, results['optimized_portfolio']
                )
                results['backtest_results'] = backtest_results
            
            return results
            
        except Exception as e:
            logger.error(f"통합 포트폴리오 생성 오류: {str(e)}")
            return {}
    
    def calculate_composite_score(self, zscore_factors: Dict[str, pd.Series]) -> pd.Series:
        """복합 팩터 점수 계산"""
        try:
            if not zscore_factors:
                return pd.Series()
            
            # 팩터별 가중치 (설정 가능)
            weights = {
                'value': 0.4,
                'quality': 0.3, 
                'momentum': 0.3
            }
            
            # 최신 날짜의 점수들만 사용
            latest_scores = {}
            for factor_type, scores in zscore_factors.items():
                if isinstance(scores, pd.Series) and not scores.empty:
                    latest_scores[factor_type] = scores.iloc[-1] if len(scores) > 0 else 0
            
            # 가중 평균 계산
            composite_score = pd.Series(0.0, index=latest_scores.get('value', pd.Series()).index)
            
            for factor_type, weight in weights.items():
                if factor_type in latest_scores:
                    factor_scores = latest_scores[factor_type]
                    if isinstance(factor_scores, (pd.Series, dict)):
                        composite_score += factor_scores * weight
            
            return composite_score
            
        except Exception as e:
            logger.error(f"복합 점수 계산 오류: {str(e)}")
            return pd.Series()
    
    def select_portfolio_candidates(self, composite_score: pd.Series) -> Dict[str, Any]:
        """포트폴리오 후보 선별"""
        try:
            if composite_score.empty:
                return {'long_positions': None, 'short_positions': None}
            
            # Z-Score 임계값 기준으로 선별
            high_score = composite_score[composite_score > self.config.zscore_threshold_high]
            low_score = composite_score[composite_score < self.config.zscore_threshold_low]
            
            # 상위/하위 N개 선별 (임계값에 상관없이)
            n_positions = min(10, len(composite_score) // 2)
            top_positions = composite_score.nlargest(n_positions)
            bottom_positions = composite_score.nsmallest(n_positions)
            
            return {
                'long_positions': top_positions,
                'short_positions': bottom_positions,
                'high_zscore': high_score,
                'low_zscore': low_score,
                'all_scores': composite_score
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 후보 선별 오류: {str(e)}")
            return {'long_positions': None, 'short_positions': None}
    
    def optimize_portfolio(self, data: Dict[str, pd.DataFrame], 
                          candidates: Dict[str, Any],
                          factor_results: Dict[str, Any]) -> Dict[str, Any]:
        """포트폴리오 최적화"""
        try:
            if not candidates['long_positions'] is not None:
                return {}
            
            # 최적화 제약조건 설정
            constraints = OptimizationConstraints(
                max_weight=self.config.max_position_size,
                min_weight=self.config.min_position_size,
                long_only=True,
                leverage=1.0
            )
            
            # 기대수익률로 복합 점수 사용
            expected_returns = candidates['long_positions'] / candidates['long_positions'].std()
            
            # 포트폴리오 최적화 실행
            optimization_result = self.optimizer.optimize_portfolio(
                returns=data['returns'],
                factor_scores=pd.DataFrame({'composite': expected_returns}),
                method='mean_variance',
                constraints=constraints
            )
            
            return {
                'weights': optimization_result.weights,
                'expected_return': optimization_result.expected_return,
                'expected_volatility': optimization_result.expected_volatility,
                'sharpe_ratio': optimization_result.sharpe_ratio,
                'optimization_status': optimization_result.optimization_status
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 최적화 오류: {str(e)}")
            return {}
    
    def analyze_portfolio_risk(self, data: Dict[str, pd.DataFrame], 
                             portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """포트폴리오 리스크 분석"""
        try:
            if not portfolio or 'weights' not in portfolio:
                return {}
            
            weights = portfolio['weights']
            returns = data.get('returns', pd.DataFrame())
            
            if returns.empty:
                return {}
            
            # 포트폴리오 수익률 계산
            common_assets = weights.index.intersection(returns.columns)
            if len(common_assets) == 0:
                return {}
            
            aligned_weights = weights[common_assets]
            aligned_returns = returns[common_assets]
            
            portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)
            
            # 리스크 지표 계산
            risk_metrics = {
                'portfolio_volatility': portfolio_returns.std() * np.sqrt(252),
                'var_95': portfolio_returns.quantile(0.05),
                'var_99': portfolio_returns.quantile(0.01),
                'max_drawdown': self.calculate_max_drawdown(portfolio_returns),
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0,
                'sortino_ratio': self.calculate_sortino_ratio(portfolio_returns),
                'beta': self.calculate_beta(portfolio_returns, returns.mean(axis=1))
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"리스크 분석 오류: {str(e)}")
            return {}
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """최대 손실 계산"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """소르티노 비율 계산"""
        try:
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return 0.0
            downside_std = downside_returns.std()
            return returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
        except:
            return 0.0
    
    def calculate_beta(self, portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """베타 계산"""
        try:
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance > 0 else 1.0
        except:
            return 1.0
    
    def run_comprehensive_backtest(self, data: Dict[str, pd.DataFrame],
                                 portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """종합 백테스팅"""
        try:
            if not portfolio or 'weights' not in portfolio:
                return {}
            
            weights = portfolio['weights']
            
            # 팩터 점수를 DataFrame 형태로 변환
            factor_scores = pd.DataFrame()
            for date in data['returns'].index:
                factor_scores.loc[date, weights.index] = weights.values
            
            # 백테스팅 실행
            backtest_results = self.backtest_engine.run_backtest(
                factor_scores=factor_scores,
                returns=data['returns'],
                portfolio_method='factor_weight',
                n_assets=len(weights)
            )
            
            return {
                'portfolio_returns': backtest_results.portfolio_returns,
                'performance_metrics': backtest_results.performance_metrics,
                'turnover': backtest_results.turnover.mean() if not backtest_results.turnover.empty else 0,
                'transaction_costs': backtest_results.transaction_costs.sum() if not backtest_results.transaction_costs.empty else 0
            }
            
        except Exception as e:
            logger.error(f"백테스팅 오류: {str(e)}")
            return {}
    
    def generate_alerts(self, factor_results: Dict[str, Any], 
                       portfolio_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """알림 생성"""
        try:
            alerts = []
            
            # Z-Score 기반 알림
            if 'zscore_factors' in factor_results:
                for factor_type, scores in factor_results['zscore_factors'].items():
                    if isinstance(scores, pd.Series) and not scores.empty:
                        latest_score = scores.iloc[-1] if len(scores) > 0 else 0
                        
                        if abs(latest_score) > 2.0:
                            alerts.append({
                                'type': 'EXTREME_FACTOR',
                                'factor': factor_type,
                                'value': latest_score,
                                'message': f"{factor_type} 팩터가 극단값 ({latest_score:.2f})에 도달했습니다.",
                                'severity': 'HIGH'
                            })
            
            # 포트폴리오 리스크 알림
            if 'risk_analysis' in portfolio_results:
                risk = portfolio_results['risk_analysis']
                
                if risk.get('portfolio_volatility', 0) > 0.25:
                    alerts.append({
                        'type': 'HIGH_VOLATILITY',
                        'value': risk['portfolio_volatility'],
                        'message': f"포트폴리오 변동성이 높습니다 ({risk['portfolio_volatility']:.1%})",
                        'severity': 'MEDIUM'
                    })
                
                if risk.get('max_drawdown', 0) < -0.2:
                    alerts.append({
                        'type': 'HIGH_DRAWDOWN', 
                        'value': risk['max_drawdown'],
                        'message': f"최대 손실이 큽니다 ({risk['max_drawdown']:.1%})",
                        'severity': 'HIGH'
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"알림 생성 오류: {str(e)}")
            return []
    
    def save_results_to_database(self, results: Dict[str, Any]):
        """결과를 데이터베이스에 저장"""
        try:
            # 팩터 결과 저장
            if 'factor_results' in results:
                for category, factors in results['factor_results'].get('raw_factors', {}).items():
                    for factor_name, factor_data in factors.items():
                        if not factor_data.empty:
                            # 팩터 정의 저장
                            factor_id = self.database.save_factor_definition(
                                name=f"{category}_{factor_name}",
                                category=category,
                                description=self.factor_engine.get_factor_description(category, factor_name)
                            )
                            
                            # 팩터 값 저장
                            if isinstance(factor_data, pd.DataFrame):
                                self.database.save_factor_values(factor_id, factor_data)
            
            # 백테스팅 결과 저장
            if 'portfolio_results' in results and 'backtest_results' in results['portfolio_results']:
                backtest = results['portfolio_results']['backtest_results']
                if 'performance_metrics' in backtest:
                    self.database.save_backtest_results(
                        factor_id=1,  # 통합 팩터 ID
                        config=self.config.__dict__,
                        performance_metrics=backtest['performance_metrics'],
                        portfolio_returns=backtest.get('portfolio_returns', pd.Series())
                    )
            
            logger.info("결과 데이터베이스 저장 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 저장 오류: {str(e)}")
    
    def run_complete_analysis(self, tickers: List[str]) -> Dict[str, Any]:
        """완전한 분석 파이프라인 실행"""
        try:
            st.header("🚀 통합 알파 팩터 분석 시스템")
            st.write("모든 모듈을 통합한 완전한 포트폴리오 분석을 실행합니다.")
            
            results = {}
            
            # 1. 데이터 로딩
            with st.expander("📊 1단계: 시장 데이터 로딩", expanded=True):
                raw_data = self.load_market_data(tickers)
                if not raw_data:
                    st.error("❌ 데이터 로딩 실패")
                    return {}
                
                processed_data = self.process_market_data(raw_data)
                results['market_data'] = processed_data
                
                st.success(f"✅ {len(processed_data['prices'].columns)}개 종목 데이터 준비 완료")
            
            # 2. 팩터 분석
            with st.expander("🔢 2단계: 팩터 분석 (Z-Score 포함)", expanded=True):
                factor_results = self.calculate_all_factors(processed_data)
                results['factor_results'] = factor_results
                
                if factor_results:
                    st.success("✅ 팩터 분석 완료")
                    
                    # 팩터 요약 표시
                    self.display_factor_summary(factor_results)
                else:
                    st.error("❌ 팩터 분석 실패")
            
            # 3. 포트폴리오 구성
            with st.expander("📈 3단계: 통합 포트폴리오 구성", expanded=True):
                portfolio_results = self.create_unified_portfolio(processed_data, factor_results)
                results['portfolio_results'] = portfolio_results
                
                if portfolio_results:
                    st.success("✅ 포트폴리오 구성 완료")
                    
                    # 포트폴리오 요약 표시
                    self.display_portfolio_summary(portfolio_results)
                else:
                    st.error("❌ 포트폴리오 구성 실패")
            
            # 4. 알림 및 리스크 관리
            with st.expander("⚠️ 4단계: 알림 및 리스크 관리", expanded=True):
                alerts = self.generate_alerts(factor_results, portfolio_results)
                results['alerts'] = alerts
                
                self.display_alerts(alerts)
            
            # 5. 종합 대시보드
            with st.expander("📊 5단계: 종합 분석 대시보드", expanded=True):
                self.display_comprehensive_dashboard(results)
            
            # 6. 데이터베이스 저장
            if st.button("💾 결과 저장"):
                self.save_results_to_database(results)
                st.success("✅ 결과가 데이터베이스에 저장되었습니다.")
            
            return results
            
        except Exception as e:
            logger.error(f"완전 분석 실행 오류: {str(e)}")
            st.error(f"분석 실행 오류: {str(e)}")
            return {}

    # UI 표시 메서드들은 다음 파일에서 계속됩니다...
    
    def display_factor_summary(self, factor_results: Dict[str, Any]):
        """팩터 분석 결과 요약 표시"""
        if 'zscore_factors' in factor_results and factor_results['zscore_factors']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Z-Score 팩터 현황**")
                summary_data = []
                
                for factor_type, scores in factor_results['zscore_factors'].items():
                    if isinstance(scores, pd.Series) and not scores.empty:
                        latest_score = scores.iloc[-1] if len(scores) > 0 else 0
                        summary_data.append({
                            'Factor': factor_type.title(),
                            'Latest Z-Score': f"{latest_score:.3f}",
                            'Signal': '🔴 매도' if latest_score < -1 else '🟡 중립' if abs(latest_score) < 1 else '🟢 매수'
                        })
                
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data))
            
            with col2:
                if 'factor_correlation' in factor_results and not factor_results['factor_correlation'].empty:
                    st.write("**팩터 상관관계**")
                    fig = px.imshow(
                        factor_results['factor_correlation'],
                        title="팩터 간 상관관계",
                        color_continuous_scale='RdBu_r',
                        aspect='auto'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    def display_portfolio_summary(self, portfolio_results: Dict[str, Any]):
        """포트폴리오 결과 요약 표시"""
        if 'optimized_portfolio' in portfolio_results:
            portfolio = portfolio_results['optimized_portfolio']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "예상 연수익률",
                    f"{portfolio.get('expected_return', 0) * 100:.2f}%"
                )
            
            with col2:
                st.metric(
                    "예상 변동성", 
                    f"{portfolio.get('expected_volatility', 0) * 100:.2f}%"
                )
            
            with col3:
                st.metric(
                    "샤프 비율",
                    f"{portfolio.get('sharpe_ratio', 0):.3f}"
                )
            
            # 포트폴리오 구성 표시
            if 'weights' in portfolio:
                weights = portfolio['weights']
                if not weights.empty:
                    st.write("**포트폴리오 구성**")
                    
                    # 상위 10개 종목만 표시
                    top_weights = weights.sort_values(ascending=False).head(10)
                    
                    fig = px.bar(
                        x=top_weights.values * 100,
                        y=top_weights.index,
                        orientation='h',
                        title="포트폴리오 비중 (상위 10개)",
                        labels={'x': '비중 (%)', 'y': '종목'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    def display_alerts(self, alerts: List[Dict[str, Any]]):
        """알림 표시"""
        if not alerts:
            st.success("✅ 현재 특별한 알림이 없습니다.")
            return
        
        for alert in alerts:
            severity = alert.get('severity', 'LOW')
            
            if severity == 'HIGH':
                st.error(f"🚨 {alert['message']}")
            elif severity == 'MEDIUM':
                st.warning(f"⚠️ {alert['message']}")
            else:
                st.info(f"ℹ️ {alert['message']}")
    
    def display_comprehensive_dashboard(self, results: Dict[str, Any]):
        """종합 분석 대시보드"""
        if 'portfolio_results' in results and 'backtest_results' in results['portfolio_results']:
            backtest = results['portfolio_results']['backtest_results']
            
            if 'portfolio_returns' in backtest and not backtest['portfolio_returns'].empty:
                returns = backtest['portfolio_returns']
                
                # 수익률 차트
                cumulative_returns = (1 + returns).cumprod()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    mode='lines',
                    name='누적 수익률',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title="포트폴리오 누적 수익률",
                    xaxis_title="날짜",
                    yaxis_title="누적 수익률",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 성과 지표 표시
                if 'performance_metrics' in backtest:
                    metrics = backtest['performance_metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("총 수익률", f"{metrics.get('Total Return', 0) * 100:.2f}%")
                    with col2:
                        st.metric("연수익률", f"{metrics.get('Annualized Return', 0) * 100:.2f}%")
                    with col3:
                        st.metric("샤프 비율", f"{metrics.get('Sharpe Ratio', 0):.3f}")
                    with col4:
                        st.metric("최대 손실", f"{metrics.get('Max Drawdown', 0) * 100:.2f}%")

# 메인 클래스 완료
if __name__ == "__main__":
    st.set_page_config(
        page_title="통합 알파 팩터 시스템",
        page_icon="🚀", 
        layout="wide"
    )
    
    # 시스템 초기화
    config = UnifiedSystemConfig()
    system = UnifiedAlphaSystem(config)
    
    # 테스트용 티커
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # 완전한 분석 실행
    results = system.run_complete_analysis(test_tickers)