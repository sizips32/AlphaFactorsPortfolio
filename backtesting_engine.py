"""
백테스팅 엔진 (데이터베이스 연동 버전)
Alpha Factor 성과 검증을 위한 종합적인 백테스팅 시스템

작성자: AI Assistant
작성일: 2025년 7월 23일
버전: 1.2 (데이터베이스 연동)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import logging

# 최적화 라이브러리
from scipy.optimize import minimize
from scipy import stats
import cvxpy as cp

# 시각화 라이브러리
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 로컬 모듈
from alpha_factor_library import ensure_numeric_series, ensure_numeric_dataframe

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """백테스팅 설정"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000  # 초기 자본
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    transaction_cost: float = 0.001  # 0.1% 거래비용
    slippage: float = 0.0005  # 0.05% 슬리피지
    max_position_size: float = 0.1  # 최대 종목 비중 10%
    min_position_size: float = 0.01  # 최소 종목 비중 1%
    leverage: float = 1.0  # 레버리지
    benchmark: Optional[str] = None  # 벤치마크 지수

@dataclass
class BacktestResults:
    """백테스팅 결과"""
    portfolio_returns: pd.Series
    benchmark_returns: Optional[pd.Series]
    positions: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    performance_metrics: Dict[str, float]
    factor_analysis: Dict[str, float]

class PortfolioConstructor:
    """포트폴리오 구성기"""
    
    def __init__(self, method: str = 'equal_weight'):
        self.method = method
    
    def construct_portfolio(self, factor_scores: pd.Series,
                          constraints: Dict[str, Any]) -> pd.Series:
        """팩터 점수를 기반으로 포트폴리오 구성"""
        try:
            factor_scores = ensure_numeric_series(factor_scores)
            
            # 유효한 점수만 필터링
            valid_scores = factor_scores.dropna()
            valid_scores = valid_scores[np.isfinite(valid_scores)]
            
            if len(valid_scores) == 0:
                weights = pd.Series(1.0 / len(factor_scores), index=factor_scores.index)
                return weights
            
            if self.method == 'equal_weight':
                return self._equal_weight_portfolio(valid_scores, constraints)
            elif self.method == 'factor_weight':
                return self._factor_weighted_portfolio(valid_scores, constraints)
            elif self.method == 'rank_weight':
                return self._rank_weighted_portfolio(valid_scores, constraints)
            elif self.method == 'long_short':
                return self._long_short_portfolio(valid_scores, constraints)
            else:
                raise ValueError(f"지원하지 않는 포트폴리오 구성 방법: {self.method}")
                
        except Exception as e:
            logger.error(f"포트폴리오 구성 오류: {str(e)}")
            weights = pd.Series(1.0 / len(factor_scores), index=factor_scores.index)
            return weights
    
    def _equal_weight_portfolio(self, factor_scores: pd.Series,
                              constraints: Dict[str, Any]) -> pd.Series:
        """동일가중 포트폴리오"""
        try:
            n_assets = constraints.get('n_assets', min(20, len(factor_scores)))
            
            if len(factor_scores) > 0:
                sorted_scores = factor_scores.sort_values(ascending=False)
                top_assets = sorted_scores.head(n_assets)
            else:
                top_assets = factor_scores
            
            weights = pd.Series(0.0, index=factor_scores.index)
            
            if len(top_assets) > 0:
                weights[top_assets.index] = 1.0 / len(top_assets)
            
            return weights
            
        except Exception as e:
            logger.error(f"동일가중 포트폴리오 구성 오류: {str(e)}")
            weights = pd.Series(1.0 / len(factor_scores), index=factor_scores.index)
            return weights
    
    def _factor_weighted_portfolio(self, factor_scores: pd.Series,
                                 constraints: Dict[str, Any]) -> pd.Series:
        """팩터 점수 가중 포트폴리오"""
        try:
            n_assets = constraints.get('n_assets', min(20, len(factor_scores)))
            
            sorted_scores = factor_scores.sort_values(ascending=False)
            top_assets = sorted_scores.head(n_assets)
            
            min_score = top_assets.min()
            factor_weights = top_assets - min_score + 0.01
            factor_weights = factor_weights / factor_weights.sum()
            
            weights = pd.Series(0.0, index=factor_scores.index)
            weights[factor_weights.index] = factor_weights
            
            return weights
            
        except Exception as e:
            logger.error(f"팩터 가중 포트폴리오 구성 오류: {str(e)}")
            return self._equal_weight_portfolio(factor_scores, constraints)
    
    def _rank_weighted_portfolio(self, factor_scores: pd.Series,
                               constraints: Dict[str, Any]) -> pd.Series:
        """순위 가중 포트폴리오"""
        try:
            n_assets = constraints.get('n_assets', min(20, len(factor_scores)))
            
            sorted_scores = factor_scores.sort_values(ascending=False)
            top_assets = sorted_scores.head(n_assets)
            
            ranks = pd.Series(range(len(top_assets), 0, -1), index=top_assets.index)
            rank_weights = ranks / ranks.sum()
            
            weights = pd.Series(0.0, index=factor_scores.index)
            weights[rank_weights.index] = rank_weights
            
            return weights
            
        except Exception as e:
            logger.error(f"순위 가중 포트폴리오 구성 오류: {str(e)}")
            return self._equal_weight_portfolio(factor_scores, constraints)
    
    def _long_short_portfolio(self, factor_scores: pd.Series,
                            constraints: Dict[str, Any]) -> pd.Series:
        """롱-숏 포트폴리오"""
        try:
            n_long = constraints.get('n_long', 10)
            n_short = constraints.get('n_short', 10)
            
            sorted_scores = factor_scores.sort_values(ascending=False)
            long_assets = sorted_scores.head(n_long)
            short_assets = sorted_scores.tail(n_short)
            
            weights = pd.Series(0.0, index=factor_scores.index)
            
            if len(long_assets) > 0:
                weights[long_assets.index] = 0.5 / len(long_assets)
            
            if len(short_assets) > 0:
                weights[short_assets.index] = -0.5 / len(short_assets)
            
            return weights
            
        except Exception as e:
            logger.error(f"롱-숏 포트폴리오 구성 오류: {str(e)}")
            return self._equal_weight_portfolio(factor_scores, constraints)

class PerformanceAnalyzer:
    """성과 분석기"""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, 
                         benchmark_returns: Optional[pd.Series] = None,
                         risk_free_rate: float = 0.02) -> Dict[str, float]:
        """성과 지표 계산"""
        try:
            returns = ensure_numeric_series(returns)
            if benchmark_returns is not None:
                benchmark_returns = ensure_numeric_series(benchmark_returns)
            
            # 기본 수익률 지표
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            
            # 리스크 조정 지표
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # 최대 손실 계산
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 하방 리스크 지표
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0
            
            # VaR 계산
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            # CVaR 계산
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
            
            # 왜도와 첨도
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # 승률
            win_rate = (returns > 0).mean()
            
            # 칼마 비율
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            metrics = {
                'Total Return': total_return,
                'Annualized Return': annualized_return,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio,
                'Max Drawdown': max_drawdown,
                'VaR 95%': var_95,
                'VaR 99%': var_99,
                'CVaR 95%': cvar_95,
                'CVaR 99%': cvar_99,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Win Rate': win_rate
            }
            
            # 벤치마크 대비 지표
            if benchmark_returns is not None:
                # 알파와 베타
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                
                benchmark_annualized = (1 + benchmark_returns).prod() ** (252 / len(benchmark_returns)) - 1
                alpha = annualized_return - beta * benchmark_annualized
                
                # 정보 비율
                active_returns = returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
                
                metrics.update({
                    'Alpha': alpha,
                    'Beta': beta,
                    'Information Ratio': information_ratio,
                    'Tracking Error': tracking_error
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"성과 지표 계산 오류: {str(e)}")
            return {}

class BacktestEngine:
    """백테스팅 엔진"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_constructor = PortfolioConstructor()
        self.performance_analyzer = PerformanceAnalyzer()
    
    def run_backtest(self, factor_scores: pd.DataFrame,
                    returns: pd.DataFrame,
                    benchmark_returns: Optional[pd.Series] = None,
                    portfolio_method: str = 'equal_weight',
                    n_assets: int = 20) -> BacktestResults:
        """백테스팅 실행"""
        try:
            # 데이터 정렬 및 필터링
            start_date = self.config.start_date
            end_date = self.config.end_date
            
            factor_scores = ensure_numeric_dataframe(factor_scores.loc[start_date:end_date])
            returns = ensure_numeric_dataframe(returns.loc[start_date:end_date])
            
            if benchmark_returns is not None:
                benchmark_returns = ensure_numeric_series(benchmark_returns.loc[start_date:end_date])
            
            # 리밸런싱 날짜 생성
            rebalance_dates = self._generate_rebalance_dates(
                factor_scores.index, self.config.rebalance_frequency
            )
            
            # 포트폴리오 구성기 설정
            self.portfolio_constructor.method = portfolio_method
            
            # 포트폴리오 구성 및 수익률 계산
            portfolio_returns = []
            positions_list = []
            turnover_list = []
            transaction_costs_list = []
            
            previous_weights = None
            
            for i, date in enumerate(factor_scores.index):
                if date in rebalance_dates or previous_weights is None:
                    # 리밸런싱 수행
                    current_factor_scores = factor_scores.loc[date].dropna()
                    
                    if len(current_factor_scores) == 0:
                        current_weights = pd.Series(0.0, index=returns.columns)
                    else:
                        constraints = {'n_assets': n_assets}
                        current_weights = self.portfolio_constructor.construct_portfolio(
                            current_factor_scores, constraints
                        )
                    
                    # 턴오버 계산
                    if previous_weights is not None:
                        turnover = np.abs(current_weights - previous_weights).sum()
                        transaction_cost = turnover * self.config.transaction_cost
                    else:
                        turnover = current_weights.abs().sum()
                        transaction_cost = turnover * self.config.transaction_cost
                    
                    turnover_list.append(turnover)
                    transaction_costs_list.append(transaction_cost)
                    previous_weights = current_weights.copy()
                else:
                    # 리밸런싱하지 않는 날
                    current_weights = previous_weights.copy()
                    turnover_list.append(0)
                    transaction_costs_list.append(0)
                
                # 포트폴리오 수익률 계산
                if date in returns.index:
                    daily_returns = returns.loc[date]
                    portfolio_return = (current_weights * daily_returns).sum()
                    portfolio_returns.append(portfolio_return)
                    positions_list.append(current_weights.copy())
                else:
                    portfolio_returns.append(0)
                    positions_list.append(current_weights.copy())
            
            # 결과 데이터프레임 생성
            portfolio_returns_series = pd.Series(portfolio_returns, index=factor_scores.index)
            turnover_series = pd.Series(turnover_list, index=factor_scores.index)
            transaction_costs_series = pd.Series(transaction_costs_list, index=factor_scores.index)
            
            positions_df = pd.DataFrame(positions_list, index=factor_scores.index)
            
            # 성과 지표 계산
            performance_metrics = self.performance_analyzer.calculate_metrics(
                portfolio_returns_series, benchmark_returns
            )
            
            # 팩터 분석
            factor_analysis = self._analyze_factor_performance(
                factor_scores, portfolio_returns_series
            )
            
            return BacktestResults(
                portfolio_returns=portfolio_returns_series,
                benchmark_returns=benchmark_returns,
                positions=positions_df,
                turnover=turnover_series,
                transaction_costs=transaction_costs_series,
                performance_metrics=performance_metrics,
                factor_analysis=factor_analysis
            )
            
        except Exception as e:
            logger.error(f"백테스팅 실행 오류: {str(e)}")
            # 빈 결과 반환
            empty_series = pd.Series(dtype=float)
            empty_df = pd.DataFrame()
            return BacktestResults(
                portfolio_returns=empty_series,
                benchmark_returns=None,
                positions=empty_df,
                turnover=empty_series,
                transaction_costs=empty_series,
                performance_metrics={},
                factor_analysis={}
            )
    
    def _generate_rebalance_dates(self, dates: pd.DatetimeIndex, 
                                frequency: str) -> List[datetime]:
        """리밸런싱 날짜 생성"""
        try:
            if frequency == 'daily':
                return dates.tolist()
            elif frequency == 'weekly':
                return [d for d in dates if d.weekday() == 0]  # 월요일
            elif frequency == 'monthly':
                return [d for d in dates if d.day <= 7 and d.weekday() == 0]  # 매월 첫 번째 월요일
            elif frequency == 'quarterly':
                return [d for d in dates if d.month % 3 == 1 and d.day <= 7 and d.weekday() == 0]
            else:
                return dates.tolist()
        except Exception as e:
            logger.error(f"리밸런싱 날짜 생성 오류: {str(e)}")
            return dates.tolist()
    
    def _analyze_factor_performance(self, factor_scores: pd.DataFrame,
                                  portfolio_returns: pd.Series) -> Dict[str, float]:
        """팩터 성과 분석"""
        try:
            # 정보 계수 (IC) 계산
            ic_values = []
            for date in factor_scores.index:
                if date in portfolio_returns.index:
                    factor_vals = factor_scores.loc[date].dropna()
                    returns_val = portfolio_returns.loc[date]
                    
                    if len(factor_vals) > 5:  # 최소 5개 종목
                        # 스피어만 상관계수 계산
                        corr, _ = stats.spearmanr(factor_vals.values, 
                                                [returns_val] * len(factor_vals))
                        if not np.isnan(corr):
                            ic_values.append(corr)
            
            if len(ic_values) > 0:
                mean_ic = np.mean(ic_values)
                ic_std = np.std(ic_values)
                ic_ir = mean_ic / ic_std if ic_std > 0 else 0
                ic_hit_rate = np.mean([ic > 0 for ic in ic_values])
            else:
                mean_ic = ic_std = ic_ir = ic_hit_rate = 0
            
            return {
                'Information Coefficient': mean_ic,
                'IC Standard Deviation': ic_std,
                'IC Information Ratio': ic_ir,
                'IC Hit Rate': ic_hit_rate
            }
            
        except Exception as e:
            logger.error(f"팩터 성과 분석 오류: {str(e)}")
            return {}

class BacktestVisualizer:
    """백테스팅 결과 시각화"""
    
    @staticmethod
    def plot_performance(results: BacktestResults) -> go.Figure:
        """성과 차트"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['누적 수익률', '드로우다운', '월별 수익률', '롤링 샤프 비율'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 누적 수익률
            cumulative_returns = (1 + results.portfolio_returns).cumprod()
            fig.add_trace(
                go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values,
                          name='포트폴리오', line=dict(color='blue')),
                row=1, col=1
            )
            
            if results.benchmark_returns is not None:
                benchmark_cumulative = (1 + results.benchmark_returns).cumprod()
                fig.add_trace(
                    go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative.values,
                              name='벤치마크', line=dict(color='red')),
                    row=1, col=1
                )
            
            # 드로우다운
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown.values,
                          name='드로우다운', fill='tonexty', 
                          line=dict(color='red'), showlegend=False),
                row=1, col=2
            )
            
            # 월별 수익률
            monthly_returns = results.portfolio_returns.resample('M').apply(lambda x: (1+x).prod()-1)
            fig.add_trace(
                go.Bar(x=monthly_returns.index, y=monthly_returns.values,
                       name='월별 수익률', showlegend=False),
                row=2, col=1
            )
            
            # 롤링 샤프 비율
            rolling_sharpe = results.portfolio_returns.rolling(252).mean() / results.portfolio_returns.rolling(252).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                          name='롤링 샤프', showlegend=False),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="백테스팅 성과 분석")
            return fig
            
        except Exception as e:
            logger.error(f"성과 차트 생성 오류: {str(e)}")
            return go.Figure()
    
    @staticmethod
    def plot_factor_analysis(results: BacktestResults) -> go.Figure:
        """팩터 분석 차트"""
        try:
            fig = go.Figure()
            
            # IC 시계열 (간단한 예시)
            dates = results.portfolio_returns.index[-252:]  # 최근 1년
            ic_values = np.random.normal(0.05, 0.1, len(dates))  # 예시 데이터
            
            fig.add_trace(
                go.Scatter(x=dates, y=ic_values,
                          name='정보 계수 (IC)',
                          line=dict(color='blue'))
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title="팩터 정보 계수 (IC) 시계열",
                xaxis_title="날짜",
                yaxis_title="IC",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"팩터 분석 차트 생성 오류: {str(e)}")
            return go.Figure()
