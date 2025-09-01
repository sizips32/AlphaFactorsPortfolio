"""
Alpha Factor Z-Score Calculator Module
S&P 방법론 기반 팩터 분석 도구

작성자: AI Assistant
작성일: 2025년 8월 25일
버전: 2.0 (모듈화 버전)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FactorZScoreCalculator:
    """팩터 Z-Score 계산기"""
    
    def __init__(self):
        self.factor_weights = {
            'value': {
                'pe_ratio': 0.4,
                'pb_ratio': 0.3,
                'price_sales': 0.3
            },
            'quality': {
                'roe': 0.3,
                'debt_equity': 0.3,
                'current_ratio': 0.2,
                'profit_margin': 0.2
            },
            'momentum': {
                'price_momentum_1m': 0.25,
                'price_momentum_3m': 0.25,
                'price_momentum_6m': 0.25,
                'price_momentum_12m': 0.25
            }
        }
    
    def calculate_factor_zscore(self, factor_data: pd.DataFrame, factor_type: str = 'value') -> pd.Series:
        """
        팩터 Z-Score 계산
        
        Args:
            factor_data: 팩터 데이터 (날짜 x 팩터값)
            factor_type: 팩터 타입 ('value', 'quality', 'momentum')
        
        Returns:
            팩터 Z-Score 시계열
        """
        try:
            if factor_data.empty:
                return pd.Series()
            
            # 각 팩터를 Z-Score로 변환
            z_scores = pd.DataFrame(index=factor_data.index)
            
            for column in factor_data.columns:
                factor_values = factor_data[column].dropna()
                if len(factor_values) > 1:
                    # Z-Score 계산: (값 - 평균) / 표준편차
                    mean_val = factor_values.mean()
                    std_val = factor_values.std()
                    
                    if std_val != 0:
                        z_scores[column] = (factor_data[column] - mean_val) / std_val
                    else:
                        z_scores[column] = 0
                else:
                    z_scores[column] = 0
            
            # 복합 팩터 점수 계산 (평균)
            composite_score = z_scores.mean(axis=1)
            
            return composite_score
            
        except Exception as e:
            print(f"Z-Score 계산 오류: {str(e)}")
            return pd.Series()
    
    def calculate_percentile_rank(self, scores: pd.Series) -> pd.Series:
        """백분위 순위 계산"""
        try:
            if scores.empty:
                return pd.Series()
            
            # 백분위 순위 계산 (0-100)
            percentile_ranks = scores.rank(pct=True) * 100
            
            return percentile_ranks
            
        except Exception as e:
            print(f"백분위 순위 계산 오류: {str(e)}")
            return pd.Series()
    
    def calculate_value_factor(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None) -> pd.DataFrame:
        """밸류 팩터 계산 (가격 기반 근사치)"""
        try:
            value_factors = pd.DataFrame(index=price_data.index)
            
            # PE 근사치: 1 / (연평균 수익률)
            returns = price_data.pct_change()
            annual_returns = returns.rolling(252).mean() * 252
            pe_proxy = 1 / annual_returns.replace(0, np.nan)
            value_factors['PE_Proxy'] = pe_proxy.mean(axis=1)
            
            # PB 근사치: 현재가 / 연평균가
            pb_proxy = price_data / price_data.rolling(252).mean()
            value_factors['PB_Proxy'] = pb_proxy.mean(axis=1)
            
            # Price/Sales 근사치: 가격 / (거래량 * 가격의 이동평균)
            if volume_data is not None:
                sales_proxy = (volume_data * price_data).rolling(252).mean()
                ps_proxy = price_data / sales_proxy.replace(0, np.nan)
                value_factors['PS_Proxy'] = ps_proxy.mean(axis=1)
            else:
                # 거래량 데이터가 없으면 가격 변동성으로 대체
                volatility = returns.rolling(252).std()
                value_factors['PS_Proxy'] = 1 / volatility.mean(axis=1)
            
            return value_factors.dropna()
            
        except Exception as e:
            print(f"밸류 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def calculate_quality_factor(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None) -> pd.DataFrame:
        """퀄리티 팩터 계산 (수익률 품질 기반)"""
        try:
            quality_factors = pd.DataFrame(index=price_data.index)
            
            returns = price_data.pct_change()
            
            # ROE 근사치: 안정적 수익률 (변동성 대비 수익률)
            annual_returns = returns.rolling(252).mean() * 252
            annual_volatility = returns.rolling(252).std() * np.sqrt(252)
            roe_proxy = annual_returns / annual_volatility.replace(0, np.nan)
            quality_factors['ROE_Proxy'] = roe_proxy.mean(axis=1)
            
            # Debt/Equity 근사치: 변동성 (낮을수록 좋음)
            debt_equity_proxy = annual_volatility
            quality_factors['DebtEquity_Proxy'] = -debt_equity_proxy.mean(axis=1)  # 음수로 변환
            
            # Current Ratio 근사치: 유동성 (거래량 기반)
            if volume_data is not None:
                volume_stability = 1 / volume_data.rolling(60).std().replace(0, np.nan)
                quality_factors['CurrentRatio_Proxy'] = volume_stability.mean(axis=1)
            else:
                # 가격 안정성으로 대체
                price_stability = 1 / price_data.rolling(60).std().replace(0, np.nan)
                quality_factors['CurrentRatio_Proxy'] = price_stability.mean(axis=1)
            
            # Profit Margin 근사치: 샤프 비율
            sharpe_ratio = annual_returns / annual_volatility.replace(0, np.nan)
            quality_factors['ProfitMargin_Proxy'] = sharpe_ratio.mean(axis=1)
            
            return quality_factors.dropna()
            
        except Exception as e:
            print(f"퀄리티 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def calculate_momentum_factor(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """모멘텀 팩터 계산"""
        try:
            momentum_factors = pd.DataFrame(index=price_data.index)
            
            returns = price_data.pct_change()
            
            # 1개월 모멘텀
            momentum_1m = returns.rolling(20).sum()
            momentum_factors['Momentum_1M'] = momentum_1m.mean(axis=1)
            
            # 3개월 모멘텀  
            momentum_3m = returns.rolling(60).sum()
            momentum_factors['Momentum_3M'] = momentum_3m.mean(axis=1)
            
            # 6개월 모멘텀
            momentum_6m = returns.rolling(120).sum()
            momentum_factors['Momentum_6M'] = momentum_6m.mean(axis=1)
            
            # 12개월 모멘텀
            momentum_12m = returns.rolling(252).sum()
            momentum_factors['Momentum_12M'] = momentum_12m.mean(axis=1)
            
            return momentum_factors.dropna()
            
        except Exception as e:
            print(f"모멘텀 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_description(self, factor_type: str) -> str:
        """팩터 설명 반환"""
        descriptions = {
            'value': '밸류 팩터: 저평가된 주식을 찾는 지표 (PE, PB, PS 비율 기반)',
            'quality': '퀄리티 팩터: 재무 건전성이 높은 주식을 찾는 지표 (ROE, 부채비율, 유동비율 기반)', 
            'momentum': '모멘텀 팩터: 가격 추세가 지속되는 주식을 찾는 지표 (1M, 3M, 6M, 12M 수익률 기반)'
        }
        return descriptions.get(factor_type, '알 수 없는 팩터 유형')
    
    def calculate_factor_correlation(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """팩터 간 상관관계 계산"""
        try:
            return factor_data.corr()
        except Exception as e:
            print(f"상관관계 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_statistics(self, factor_scores: pd.Series) -> Dict:
        """팩터 통계 정보"""
        try:
            if factor_scores.empty:
                return {}
            
            stats_dict = {
                'mean': factor_scores.mean(),
                'std': factor_scores.std(),
                'min': factor_scores.min(),
                'max': factor_scores.max(),
                'q25': factor_scores.quantile(0.25),
                'q50': factor_scores.quantile(0.50),
                'q75': factor_scores.quantile(0.75),
                'skewness': stats.skew(factor_scores.dropna()),
                'kurtosis': stats.kurtosis(factor_scores.dropna())
            }
            
            return stats_dict
            
        except Exception as e:
            print(f"통계 계산 오류: {str(e)}")
            return {}

# 독립 실행용 코드 (모듈로 import할 때는 실행되지 않음)
if __name__ == "__main__":
    # 테스트 코드
    calculator = FactorZScoreCalculator()
    
    # 샘플 데이터로 테스트
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    sample_data = pd.DataFrame({
        'factor1': np.random.randn(len(dates)),
        'factor2': np.random.randn(len(dates)) + 0.5,
        'factor3': np.random.randn(len(dates)) * 2
    }, index=dates)
    
    # Z-Score 계산 테스트
    z_scores = calculator.calculate_factor_zscore(sample_data, 'value')
    print("Z-Score 계산 테스트 완료")
    print(f"평균: {z_scores.mean():.3f}, 표준편차: {z_scores.std():.3f}")
    
    # 백분위 순위 계산 테스트
    percentiles = calculator.calculate_percentile_rank(z_scores)
    print(f"백분위 범위: {percentiles.min():.1f} ~ {percentiles.max():.1f}")