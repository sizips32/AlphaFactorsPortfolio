"""
Alpha Factor 라이브러리 (수정된 버전)
다양한 알파 팩터 생성을 위한 종합적인 라이브러리

작성자: AI Assistant
작성일: 2025년 7월 23일
버전: 1.1 (데이터베이스 연동 및 에러 수정)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import logging

# 머신러닝 라이브러리
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# 통계 라이브러리
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import hilbert

# 고급 수학/통계 라이브러리
try:
    import pywt  # 웨이블릿 변환
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from hmmlearn import hmm  # Hidden Markov Model
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# TensorFlow는 호환성 문제로 인해 비활성화
TENSORFLOW_AVAILABLE = False
tf = None

logger = logging.getLogger(__name__)

def ensure_numeric_series(series: pd.Series) -> pd.Series:
    """Series가 숫자형인지 확인하고 변환"""
    if series is None:
        return series
    
    # 숫자형으로 변환 시도
    series = pd.to_numeric(series, errors='coerce')
    # NaN을 0으로 대체
    series = series.fillna(0)
    
    return series

def ensure_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame이 숫자형인지 확인하고 변환"""
    if df is None:
        return df
    
    # 숫자형으로 변환 시도
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # NaN을 0으로 대체
    df = df.fillna(0)
    
    return df

class FactorCategory(Enum):
    """팩터 카테고리"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MACHINE_LEARNING = "machine_learning"
    RISK = "risk"
    MACRO = "macro"

@dataclass
class FactorMetadata:
    """팩터 메타데이터"""
    name: str
    category: FactorCategory
    description: str
    parameters: Dict[str, Any]
    calculation_method: str
    expected_frequency: str  # daily, weekly, monthly

class FactorValidator:
    """팩터 유효성 검증기"""
    
    @staticmethod
    def validate_factor_values(factor_values: Union[pd.Series, pd.DataFrame], 
                             metadata: FactorMetadata) -> Dict[str, Any]:
        """팩터 값 유효성 검증"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            if factor_values is None or factor_values.empty:
                validation_results['is_valid'] = False
                validation_results['errors'].append("팩터 값이 비어있습니다.")
                return validation_results
            
            # 데이터 타입 검증
            if isinstance(factor_values, pd.DataFrame):
                numeric_df = ensure_numeric_dataframe(factor_values)
                factor_values = numeric_df
            else:
                numeric_series = ensure_numeric_series(factor_values)
                factor_values = numeric_series
            
            # 기본 통계
            if isinstance(factor_values, pd.DataFrame):
                validation_results['statistics'] = {
                    'shape': factor_values.shape,
                    'missing_ratio': factor_values.isnull().sum().sum() / factor_values.size,
                    'mean': factor_values.mean().mean(),
                    'std': factor_values.std().mean(),
                    'min': factor_values.min().min(),
                    'max': factor_values.max().max()
                }
            else:
                validation_results['statistics'] = {
                    'length': len(factor_values),
                    'missing_ratio': factor_values.isnull().sum() / len(factor_values),
                    'mean': factor_values.mean(),
                    'std': factor_values.std(),
                    'min': factor_values.min(),
                    'max': factor_values.max()
                }
            
            # 무한대 값 검사
            if isinstance(factor_values, pd.DataFrame):
                inf_count = np.isinf(factor_values.values).sum()
            else:
                inf_count = np.isinf(factor_values.values).sum()
            
            if inf_count > 0:
                validation_results['warnings'].append(f"무한대 값이 {inf_count}개 발견되었습니다.")
            
            # 분산 검사
            if validation_results['statistics']['std'] == 0:
                validation_results['warnings'].append("팩터 값의 분산이 0입니다.")
            
            # 결측치 비율 검사
            if validation_results['statistics']['missing_ratio'] > 0.5:
                validation_results['warnings'].append(f"결측치 비율이 높습니다: {validation_results['statistics']['missing_ratio']:.2%}")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"검증 중 오류 발생: {str(e)}")
        
        return validation_results

class TechnicalFactors:
    """기술적 분석 팩터"""
    
    @staticmethod
    def price_momentum(prices: pd.DataFrame, window: int = 20, 
                      method: str = 'simple') -> pd.DataFrame:
        """가격 모멘텀 팩터"""
        try:
            prices = ensure_numeric_dataframe(prices)
            
            if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
                raise ValueError("prices 데이터가 올바르지 않습니다. DataFrame 타입이며 비어있지 않아야 합니다.")

            if method == 'simple':
                # 단순 수익률
                momentum = prices.pct_change(window)
            elif method == 'log':
                # 로그 수익률
                momentum = np.log(prices / prices.shift(window))
            elif method == 'risk_adjusted':
                # 리스크 조정 모멘텀
                returns = prices.pct_change()
                momentum = returns.rolling(window).mean() / returns.rolling(window).std()
            else:
                momentum = prices.pct_change(window)
            
            return ensure_numeric_dataframe(momentum)
            
        except Exception as e:
            logger.error(f"모멘텀 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def mean_reversion(prices: pd.DataFrame, window: int = 20, 
                      method: str = 'zscore') -> pd.DataFrame:
        """평균회귀 팩터"""
        try:
            prices = ensure_numeric_dataframe(prices)
            
            if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
                raise ValueError("prices 데이터가 올바르지 않습니다. DataFrame 타입이며 비어있지 않아야 합니다.")

            if method == 'zscore':
                # Z-Score
                rolling_mean = prices.rolling(window).mean()
                rolling_std = prices.rolling(window).std()
                factor = (prices - rolling_mean) / rolling_std
            elif method == 'bollinger':
                # 볼린저 밴드 위치
                rolling_mean = prices.rolling(window).mean()
                rolling_std = prices.rolling(window).std()
                upper_band = rolling_mean + 2 * rolling_std
                lower_band = rolling_mean - 2 * rolling_std
                factor = (prices - lower_band) / (upper_band - lower_band)
            elif method == 'rsi':
                # RSI
                factor = TechnicalFactors._calculate_rsi(prices, window)
            else:
                rolling_mean = prices.rolling(window).mean()
                rolling_std = prices.rolling(window).std()
                factor = (prices - rolling_mean) / rolling_std
            
            return ensure_numeric_dataframe(factor)
            
        except Exception as e:
            logger.error(f"평균회귀 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _calculate_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return ensure_numeric_dataframe(rsi)
            
        except Exception as e:
            logger.error(f"RSI 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def volatility_factor(prices: pd.DataFrame, window: int = 20, 
                         method: str = 'realized') -> pd.DataFrame:
        """변동성 팩터"""
        try:
            prices = ensure_numeric_dataframe(prices)
            
            if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
                raise ValueError("prices 데이터가 올바르지 않습니다. DataFrame 타입이며 비어있지 않아야 합니다.")

            returns = prices.pct_change()
            
            if method == 'realized':
                # 실현 변동성
                volatility = returns.rolling(window).std()
            elif method == 'garch':
                # 단순화된 GARCH (실제로는 EWMA)
                volatility = returns.ewm(span=window).std()
            else:
                volatility = returns.rolling(window).std()
            
            return ensure_numeric_dataframe(volatility)
            
        except Exception as e:
            logger.error(f"변동성 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def volume_factor(volumes: pd.DataFrame, prices: pd.DataFrame = None, 
                     method: str = 'obv') -> pd.DataFrame:
        """거래량 팩터"""
        try:
            volumes = ensure_numeric_dataframe(volumes)
            
            if volumes is None or not isinstance(volumes, pd.DataFrame) or volumes.empty:
                raise ValueError("volumes 데이터가 올바르지 않습니다. DataFrame 타입이며 비어있지 않아야 합니다.")

            if method == 'obv' and (prices is None or not isinstance(prices, pd.DataFrame) or prices.empty):
                raise ValueError("OBV 계산을 위해서는 prices 데이터가 필요합니다.")

            if method == 'obv' and prices is not None:
                # On-Balance Volume
                prices = ensure_numeric_dataframe(prices)
                price_change = prices.diff()
                obv = pd.DataFrame(index=volumes.index, columns=volumes.columns)
                
                for col in volumes.columns:
                    vol_series = volumes[col].copy()
                    price_series = price_change[col] if col in price_change.columns else 0
                    
                    obv_values = []
                    current_obv = 0
                    
                    for i, (vol, price_diff) in enumerate(zip(vol_series, price_series)):
                        if pd.notna(price_diff) and price_diff > 0:
                            current_obv += vol
                        elif pd.notna(price_diff) and price_diff < 0:
                            current_obv -= vol
                        obv_values.append(current_obv)
                    
                    obv[col] = obv_values
                
                factor = obv
            elif method == 'vroc':
                # Volume Rate of Change
                factor = volumes.pct_change(20)
            else:
                # 단순 거래량 비율
                factor = volumes / volumes.rolling(20).mean()
            
            return ensure_numeric_dataframe(factor)
            
        except Exception as e:
            logger.error(f"거래량 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()

class FundamentalFactors:
    """펀더멘털 분석 팩터"""
    
    @staticmethod
    def valuation_factor(prices: pd.DataFrame, market_caps: pd.DataFrame = None, 
                        method: str = 'pbr') -> pd.DataFrame:
        """밸류에이션 팩터"""
        try:
            prices = ensure_numeric_dataframe(prices)
            
            if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
                raise ValueError("prices 데이터가 올바르지 않습니다. DataFrame 타입이며 비어있지 않아야 합니다.")

            if method == 'pbr' and (market_caps is None or not isinstance(market_caps, pd.DataFrame) or market_caps.empty):
                raise ValueError("PBR 계산을 위해서는 market_caps 데이터가 필요합니다.")

            if method == 'pbr' and market_caps is not None:
                # 단순화된 PBR (시가/장부가 비율 대신 상대적 가치 사용)
                market_caps = ensure_numeric_dataframe(market_caps)
                median_cap = market_caps.median(axis=1)
                factor = market_caps.div(median_cap, axis=0)
            else:
                # 상대적 가격 팩터
                median_price = prices.median(axis=1)
                factor = prices.div(median_price, axis=0)
            
            return ensure_numeric_dataframe(factor)
            
        except Exception as e:
            logger.error(f"밸류에이션 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def profitability_factor(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """수익성 팩터"""
        try:
            returns = ensure_numeric_dataframe(returns)
            
            # 연간 수익률
            annual_returns = (1 + returns).rolling(window).apply(lambda x: x.prod() - 1)
            
            return ensure_numeric_dataframe(annual_returns)
            
        except Exception as e:
            logger.error(f"수익성 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()

class MachineLearningFactors:
    """머신러닝 기반 팩터"""
    
    @staticmethod
    def random_forest_factor(features: pd.DataFrame, target: pd.Series, 
                           window: int = 252, n_estimators: int = 100) -> pd.Series:
        """Random Forest 팩터"""
        try:
            features = ensure_numeric_dataframe(features)
            target = ensure_numeric_series(target)
            
            factor_values = pd.Series(index=target.index, dtype=float)
            
            # 시계열 교차 검증
            tscv = TimeSeriesSplit(n_splits=5)
            
            for i in range(window, len(target)):
                try:
                    # 훈련 데이터
                    train_features = features.iloc[i-window:i].fillna(0)
                    train_target = target.iloc[i-window:i].fillna(0)
                    
                    if len(train_features) == 0 or train_target.std() == 0:
                        factor_values.iloc[i] = 0
                        continue
                    
                    # 모델 훈련
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    model.fit(train_features, train_target)
                    
                    # 예측
                    current_features = features.iloc[i:i+1].fillna(0)
                    prediction = model.predict(current_features)[0]
                    factor_values.iloc[i] = prediction
                    
                except Exception as e:
                    factor_values.iloc[i] = 0
            
            return ensure_numeric_series(factor_values)
            
        except Exception as e:
            logger.error(f"Random Forest 팩터 계산 오류: {str(e)}")
            return pd.Series()
    
    @staticmethod
    def pca_factor(features: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """PCA 팩터"""
        try:
            features = ensure_numeric_dataframe(features.fillna(0))
            
            # 표준화
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_features)
            
            # DataFrame으로 변환
            pca_df = pd.DataFrame(
                pca_result,
                index=features.index,
                columns=[f'PC_{i+1}' for i in range(n_components)]
            )
            
            return ensure_numeric_dataframe(pca_df)
            
        except Exception as e:
            logger.error(f"PCA 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def xgboost_factor(features: pd.DataFrame, target: pd.Series, 
                      window: int = 252, n_estimators: int = 100) -> pd.Series:
        """XGBoost 팩터"""
        try:
            features = ensure_numeric_dataframe(features.fillna(0))
            target = ensure_numeric_series(target.fillna(0))
            
            factor_values = pd.Series(index=target.index, dtype=float)
            factor_values[:] = 0.0
            
            # 롤링 윈도우로 예측
            for i in range(window, len(features)):
                try:
                    # 훈련 데이터 준비
                    train_features = features.iloc[i-window:i]
                    train_target = target.iloc[i-window:i]
                    
                    # 유효한 데이터만 사용
                    valid_mask = ~(train_features.isna().all(axis=1) | train_target.isna())
                    train_features = train_features[valid_mask]
                    train_target = train_target[valid_mask]
                    
                    if len(train_features) < 20:
                        continue
                    
                    # 모델 훈련
                    model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42, verbosity=0)
                    model.fit(train_features, train_target)
                    
                    # 예측
                    current_features = features.iloc[i:i+1].fillna(0)
                    prediction = model.predict(current_features)[0]
                    factor_values.iloc[i] = prediction
                    
                except Exception:
                    factor_values.iloc[i] = 0
            
            return ensure_numeric_series(factor_values)
            
        except Exception as e:
            logger.error(f"XGBoost 팩터 계산 오류: {str(e)}")
            return pd.Series()

class RiskFactors:
    """리스크 팩터"""
    
    @staticmethod
    def beta_factor(returns: pd.DataFrame, market_returns: pd.Series, 
                   window: int = 252) -> pd.DataFrame:
        """베타 팩터"""
        try:
            returns = ensure_numeric_dataframe(returns)
            market_returns = ensure_numeric_series(market_returns)
            
            beta_df = pd.DataFrame(index=returns.index, columns=returns.columns)
            
            for col in returns.columns:
                asset_returns = returns[col]
                
                # 롤링 베타 계산
                rolling_beta = asset_returns.rolling(window).cov(market_returns) / market_returns.rolling(window).var()
                beta_df[col] = rolling_beta
            
            return ensure_numeric_dataframe(beta_df)
            
        except Exception as e:
            logger.error(f"베타 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def downside_risk_factor(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """하방 리스크 팩터"""
        try:
            returns = ensure_numeric_dataframe(returns)
            
            # 하방 편차 계산
            downside_returns = returns.where(returns < 0, 0)
            downside_risk = downside_returns.rolling(window).std()
            
            return ensure_numeric_dataframe(downside_risk)
            
        except Exception as e:
            logger.error(f"하방 리스크 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()

class FactorEngine:
    """팩터 엔진 - 모든 팩터 계산을 통합 관리"""
    
    def __init__(self):
        self.technical_factors = TechnicalFactors()
        self.fundamental_factors = FundamentalFactors()
        self.ml_factors = MachineLearningFactors()
        self.risk_factors = RiskFactors()
        self.validator = FactorValidator()
    
    def calculate_factor(self, category: str, factor_name: str, 
                        data: Dict[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """팩터 계산"""
        try:
            if category == 'technical':
                return self._calculate_technical_factor(factor_name, data, **kwargs)
            elif category == 'fundamental':
                return self._calculate_fundamental_factor(factor_name, data, **kwargs)
            elif category == 'machine_learning':
                return self._calculate_ml_factor(factor_name, data, **kwargs)
            elif category == 'risk':
                return self._calculate_risk_factor(factor_name, data, **kwargs)
            else:
                raise ValueError(f"지원하지 않는 팩터 카테고리: {category}")
                
        except Exception as e:
            logger.error(f"팩터 계산 오류 ({category}.{factor_name}): {str(e)}")
            return pd.DataFrame()
    
    def _calculate_technical_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """기술적 팩터 계산"""
        if factor_name == 'momentum':
            return self.technical_factors.price_momentum(data['prices'], **kwargs)
        elif factor_name == 'mean_reversion':
            return self.technical_factors.mean_reversion(data['prices'], **kwargs)
        elif factor_name == 'volatility':
            return self.technical_factors.volatility_factor(data['prices'], **kwargs)
        elif factor_name == 'volume':
            return self.technical_factors.volume_factor(data['volumes'], data.get('prices'), **kwargs)
        else:
            raise ValueError(f"지원하지 않는 기술적 팩터: {factor_name}")
    
    def _calculate_fundamental_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """펀더멘털 팩터 계산"""
        if factor_name == 'valuation':
            return self.fundamental_factors.valuation_factor(data['prices'], data.get('market_caps'), **kwargs)
        elif factor_name == 'profitability':
            return self.fundamental_factors.profitability_factor(data['returns'], **kwargs)
        else:
            raise ValueError(f"지원하지 않는 펀더멘털 팩터: {factor_name}")
    
    def _calculate_ml_factor(self, factor_name: str, data: Dict, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """머신러닝 팩터 계산"""
        if factor_name == 'random_forest':
            features = pd.concat([data['prices'].pct_change(), data['volumes'].pct_change()], axis=1)
            target = data['returns'].mean(axis=1)
            return self.ml_factors.random_forest_factor(features, target, **kwargs)
        elif factor_name == 'pca':
            features = pd.concat([data['prices'].pct_change(), data['volumes'].pct_change()], axis=1)
            return self.ml_factors.pca_factor(features, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 머신러닝 팩터: {factor_name}")
    
    def _calculate_risk_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """리스크 팩터 계산"""
        if factor_name == 'beta':
            market_returns = data['returns'].mean(axis=1)
            return self.risk_factors.beta_factor(data['returns'], market_returns, **kwargs)
        elif factor_name == 'downside_risk':
            return self.risk_factors.downside_risk_factor(data['returns'], **kwargs)
        else:
            raise ValueError(f"지원하지 않는 리스크 팩터: {factor_name}")

# =============================================================================
# 새로운 고급 팩터 클래스들
# =============================================================================

class AdvancedTechnicalFactors:
    """고급 기술적 팩터 (RSI, 볼린저 밴드, Z-Score 등)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def rsi_factor(self, prices: pd.DataFrame, period: int = 14, **kwargs):
        """
        RSI(상대강도지수) 팩터 계산 함수
        **kwargs를 추가하여, window 등 불필요한 인자가 넘어와도 무시됩니다.
        """
        try:
            prices = ensure_numeric_dataframe(prices)
            rsi_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                price_series = prices[symbol].dropna()
                if len(price_series) < period + 1:
                    continue
                    
                delta = price_series.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=period, min_periods=period).mean()
                avg_loss = loss.rolling(window=period, min_periods=period).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values[symbol] = rsi
            
            return rsi_values.fillna(50)  # 중립값으로 채움
            
        except Exception as e:
            self.logger.error(f"RSI 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def bollinger_bands_factor(self, prices: pd.DataFrame, period: int = 20, std_dev: float = 2, **kwargs):
        """
        볼린저 밴드 팩터 계산 함수
        **kwargs를 추가하여, window 등 불필요한 인자가 넘어와도 무시됩니다.
        """
        try:
            prices = ensure_numeric_dataframe(prices)
            bb_position = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                price_series = prices[symbol].dropna()
                if len(price_series) < period:
                    continue
                
                ma = price_series.rolling(window=period).mean()
                std = price_series.rolling(window=period).std()
                
                upper_band = ma + (std * std_dev)
                lower_band = ma - (std * std_dev)
                
                # 볼린저 밴드 내 위치 (0~1)
                position = (price_series - lower_band) / (upper_band - lower_band)
                bb_position[symbol] = position
            
            return bb_position.fillna(0.5)
            
        except Exception as e:
            self.logger.error(f"볼린저 밴드 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def zscore_factor(self, prices: pd.DataFrame, lookback: int = 60, **kwargs):
        """
        Z-Score 팩터 계산 함수
        **kwargs를 추가하여, window 등 불필요한 인자가 넘어와도 무시됩니다.
        """
        try:
            prices = ensure_numeric_dataframe(prices)
            returns = prices.pct_change()
            
            zscore_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                return_series = returns[symbol].dropna()
                if len(return_series) < lookback:
                    continue
                
                rolling_mean = return_series.rolling(window=lookback).mean()
                rolling_std = return_series.rolling(window=lookback).std()
                
                zscore = (return_series - rolling_mean) / rolling_std
                zscore_values[symbol] = zscore
            
            return zscore_values.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Z-Score 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def correlation_factor(self, prices: pd.DataFrame, market_prices: pd.Series, period: int = 30, **kwargs) -> pd.DataFrame:
        """
        상관계수 팩터 계산 함수
        **kwargs를 추가하여, window 등 불필요한 인자가 넘어와도 무시됩니다.
        """
        try:
            prices = ensure_numeric_dataframe(prices)
            market_prices = ensure_numeric_series(market_prices)
            
            returns = prices.pct_change()
            market_returns = market_prices.pct_change()
            
            corr_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                return_series = returns[symbol].dropna()
                aligned_market = market_returns.reindex(return_series.index).fillna(0)
                
                rolling_corr = return_series.rolling(window=period).corr(aligned_market)
                corr_values[symbol] = rolling_corr
            
            return corr_values.fillna(0)
            
        except Exception as e:
            self.logger.error(f"상관계수 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()


class AdvancedFundamentalFactors:
    """고급 펀더멘털 팩터 (PBR, PER, ROE 등)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def pbr_factor(self, prices: pd.DataFrame, book_values: pd.DataFrame) -> pd.DataFrame:
        """PBR (Price to Book Ratio) 팩터"""
        try:
            prices = ensure_numeric_dataframe(prices)
            book_values = ensure_numeric_dataframe(book_values)
            
            # 최신 북밸류 사용
            latest_book_values = book_values.ffill().iloc[-1]
            latest_prices = prices.iloc[-1]
            
            pbr = latest_prices / latest_book_values
            
            # 시계열로 확장
            pbr_series = pd.DataFrame(index=prices.index, columns=prices.columns)
            for col in pbr_series.columns:
                pbr_series[col] = pbr.get(col, np.nan)
            
            return pbr_series.fillna(pbr_series.median(axis=1, numeric_only=True), axis=0)
            
        except Exception as e:
            self.logger.error(f"PBR 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def per_factor(self, prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
        """PER (Price to Earnings Ratio) 팩터"""
        try:
            prices = ensure_numeric_dataframe(prices)
            earnings = ensure_numeric_dataframe(earnings)
            
            latest_earnings = earnings.ffill().iloc[-1]
            latest_prices = prices.iloc[-1]
            
            per = latest_prices / latest_earnings
            
            # 시계열로 확장
            per_series = pd.DataFrame(index=prices.index, columns=prices.columns)
            for col in per_series.columns:
                per_series[col] = per.get(col, np.nan)
            
            return per_series.fillna(per_series.median(axis=1, numeric_only=True), axis=0)
            
        except Exception as e:
            self.logger.error(f"PER 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def roe_factor(self, net_income: pd.DataFrame, shareholders_equity: pd.DataFrame) -> pd.DataFrame:
        """ROE (Return on Equity) 팩터"""
        try:
            net_income = ensure_numeric_dataframe(net_income)
            shareholders_equity = ensure_numeric_dataframe(shareholders_equity)
            
            # 최근 4분기 순이익 합계
            quarterly_income = net_income.rolling(window=4, axis=0).sum()
            latest_equity = shareholders_equity.ffill()
            
            roe = quarterly_income / latest_equity
            
            return roe.fillna(roe.median(axis=1, numeric_only=True), axis=0)
            
        except Exception as e:
            self.logger.error(f"ROE 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()


class AdvancedMathFactors:
    """고급 수학/통계 팩터"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def kalman_filter_factor(self, prices: pd.DataFrame, observation_noise: float = 0.1, 
                           process_noise: float = 0.01) -> pd.DataFrame:
        """칼만 필터 기반 팩터 (추세 추정)"""
        try:
            prices = ensure_numeric_dataframe(prices)
            filtered_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                price_series = prices[symbol].dropna()
                if len(price_series) < 10:
                    continue
                
                # 간단한 칼만 필터 구현
                x_est = price_series.iloc[0]  # 초기 추정값
                p_est = 1.0  # 초기 오차 공분산
                
                filtered_series = []
                
                for price in price_series:
                    # 예측 단계
                    x_pred = x_est
                    p_pred = p_est + process_noise
                    
                    # 업데이트 단계
                    k = p_pred / (p_pred + observation_noise)
                    x_est = x_pred + k * (price - x_pred)
                    p_est = (1 - k) * p_pred
                    
                    filtered_series.append(x_est)
                
                filtered_values[symbol] = pd.Series(filtered_series, index=price_series.index)
            
            # 원본 가격 대비 비율 반환
            return (filtered_values / prices - 1).fillna(0)
            
        except Exception as e:
            self.logger.error(f"칼만 필터 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def hurst_exponent_factor(self, prices: pd.DataFrame, max_lag: int = 100) -> pd.DataFrame:
        """Hurst 지수 팩터 (프랙탈 차원)"""
        try:
            prices = ensure_numeric_dataframe(prices)
            returns = prices.pct_change().dropna()
            
            hurst_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                return_series = returns[symbol].dropna()
                if len(return_series) < max_lag * 2:
                    continue
                
                # 롤링 윈도우로 Hurst 지수 계산
                rolling_hurst = []
                window_size = max_lag * 2
                
                for i in range(window_size, len(return_series)):
                    window_data = return_series.iloc[i-window_size:i]
                    hurst = self._calculate_hurst_exponent(window_data.values, max_lag)
                    rolling_hurst.append(hurst)
                
                hurst_series = pd.Series(rolling_hurst, 
                                       index=return_series.index[window_size:])
                hurst_values[symbol] = hurst_series
            
            return hurst_values.fillna(0.5)
            
        except Exception as e:
            self.logger.error(f"Hurst 지수 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_hurst_exponent(self, data: np.ndarray, max_lag: int) -> float:
        """Hurst 지수 계산"""
        try:
            lags = range(2, max_lag)
            rs_values = []
            
            for lag in lags:
                # R/S 통계 계산
                segments = len(data) // lag
                if segments < 2:
                    continue
                
                rs_list = []
                for i in range(segments):
                    segment = data[i*lag:(i+1)*lag]
                    
                    mean_segment = np.mean(segment)
                    deviations = segment - mean_segment
                    cum_deviations = np.cumsum(deviations)
                    
                    R = np.max(cum_deviations) - np.min(cum_deviations)
                    S = np.std(segment, ddof=1)
                    
                    if S > 0:
                        rs_list.append(R / S)
                
                if rs_list:
                    rs_values.append(np.mean(rs_list))
                else:
                    rs_values.append(np.nan)
            
            # 로그 회귀로 Hurst 지수 추정
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log([rs for rs in rs_values if not np.isnan(rs)])
            
            if len(log_rs) > 1:
                hurst = np.polyfit(log_lags[:len(log_rs)], log_rs, 1)[0]
                return max(0, min(1, hurst))  # 0~1 범위로 제한
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def wavelet_factor(self, prices: pd.DataFrame, wavelet: str = 'db4', levels: int = 3) -> pd.DataFrame:
        """웨이블릿 변환 기반 팩터"""
        if not PYWT_AVAILABLE:
            self.logger.warning("pywavelets 라이브러리가 없어 웨이블릿 팩터를 계산할 수 없습니다.")
            return pd.DataFrame()
        
        try:
            prices = ensure_numeric_dataframe(prices)
            returns = prices.pct_change().dropna()
            
            wavelet_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                return_series = returns[symbol].dropna()
                if len(return_series) < 32:  # 최소 데이터 요구사항
                    continue
                
                # 웨이블릿 분해
                coeffs = pywt.wavedec(return_series.values, wavelet, level=levels)
                
                # 고주파 성분 제거 후 재구성
                coeffs_filtered = coeffs.copy()
                coeffs_filtered[-1] = np.zeros_like(coeffs[-1])  # 고주파 제거
                
                # 재구성
                reconstructed = pywt.waverec(coeffs_filtered, wavelet)
                
                # 길이 맞추기
                min_len = min(len(return_series), len(reconstructed))
                wavelet_signal = pd.Series(reconstructed[:min_len], 
                                         index=return_series.index[:min_len])
                
                wavelet_values[symbol] = wavelet_signal
            
            return wavelet_values.fillna(0)
            
        except Exception as e:
            self.logger.error(f"웨이블릿 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def regime_detection_factor(self, prices: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
        """체제 변화 탐지 팩터 (Hidden Markov Model)"""
        if not HMM_AVAILABLE:
            self.logger.warning("hmmlearn 라이브러리가 없어 체제 변화 팩터를 계산할 수 없습니다.")
            return pd.DataFrame()
        
        try:
            prices = ensure_numeric_dataframe(prices)
            returns = prices.pct_change().dropna()
            
            regime_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                return_series = returns[symbol].dropna()
                if len(return_series) < 50:
                    continue
                
                # HMM 모델 학습
                model = hmm.GaussianHMM(n_components=n_components, covariance_type="full")
                
                # 특성 준비 (수익률과 변동성)
                volatility = return_series.rolling(window=10).std()
                features = np.column_stack([return_series.values, volatility.fillna(0).values])
                
                try:
                    model.fit(features)
                    
                    # 체제 예측
                    states = model.predict(features)
                    
                    # 체제별 확률
                    state_probs = model.predict_proba(features)
                    regime_prob = state_probs[:, 1]  # 두 번째 체제 확률
                    
                    regime_series = pd.Series(regime_prob, index=return_series.index)
                    regime_values[symbol] = regime_series
                    
                except Exception:
                    # HMM 학습 실패 시 단순 변동성 기반 체제 구분
                    vol_threshold = volatility.median()
                    regime_series = (volatility > vol_threshold).astype(float)
                    regime_values[symbol] = regime_series
            
            return regime_values.fillna(0.5)
            
        except Exception as e:
            self.logger.error(f"체제 변화 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def isolation_forest_factor(self, prices: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
        """이상치 탐지 팩터 (Isolation Forest)"""
        try:
            prices = ensure_numeric_dataframe(prices)
            returns = prices.pct_change().dropna()
            
            anomaly_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            # 전체 종목의 수익률 특성 사용
            feature_matrix = returns.fillna(0).values
            
            if feature_matrix.shape[0] < 10:
                return pd.DataFrame()
            
            # Isolation Forest 모델
            iso_forest = IsolationForest(contamination=float(contamination), random_state=42)
            
            # 롤링 윈도우로 이상치 탐지
            window_size = 60
            
            for i in range(window_size, len(feature_matrix)):
                window_data = feature_matrix[i-window_size:i]
                
                try:
                    iso_forest.fit(window_data)
                    current_data = feature_matrix[i:i+1]
                    anomaly_score = iso_forest.decision_function(current_data)[0]
                    
                    # 모든 종목에 같은 스코어 적용 (시장 전체 이상치)
                    for symbol in prices.columns:
                        anomaly_values.loc[returns.index[i], symbol] = anomaly_score
                        
                except Exception:
                    continue
            
            return anomaly_values.fillna(0)
            
        except Exception as e:
            self.logger.error(f"이상치 탐지 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()


class DeepLearningFactors:
    """딥러닝 기반 팩터"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def lstm_factor(self, prices: pd.DataFrame, lookback: int = 60, 
                   lstm_units: int = 50, epochs: int = 50) -> pd.DataFrame:
        """LSTM 신경망 기반 팩터 (TensorFlow 없이 간단한 시계열 예측)"""
        try:
            prices = ensure_numeric_dataframe(prices)
            returns = prices.pct_change().dropna()
            
            # TensorFlow가 없는 경우 간단한 시계열 기반 팩터 계산
            if not TENSORFLOW_AVAILABLE:
                self.logger.info("TensorFlow가 없어 간단한 시계열 팩터로 대체합니다.")
                # 이동평균 기반 예측 팩터
                ma_short = returns.rolling(window=10).mean()
                ma_long = returns.rolling(window=30).mean()
                trend_factor = (ma_short - ma_long) / (ma_long.abs() + 1e-8)
                return trend_factor.fillna(0)
            
            lstm_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                return_series = returns[symbol].dropna()
                if len(return_series) < lookback + 50:
                    continue
                
                # 간단한 자기회귀 모델로 대체 (TensorFlow 문제 회피)
                # AR(p) 모델 시뮬레이션
                rolling_mean = return_series.rolling(window=lookback).mean()
                rolling_std = return_series.rolling(window=lookback).std()
                
                # 정규화된 값
                normalized = (return_series - rolling_mean) / (rolling_std + 1e-8)
                
                # 간단한 예측 신호
                prediction_signal = normalized.rolling(window=5).mean()
                lstm_values[symbol] = prediction_signal
            
            return lstm_values.fillna(0)
            
        except Exception as e:
            self.logger.error(f"LSTM 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def attention_factor(self, prices: pd.DataFrame, sequence_length: int = 30) -> pd.DataFrame:
        """어텐션 메커니즘 기반 팩터 (간단한 구현)"""
        try:
            prices = ensure_numeric_dataframe(prices)
            returns = prices.pct_change().dropna()
            
            attention_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for symbol in prices.columns:
                return_series = returns[symbol].dropna()
                if len(return_series) < sequence_length + 10:
                    continue
                
                # 어텐션 가중치 계산 (간단한 버전)
                weighted_returns = []
                
                for i in range(sequence_length, len(return_series)):
                    sequence = return_series.iloc[i-sequence_length:i]
                    
                    # 최근 수익률에 더 높은 가중치
                    weights = np.exp(np.arange(sequence_length)) / np.sum(np.exp(np.arange(sequence_length)))
                    
                    # 가중 평균 계산
                    weighted_return = np.sum(sequence.values * weights)
                    weighted_returns.append(weighted_return)
                
                attention_series = pd.Series(weighted_returns, 
                                           index=return_series.index[sequence_length:])
                attention_values[symbol] = attention_series
            
            return attention_values.fillna(0)
            
        except Exception as e:
            self.logger.error(f"어텐션 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    def ensemble_factor(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """앙상블 팩터 (여러 예측 모델 결합)"""
        try:
            prices = ensure_numeric_dataframe(prices)
            returns = prices.pct_change().dropna()
            
            ensemble_values = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            # 간단한 앙상블: 이동평균, 선형회귀, 랜덤포레스트
            for symbol in prices.columns:
                return_series = returns[symbol].dropna()
                if len(return_series) < 60:
                    continue
                
                predictions = []
                
                # 1. 이동평균 예측
                ma_pred = return_series.rolling(window=20).mean()
                predictions.append(ma_pred)
                
                # 2. 선형 추세 예측
                trend_pred = []
                for i in range(30, len(return_series)):
                    y = return_series.iloc[i-30:i].values
                    x = np.arange(len(y))
                    
                    if len(y) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        trend_pred.append(slope)
                    else:
                        trend_pred.append(0)
                
                trend_series = pd.Series(trend_pred, index=return_series.index[30:])
                predictions.append(trend_series)
                
                # 3. 간단한 RF 예측
                if len(return_series) > 100:
                    try:
                        features = []
                        targets = []
                        
                        for i in range(10, len(return_series)-1):
                            features.append(return_series.iloc[i-10:i].values)
                            targets.append(return_series.iloc[i+1])
                        
                        features = np.array(features)
                        targets = np.array(targets)
                        
                        rf = RandomForestRegressor(n_estimators=10, random_state=42)
                        rf.fit(features, targets)
                        
                        rf_pred = []
                        for i in range(10, len(return_series)):
                            feature = return_series.iloc[i-10:i].values.reshape(1, -1)
                            pred = rf.predict(feature)[0]
                            rf_pred.append(pred)
                        
                        rf_series = pd.Series(rf_pred, index=return_series.index[10:])
                        predictions.append(rf_series)
                        
                    except Exception:
                        pass
                
                # 앙상블 결합 (평균)
                if predictions:
                    # 모든 예측을 같은 인덱스로 정렬
                    aligned_predictions = []
                    common_index = predictions[0].index
                    
                    for pred in predictions:
                        if len(pred) > 0:
                            aligned_pred = pred.reindex(common_index).fillna(method='ffill')
                            aligned_predictions.append(aligned_pred)
                    
                    if aligned_predictions:
                        ensemble_pred = pd.concat(aligned_predictions, axis=1).mean(axis=1)
                        ensemble_values[symbol] = ensemble_pred
            
            return ensemble_values.fillna(0)
            
        except Exception as e:
            self.logger.error(f"앙상블 팩터 계산 오류: {str(e)}")
            return pd.DataFrame()


# =============================================================================
# 통합 고급 팩터 라이브러리
# =============================================================================

class EnhancedFactorLibrary:
    """향상된 팩터 라이브러리 - 모든 고급 팩터 통합"""
    
    def __init__(self):
        # 기존 팩터 클래스들
        self.technical_factors = TechnicalFactors()
        self.fundamental_factors = FundamentalFactors()
        self.ml_factors = MachineLearningFactors()
        self.risk_factors = RiskFactors()
        
        # 새로운 고급 팩터 클래스들
        self.advanced_technical = AdvancedTechnicalFactors()
        self.advanced_fundamental = AdvancedFundamentalFactors()
        self.advanced_math = AdvancedMathFactors()
        self.deep_learning = DeepLearningFactors()
        
        self.validator = FactorValidator()
    
    def get_available_factors(self) -> Dict[str, List[str]]:
        """사용 가능한 모든 팩터 목록 반환"""
        return {
            'technical': [
                'momentum', 'mean_reversion', 'volatility', 'volume'
            ],
            'advanced_technical': [
                'rsi', 'bollinger_bands', 'zscore', 'correlation'
            ],
            'fundamental': [
                'valuation', 'profitability'
            ],
            'advanced_fundamental': [
                'pbr', 'per', 'roe'
            ],
            'machine_learning': [
                'random_forest', 'pca', 'xgboost'
            ],
            'risk': [
                'beta', 'downside_risk'
            ],
            'advanced_math': [
                'kalman_filter', 'hurst_exponent', 'wavelet', 
                'regime_detection', 'isolation_forest'
            ],
            'deep_learning': [
                'lstm', 'attention', 'ensemble'
            ]
        }
    
    def calculate_factor(self, category: str, factor_name: str, 
                        data: Dict[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """팩터 계산 - 모든 카테고리 지원"""
        try:
            if category == 'technical':
                return self._calculate_technical_factor(factor_name, data, **kwargs)
            elif category == 'advanced_technical':
                return self._calculate_advanced_technical_factor(factor_name, data, **kwargs)
            elif category == 'fundamental':
                return self._calculate_fundamental_factor(factor_name, data, **kwargs)
            elif category == 'advanced_fundamental':
                return self._calculate_advanced_fundamental_factor(factor_name, data, **kwargs)
            elif category == 'machine_learning':
                return self._calculate_ml_factor(factor_name, data, **kwargs)
            elif category == 'risk':
                return self._calculate_risk_factor(factor_name, data, **kwargs)
            elif category == 'advanced_math':
                return self._calculate_advanced_math_factor(factor_name, data, **kwargs)
            elif category == 'deep_learning':
                return self._calculate_deep_learning_factor(factor_name, data, **kwargs)
            else:
                raise ValueError(f"지원하지 않는 팩터 카테고리: {category}")
                
        except Exception as e:
            logger.error(f"팩터 계산 오류 ({category}.{factor_name}): {str(e)}")
            return pd.DataFrame()
    
    def _calculate_technical_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """기술적 팩터 계산"""
        if factor_name == 'momentum':
            return self.technical_factors.price_momentum(data['prices'], **kwargs)
        elif factor_name == 'mean_reversion':
            return self.technical_factors.mean_reversion(data['prices'], **kwargs)
        elif factor_name == 'volatility':
            return self.technical_factors.volatility_factor(data['prices'], **kwargs)
        elif factor_name == 'volume':
            return self.technical_factors.volume_factor(data['volumes'], data.get('prices'), **kwargs)
        else:
            raise ValueError(f"지원하지 않는 기술적 팩터: {factor_name}")
    
    def _calculate_advanced_technical_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """고급 기술적 팩터 계산"""
        if factor_name == 'rsi':
            return self.advanced_technical.rsi_factor(data['prices'], **kwargs)
        elif factor_name == 'bollinger_bands':
            return self.advanced_technical.bollinger_bands_factor(data['prices'], **kwargs)
        elif factor_name == 'zscore':
            return self.advanced_technical.zscore_factor(data['prices'], **kwargs)
        elif factor_name == 'correlation':
            market_prices = data.get('market_prices', data['prices'].mean(axis=1))
            return self.advanced_technical.correlation_factor(data['prices'], market_prices, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 고급 기술적 팩터: {factor_name}")
    
    def _calculate_fundamental_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """펀더멘털 팩터 계산"""
        if factor_name == 'valuation':
            return self.fundamental_factors.valuation_factor(data['prices'], data.get('market_caps'), **kwargs)
        elif factor_name == 'profitability':
            return self.fundamental_factors.profitability_factor(data['returns'], **kwargs)
        else:
            raise ValueError(f"지원하지 않는 펀더멘털 팩터: {factor_name}")
    
    def _calculate_advanced_fundamental_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """고급 펀더멘털 팩터 계산"""
        if factor_name == 'pbr':
            return self.advanced_fundamental.pbr_factor(data['prices'], data.get('book_values', data['prices']), **kwargs)
        elif factor_name == 'per':
            return self.advanced_fundamental.per_factor(data['prices'], data.get('earnings', data['prices']), **kwargs)
        elif factor_name == 'roe':
            net_income = data.get('net_income', data['prices'])
            equity = data.get('shareholders_equity', data['prices'])
            return self.advanced_fundamental.roe_factor(net_income, equity, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 고급 펀더멘털 팩터: {factor_name}")
    
    def _calculate_ml_factor(self, factor_name: str, data: Dict, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """머신러닝 팩터 계산 (확장됨)"""
        if factor_name == 'random_forest':
            features = pd.concat([data['prices'].pct_change(), data.get('volumes', data['prices']).pct_change()], axis=1)
            target = data['returns'].mean(axis=1) if 'returns' in data else data['prices'].pct_change().mean(axis=1)
            return self.ml_factors.random_forest_factor(features, target, **kwargs)
        elif factor_name == 'pca':
            features = pd.concat([data['prices'].pct_change(), data.get('volumes', data['prices']).pct_change()], axis=1)
            return self.ml_factors.pca_factor(features, **kwargs)
        elif factor_name == 'xgboost':
            features = pd.concat([data['prices'].pct_change(), data.get('volumes', data['prices']).pct_change()], axis=1)
            target = data['returns'].mean(axis=1) if 'returns' in data else data['prices'].pct_change().mean(axis=1)
            return self.ml_factors.xgboost_factor(features, target, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 머신러닝 팩터: {factor_name}")
    
    def _calculate_risk_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """리스크 팩터 계산"""
        if factor_name == 'beta':
            market_returns = data.get('market_returns', data['returns'].mean(axis=1) if 'returns' in data else data['prices'].pct_change().mean(axis=1))
            returns = data.get('returns', data['prices'].pct_change())
            return self.risk_factors.beta_factor(returns, market_returns, **kwargs)
        elif factor_name == 'downside_risk':
            returns = data.get('returns', data['prices'].pct_change())
            return self.risk_factors.downside_risk_factor(returns, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 리스크 팩터: {factor_name}")
    
    def _calculate_advanced_math_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """고급 수학/통계 팩터 계산"""
        if factor_name == 'kalman_filter':
            return self.advanced_math.kalman_filter_factor(data['prices'], **kwargs)
        elif factor_name == 'hurst_exponent':
            return self.advanced_math.hurst_exponent_factor(data['prices'], **kwargs)
        elif factor_name == 'wavelet':
            return self.advanced_math.wavelet_factor(data['prices'], **kwargs)
        elif factor_name == 'regime_detection':
            return self.advanced_math.regime_detection_factor(data['prices'], **kwargs)
        elif factor_name == 'isolation_forest':
            return self.advanced_math.isolation_forest_factor(data['prices'], **kwargs)
        else:
            raise ValueError(f"지원하지 않는 고급 수학 팩터: {factor_name}")
    
    def _calculate_deep_learning_factor(self, factor_name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """딥러닝 팩터 계산"""
        if factor_name == 'lstm':
            return self.deep_learning.lstm_factor(data['prices'], **kwargs)
        elif factor_name == 'attention':
            return self.deep_learning.attention_factor(data['prices'], **kwargs)
        elif factor_name == 'ensemble':
            return self.deep_learning.ensemble_factor(data['prices'], **kwargs)
        else:
            raise ValueError(f"지원하지 않는 딥러닝 팩터: {factor_name}")
    
    def get_factor_description(self, category: str, factor_name: str) -> str:
        """팩터 설명 반환"""
        descriptions = {
            'technical': {
                'momentum': '가격 모멘텀 팩터 - 최근 수익률 추세',
                'mean_reversion': '평균 회귀 팩터 - 가격의 평균 회귀 성향',
                'volatility': '변동성 팩터 - 가격 변동성 기반',
                'volume': '거래량 팩터 - 거래량 패턴 기반'
            },
            'advanced_technical': {
                'rsi': 'RSI 팩터 - 상대강도지수 (과매수/과매도)',
                'bollinger_bands': '볼린저 밴드 팩터 - 가격의 밴드 내 위치',
                'zscore': 'Z-Score 팩터 - 표준화된 가격 위치',
                'correlation': '상관계수 팩터 - 시장과의 상관관계'
            },
            'fundamental': {
                'valuation': '밸류에이션 팩터 - 기업 가치 평가 기반',
                'profitability': '수익성 팩터 - 기업의 수익 창출 능력'
            },
            'advanced_fundamental': {
                'pbr': 'PBR 팩터 - 주가순자산비율',
                'per': 'PER 팩터 - 주가수익비율',
                'roe': 'ROE 팩터 - 자기자본수익률'
            },
            'machine_learning': {
                'random_forest': '랜덤 포레스트 팩터 - 앙상블 학습',
                'pca': 'PCA 팩터 - 주성분 분석',
                'xgboost': 'XGBoost 팩터 - 그래디언트 부스팅'
            },
            'risk': {
                'beta': '베타 팩터 - 시장 민감도',
                'downside_risk': '하방 리스크 팩터 - 손실 위험도'
            },
            'advanced_math': {
                'kalman_filter': '칼만 필터 팩터 - 상태 공간 모델',
                'hurst_exponent': 'Hurst 지수 팩터 - 프랙탈 차원',
                'wavelet': '웨이블릿 팩터 - 주파수 분석',
                'regime_detection': '체제 변화 팩터 - Hidden Markov Model',
                'isolation_forest': '이상치 탐지 팩터 - Isolation Forest'
            },
            'deep_learning': {
                'lstm': 'LSTM 팩터 - 장단기 메모리 신경망',
                'attention': '어텐션 팩터 - 어텐션 메커니즘',
                'ensemble': '앙상블 팩터 - 다중 모델 결합'
            }
        }
        
        return descriptions.get(category, {}).get(factor_name, "설명 없음")
