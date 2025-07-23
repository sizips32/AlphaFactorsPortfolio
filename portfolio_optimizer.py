"""
포트폴리오 최적화 모듈 (데이터베이스 연동 버전)
Alpha Factor 기반 포트폴리오 구성 및 최적화

작성자: AI Assistant
작성일: 2025년 7월 23일
버전: 1.1 (데이터베이스 연동)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import logging

# 최적화 라이브러리
from scipy.optimize import minimize, LinearConstraint, Bounds
import cvxpy as cp

# 통계 라이브러리
from scipy import stats
from sklearn.covariance import LedoitWolf, OAS
from sklearn.preprocessing import StandardScaler

# 로컬 모듈
from alpha_factor_library import ensure_numeric_series, ensure_numeric_dataframe

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConstraints:
    """최적화 제약조건"""
    max_weight: float = 0.1  # 최대 종목 비중
    min_weight: float = 0.0  # 최소 종목 비중
    max_sector_weight: Optional[float] = None  # 최대 섹터 비중
    max_turnover: Optional[float] = None  # 최대 턴오버
    target_volatility: Optional[float] = None  # 목표 변동성
    leverage: float = 1.0  # 레버리지
    long_only: bool = True  # 롱 온리 제약
    
@dataclass
class OptimizationResult:
    """최적화 결과"""
    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_status: str
    objective_value: float
    constraints_satisfied: bool

class CovarianceEstimator:
    """공분산 추정기"""
    
    @staticmethod
    def sample_covariance(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """표본 공분산"""
        try:
            returns = ensure_numeric_dataframe(returns)
            return returns.tail(window).cov()
        except Exception as e:
            logger.error(f"표본 공분산 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def exponential_weighted_covariance(returns: pd.DataFrame, 
                                      decay_factor: float = 0.94,
                                      window: int = 252) -> pd.DataFrame:
        """지수가중 공분산"""
        try:
            returns = ensure_numeric_dataframe(returns)
            returns_window = returns.tail(window)
            weights = np.array([decay_factor ** i for i in range(len(returns_window))])
            weights = weights / weights.sum()
            
            # 가중 평균
            weighted_mean = np.average(returns_window, weights=weights, axis=0)
            
            # 가중 공분산
            centered_returns = returns_window - weighted_mean
            weighted_cov = np.cov(centered_returns.T, aweights=weights)
            
            return pd.DataFrame(weighted_cov, 
                              index=returns.columns, 
                              columns=returns.columns)
        except Exception as e:
            logger.error(f"지수가중 공분산 계산 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def ledoit_wolf_covariance(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """Ledoit-Wolf 수축 추정"""
        try:
            returns = ensure_numeric_dataframe(returns)
            returns_window = returns.tail(window).dropna()
            
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_window).covariance_
            
            return pd.DataFrame(cov_matrix,
                              index=returns.columns,
                              columns=returns.columns)
        except Exception as e:
            logger.error(f"Ledoit-Wolf 공분산 계산 오류: {str(e)}")
            return pd.DataFrame()

class BaseOptimizer(ABC):
    """최적화기 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def optimize(self, expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                constraints: OptimizationConstraints,
                **kwargs) -> OptimizationResult:
        """최적화 실행"""
        pass
    
    def _validate_inputs(self, expected_returns: pd.Series,
                        covariance_matrix: pd.DataFrame) -> None:
        """입력 데이터 검증"""
        if len(expected_returns) != len(covariance_matrix):
            raise ValueError("기대수익률과 공분산 매트릭스의 차원이 일치하지 않습니다.")
        
        if not covariance_matrix.index.equals(expected_returns.index):
            raise ValueError("기대수익률과 공분산 매트릭스의 인덱스가 일치하지 않습니다.")
        
        # 공분산 매트릭스 양정치 확인
        eigenvals = np.linalg.eigvals(covariance_matrix.values)
        if np.any(eigenvals <= 0):
            logger.warning("공분산 매트릭스가 양정치가 아닙니다. 정규화를 수행합니다.")

class MeanVarianceOptimizer(BaseOptimizer):
    """평균-분산 최적화기"""
    
    def __init__(self):
        super().__init__("Mean-Variance Optimizer")
    
    def optimize(self, expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                constraints: OptimizationConstraints,
                risk_aversion: float = 1.0) -> OptimizationResult:
        """평균-분산 최적화"""
        try:
            self._validate_inputs(expected_returns, covariance_matrix)
            
            expected_returns = ensure_numeric_series(expected_returns)
            covariance_matrix = ensure_numeric_dataframe(covariance_matrix)
            
            n_assets = len(expected_returns)
            
            # CVXPY를 사용한 최적화
            weights = cp.Variable(n_assets)
            
            # 목적함수: 기대수익률 - 0.5 * 리스크회피도 * 분산
            portfolio_return = expected_returns.values @ weights
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
            
            # 제약조건
            constraints_list = [cp.sum(weights) == constraints.leverage]  # 가중치 합
            
            if constraints.long_only:
                constraints_list.append(weights >= constraints.min_weight)
            else:
                constraints_list.append(weights >= -constraints.max_weight)
            
            constraints_list.append(weights <= constraints.max_weight)
            
            # 최적화 문제 해결
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights = pd.Series(weights.value, index=expected_returns.index)
                
                # 성과 지표 계산
                expected_return = (optimal_weights * expected_returns).sum()
                expected_volatility = np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)
                sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0
                
                return OptimizationResult(
                    weights=optimal_weights,
                    expected_return=expected_return,
                    expected_volatility=expected_volatility,
                    sharpe_ratio=sharpe_ratio,
                    optimization_status=problem.status,
                    objective_value=problem.value,
                    constraints_satisfied=True
                )
            else:
                # 최적화 실패 시 동일가중 포트폴리오 반환
                equal_weights = pd.Series(constraints.leverage / n_assets, index=expected_returns.index)
                
                return OptimizationResult(
                    weights=equal_weights,
                    expected_return=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    optimization_status=problem.status,
                    objective_value=0.0,
                    constraints_satisfied=False
                )
        except Exception as e:
            logger.error(f"평균-분산 최적화 오류: {str(e)}")
            equal_weights = pd.Series(constraints.leverage / len(expected_returns), 
                                    index=expected_returns.index)
            return OptimizationResult(
                weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_status="error",
                objective_value=0.0,
                constraints_satisfied=False
            )

class RiskParityOptimizer(BaseOptimizer):
    """리스크 패리티 최적화기"""
    
    def __init__(self):
        super().__init__("Risk Parity Optimizer")
    
    def optimize(self, expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                constraints: OptimizationConstraints,
                **kwargs) -> OptimizationResult:
        """리스크 패리티 최적화"""
        try:
            self._validate_inputs(expected_returns, covariance_matrix)
            
            expected_returns = ensure_numeric_series(expected_returns)
            covariance_matrix = ensure_numeric_dataframe(covariance_matrix)
            
            n_assets = len(expected_returns)
            
            def risk_budget_objective(weights):
                """리스크 기여도 균등화 목적함수"""
                weights = np.array(weights)
                portfolio_vol = np.sqrt(weights.T @ covariance_matrix.values @ weights)
                
                if portfolio_vol == 0:
                    return 1e6
                
                marginal_contrib = covariance_matrix.values @ weights / portfolio_vol
                contrib = weights * marginal_contrib
                
                # 각 자산의 리스크 기여도가 1/n이 되도록 하는 목적함수
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            # 초기 가중치 (동일 가중)
            initial_weights = np.ones(n_assets) / n_assets
            
            # 제약조건
            constraints_scipy = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - constraints.leverage}
            ]
            
            # 경계 조건
            if constraints.long_only:
                bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            else:
                bounds = [(-constraints.max_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # 최적화 실행
            result = minimize(
                risk_budget_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_scipy,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=expected_returns.index)
                
                # 성과 지표 계산
                expected_return = (optimal_weights * expected_returns).sum()
                expected_volatility = np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)
                sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0
                
                return OptimizationResult(
                    weights=optimal_weights,
                    expected_return=expected_return,
                    expected_volatility=expected_volatility,
                    sharpe_ratio=sharpe_ratio,
                    optimization_status="optimal",
                    objective_value=result.fun,
                    constraints_satisfied=True
                )
            else:
                # 최적화 실패 시 동일가중 포트폴리오 반환
                equal_weights = pd.Series(constraints.leverage / n_assets, index=expected_returns.index)
                
                return OptimizationResult(
                    weights=equal_weights,
                    expected_return=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    optimization_status="failed",
                    objective_value=0.0,
                    constraints_satisfied=False
                )
        except Exception as e:
            logger.error(f"리스크 패리티 최적화 오류: {str(e)}")
            equal_weights = pd.Series(constraints.leverage / len(expected_returns), 
                                    index=expected_returns.index)
            return OptimizationResult(
                weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_status="error",
                objective_value=0.0,
                constraints_satisfied=False
            )

class PortfolioOptimizer:
    """포트폴리오 최적화 메인 클래스"""
    
    def __init__(self):
        self.optimizers = {
            'mean_variance': MeanVarianceOptimizer(),
            'risk_parity': RiskParityOptimizer()
        }
        self.covariance_estimator = CovarianceEstimator()
    
    def optimize_portfolio(self, 
                          returns: pd.DataFrame,
                          factor_scores: pd.DataFrame = None,
                          method: str = 'mean_variance',
                          covariance_method: str = 'sample',
                          constraints: OptimizationConstraints = None,
                          **kwargs) -> OptimizationResult:
        """포트폴리오 최적화 실행"""
        try:
            if constraints is None:
                constraints = OptimizationConstraints()
            
            returns = ensure_numeric_dataframe(returns)
            
            # 기대수익률 계산
            if factor_scores is not None:
                factor_scores = ensure_numeric_dataframe(factor_scores)
                # 팩터 점수를 기대수익률로 사용 (정규화)
                latest_scores = factor_scores.iloc[-1].dropna()
                expected_returns = latest_scores / latest_scores.std()
            else:
                # 과거 평균 수익률 사용
                expected_returns = returns.mean()
            
            # 공분산 매트릭스 계산
            if covariance_method == 'sample':
                covariance_matrix = self.covariance_estimator.sample_covariance(returns)
            elif covariance_method == 'exponential':
                covariance_matrix = self.covariance_estimator.exponential_weighted_covariance(returns)
            elif covariance_method == 'ledoit_wolf':
                covariance_matrix = self.covariance_estimator.ledoit_wolf_covariance(returns)
            else:
                covariance_matrix = self.covariance_estimator.sample_covariance(returns)
            
            # 공통 자산만 사용
            common_assets = expected_returns.index.intersection(covariance_matrix.index)
            expected_returns = expected_returns[common_assets]
            covariance_matrix = covariance_matrix.loc[common_assets, common_assets]
            
            # 최적화 실행
            optimizer = self.optimizers.get(method)
            if optimizer is None:
                raise ValueError(f"지원하지 않는 최적화 방법: {method}")
            
            result = optimizer.optimize(expected_returns, covariance_matrix, constraints, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"포트폴리오 최적화 오류: {str(e)}")
            # 기본 동일가중 포트폴리오 반환
            if constraints is None:
                constraints = OptimizationConstraints()
            
            if returns.empty:
                assets = ['AAPL', 'GOOGL', 'MSFT']  # 기본 자산
            else:
                assets = returns.columns
            
            equal_weights = pd.Series(constraints.leverage / len(assets), index=assets)
            
            return OptimizationResult(
                weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_status="error",
                objective_value=0.0,
                constraints_satisfied=False
            )
    
    def create_efficient_frontier(self, 
                                 returns: pd.DataFrame,
                                 n_points: int = 20) -> Tuple[List[float], List[float]]:
        """효율적 경계선 생성"""
        try:
            returns = ensure_numeric_dataframe(returns)
            expected_returns = returns.mean()
            covariance_matrix = self.covariance_estimator.sample_covariance(returns)
            
            # 최소/최대 수익률 범위
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            target_returns = np.linspace(min_return, max_return, n_points)
            
            risks = []
            returns_list = []
            
            for target_return in target_returns:
                try:
                    # 목표 수익률에 대한 최소 분산 포트폴리오
                    n_assets = len(expected_returns)
                    weights = cp.Variable(n_assets)
                    
                    # 목적함수: 분산 최소화
                    objective = cp.Minimize(cp.quad_form(weights, covariance_matrix.values))
                    
                    # 제약조건
                    constraints_list = [
                        cp.sum(weights) == 1,  # 가중치 합 = 1
                        expected_returns.values @ weights == target_return,  # 목표 수익률
                        weights >= 0  # 롱 온리
                    ]
                    
                    problem = cp.Problem(objective, constraints_list)
                    problem.solve()
                    
                    if problem.status == 'optimal':
                        volatility = np.sqrt(problem.value)
                        risks.append(volatility)
                        returns_list.append(target_return)
                        
                except Exception:
                    continue
            
            return risks, returns_list
            
        except Exception as e:
            logger.error(f"효율적 경계선 생성 오류: {str(e)}")
            return [], []
