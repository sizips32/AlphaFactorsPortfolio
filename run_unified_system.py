"""
통합 알파 팩터 포트폴리오 시스템 런처
전체 시스템을 실행하고 검증하는 스크립트

작성자: AI Assistant  
작성일: 2025년 1월
버전: 1.0
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """필수 라이브러리 확인"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'yfinance',
        'sklearn',
        'scipy',
        'cvxpy',
        'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} - 누락")
    
    if missing_packages:
        logger.error(f"누락된 패키지들: {', '.join(missing_packages)}")
        logger.info("다음 명령어로 설치하세요:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_file_structure():
    """파일 구조 확인"""
    required_files = [
        'alpha_factor_library.py',
        'backtesting_engine.py', 
        'portfolio_optimizer.py',
        'database.py',
        'zscore.py',
        'unified_alpha_system.py',
        'unified_app.py'
    ]
    
    current_dir = Path.cwd()
    missing_files = []
    
    for file in required_files:
        file_path = current_dir / file
        if file_path.exists():
            logger.info(f"✅ {file} - 존재")
        else:
            missing_files.append(file)
            logger.error(f"❌ {file} - 누락")
    
    if missing_files:
        logger.error(f"누락된 파일들: {', '.join(missing_files)}")
        return False
    
    return True

def test_system_components():
    """시스템 구성요소 테스트"""
    try:
        logger.info("🔧 시스템 구성요소 테스트 시작...")
        
        # 1. 기본 모듈 import 테스트
        from alpha_factor_library import EnhancedFactorLibrary
        from backtesting_engine import BacktestEngine, BacktestConfig
        from portfolio_optimizer import PortfolioOptimizer
        from database import DatabaseManager
        from zscore import FactorZScoreCalculator
        from unified_alpha_system import UnifiedAlphaSystem, UnifiedSystemConfig
        
        logger.info("✅ 모든 모듈 import 성공")
        
        # 2. 기본 클래스 인스턴스 생성 테스트
        config = UnifiedSystemConfig()
        system = UnifiedAlphaSystem(config)
        
        logger.info("✅ 통합 시스템 인스턴스 생성 성공")
        
        # 3. 데이터베이스 연결 테스트
        db_stats = system.database.get_database_stats()
        logger.info(f"✅ 데이터베이스 연결 성공: {db_stats}")
        
        # 4. 팩터 엔진 테스트
        available_factors = system.factor_engine.get_available_factors()
        logger.info(f"✅ 팩터 엔진 테스트 성공: {len(available_factors)} 카테고리")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 시스템 구성요소 테스트 실패: {str(e)}")
        return False

def run_quick_test():
    """빠른 기능 테스트"""
    try:
        logger.info("🧪 빠른 기능 테스트 시작...")
        
        from unified_alpha_system import UnifiedAlphaSystem, UnifiedSystemConfig
        import pandas as pd
        import numpy as np
        
        # 테스트용 설정
        config = UnifiedSystemConfig()
        system = UnifiedAlphaSystem(config)
        
        # 테스트용 가짜 데이터 생성
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        test_data = {
            'prices': pd.DataFrame({
                'AAPL': np.random.randn(len(dates)).cumsum() + 100,
                'MSFT': np.random.randn(len(dates)).cumsum() + 80,
                'GOOGL': np.random.randn(len(dates)).cumsum() + 120
            }, index=dates),
            'volumes': pd.DataFrame({
                'AAPL': np.random.randint(1000000, 10000000, len(dates)),
                'MSFT': np.random.randint(1000000, 10000000, len(dates)),
                'GOOGL': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
        }
        
        test_data['returns'] = test_data['prices'].pct_change()
        
        # Z-Score 팩터 계산 테스트
        zscore_factors = system.calculate_zscore_factors(test_data)
        logger.info(f"✅ Z-Score 팩터 계산 성공: {list(zscore_factors.keys())}")
        
        # 복합 점수 계산 테스트
        if zscore_factors:
            composite_score = system.calculate_composite_score(zscore_factors)
            logger.info(f"✅ 복합 점수 계산 성공: {len(composite_score)} 종목")
        
        # 포트폴리오 후보 선별 테스트
        candidates = system.select_portfolio_candidates(composite_score)
        logger.info(f"✅ 포트폴리오 후보 선별 성공")
        
        logger.info("🎉 모든 기능 테스트 성공!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 기능 테스트 실패: {str(e)}")
        return False

def launch_app():
    """통합 앱 실행"""
    try:
        logger.info("🚀 통합 알파 팩터 시스템 실행 중...")
        
        # Streamlit 앱 실행
        subprocess.run([
            sys.executable, 
            "-m", "streamlit", "run", 
            "unified_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--theme.base", "light"
        ])
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 앱이 종료되었습니다.")
    except Exception as e:
        logger.error(f"앱 실행 오류: {str(e)}")

def main():
    """메인 함수"""
    print("="*60)
    print("🚀 통합 알파 팩터 포트폴리오 시스템")
    print("="*60)
    
    # 1. 의존성 확인
    print("\n1️⃣ 의존성 확인 중...")
    if not check_dependencies():
        print("❌ 의존성 확인 실패. 설치 후 다시 실행하세요.")
        return
    
    # 2. 파일 구조 확인
    print("\n2️⃣ 파일 구조 확인 중...")
    if not check_file_structure():
        print("❌ 필수 파일이 누락되었습니다.")
        return
    
    # 3. 시스템 구성요소 테스트
    print("\n3️⃣ 시스템 구성요소 테스트 중...")
    if not test_system_components():
        print("❌ 시스템 구성요소 테스트 실패.")
        return
    
    # 4. 빠른 기능 테스트
    print("\n4️⃣ 빠른 기능 테스트 중...")
    if not run_quick_test():
        print("❌ 기능 테스트 실패.")
        return
    
    print("\n✅ 모든 검증 완료!")
    print("\n🎉 통합 시스템이 준비되었습니다!")
    
    # 5. 앱 실행 여부 확인
    print("\n" + "="*60)
    response = input("🚀 통합 앱을 실행하시겠습니까? (y/n): ").strip().lower()
    
    if response in ['y', 'yes', 'ㅇ']:
        print("\n📱 브라우저에서 http://localhost:8501 을 열어주세요.")
        print("🛑 종료하려면 Ctrl+C를 누르세요.")
        launch_app()
    else:
        print("\n✋ 수동 실행을 원하시면 다음 명령어를 사용하세요:")
        print("   streamlit run unified_app.py")

if __name__ == "__main__":
    main()