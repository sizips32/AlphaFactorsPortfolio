#!/usr/bin/env python3
"""
Alpha Factor Generator - 모든 팩터 점검 테스트
전체 팩터 라이브러리의 정상 작동 여부를 확인합니다.
"""

import pandas as pd
import numpy as np
import logging
from alpha_factor_library import EnhancedFactorLibrary
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data():
    """테스트용 주식 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    # 가격 데이터 (랜덤 워크)
    prices = pd.DataFrame(index=dates, columns=symbols)
    for symbol in symbols:
        initial_price = np.random.uniform(100, 500)
        returns = np.random.normal(0.001, 0.02, len(dates))
        price_series = [initial_price]
        for r in returns[1:]:
            price_series.append(price_series[-1] * (1 + r))
        prices[symbol] = price_series
    
    # 거래량 데이터
    volumes = pd.DataFrame(index=dates, columns=symbols)
    for symbol in symbols:
        base_volume = np.random.uniform(1000000, 10000000)
        volume_changes = np.random.normal(1, 0.3, len(dates))
        volumes[symbol] = base_volume * volume_changes
    
    # 수익률 데이터
    returns = prices.pct_change().fillna(0)
    
    # 기타 데이터 (펀더멘털)
    market_caps = prices * volumes / 1000  # 간단한 시가총액 계산
    book_values = prices * 0.5  # 간단한 장부가
    earnings = prices * 0.1  # 간단한 수익
    
    return {
        'prices': prices,
        'volumes': volumes,
        'returns': returns,
        'market_caps': market_caps,
        'book_values': book_values,
        'earnings': earnings,
        'net_income': earnings,
        'shareholders_equity': book_values
    }

def test_all_factors():
    """모든 팩터 테스트 실행"""
    print("🔍 Alpha Factor Generator - 전체 팩터 점검 시작")
    print("="*60)
    
    # 팩터 라이브러리 초기화
    factory = EnhancedFactorLibrary()
    
    # 테스트 데이터 생성
    print("📊 테스트 데이터 생성 중...")
    data = generate_test_data()
    print(f"✅ 데이터 생성 완료: {len(data['prices'])}일, {len(data['prices'].columns)}개 종목")
    
    # 사용 가능한 팩터 목록
    available_factors = factory.get_available_factors()
    
    # 각 카테고리별 테스트 결과
    test_results = {}
    total_factors = 0
    successful_factors = 0
    
    print("\n🧪 팩터별 테스트 실행:")
    print("-"*60)
    
    for category, factors in available_factors.items():
        print(f"\n📁 {category.upper()} 카테고리 ({len(factors)}개 팩터)")
        category_results = []
        
        for factor_name in factors:
            total_factors += 1
            try:
                # 팩터 계산 시도
                result = factory.calculate_factor(category, factor_name, data)
                
                if isinstance(result, (pd.DataFrame, pd.Series)) and not result.empty:
                    # 결과 검증
                    if isinstance(result, pd.DataFrame):
                        shape = result.shape
                        has_data = not result.isna().all().all()
                    else:
                        shape = (len(result),)
                        has_data = not result.isna().all()
                    
                    if has_data:
                        print(f"  ✅ {factor_name}: 정상 ({shape})")
                        category_results.append((factor_name, "성공", shape))
                        successful_factors += 1
                    else:
                        print(f"  ⚠️  {factor_name}: 빈 데이터")
                        category_results.append((factor_name, "빈 데이터", shape))
                else:
                    print(f"  ❌ {factor_name}: 빈 결과")
                    category_results.append((factor_name, "빈 결과", None))
                    
            except Exception as e:
                print(f"  ❌ {factor_name}: 오류 - {str(e)[:50]}...")
                category_results.append((factor_name, f"오류: {str(e)[:30]}", None))
        
        test_results[category] = category_results
    
    # 종합 결과 출력
    print("\n"+"="*60)
    print("📋 테스트 종합 결과")
    print("="*60)
    
    for category, results in test_results.items():
        success_count = sum(1 for _, status, _ in results if status == "성공")
        total_count = len(results)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        print(f"{category:20} : {success_count:2}/{total_count:2} ({success_rate:5.1f}%)")
        
        # 실패한 팩터들 표시
        failed_factors = [(name, status) for name, status, _ in results if status != "성공"]
        if failed_factors:
            for name, status in failed_factors:
                print(f"  ⚠️  {name}: {status}")
    
    overall_success_rate = (successful_factors / total_factors * 100) if total_factors > 0 else 0
    
    print("\n" + "="*60)
    print(f"🎯 전체 성공률: {successful_factors}/{total_factors} ({overall_success_rate:.1f}%)")
    
    if overall_success_rate >= 80:
        print("🎉 팩터 라이브러리가 우수한 상태입니다!")
    elif overall_success_rate >= 60:
        print("✅ 팩터 라이브러리가 양호한 상태입니다.")
    else:
        print("⚠️  일부 팩터에 문제가 있습니다. 수정이 필요합니다.")
    
    return test_results

def test_factor_integration():
    """팩터 통합 기능 테스트"""
    print("\n🔗 팩터 통합 기능 테스트")
    print("-"*40)
    
    factory = EnhancedFactorLibrary()
    data = generate_test_data()
    
    # 여러 팩터 동시 계산 테스트
    test_factors = [
        ('technical', 'momentum'),
        ('advanced_technical', 'rsi'),
        ('fundamental', 'valuation'),
        ('machine_learning', 'pca'),
        ('risk', 'beta')
    ]
    
    combined_results = {}
    
    for category, factor_name in test_factors:
        try:
            result = factory.calculate_factor(category, factor_name, data)
            if isinstance(result, (pd.DataFrame, pd.Series)) and not result.empty:
                combined_results[f"{category}_{factor_name}"] = result
                print(f"✅ {category}.{factor_name}: 정상")
            else:
                print(f"❌ {category}.{factor_name}: 빈 결과")
        except Exception as e:
            print(f"❌ {category}.{factor_name}: {str(e)[:50]}...")
    
    print(f"\n통합 테스트 결과: {len(combined_results)}/{len(test_factors)}개 팩터 성공")
    return combined_results

if __name__ == "__main__":
    try:
        # 전체 팩터 테스트
        test_results = test_all_factors()
        
        # 통합 기능 테스트
        integration_results = test_factor_integration()
        
        print("\n🏁 모든 테스트 완료!")
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()