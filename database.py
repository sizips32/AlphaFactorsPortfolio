"""
데이터베이스 연결 및 관리 모듈
Alpha Factor Generator용 SQLite 데이터베이스 스키마 및 연결 관리

작성자: AI Assistant
작성일: 2025년 7월 23일
버전: 1.0
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """데이터베이스 연결 및 관리 클래스"""
    
    def __init__(self, db_path: str = "alpha_factors.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 및 테이블 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 팩터 정의 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS factor_definitions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        category TEXT NOT NULL,
                        description TEXT,
                        parameters TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 팩터 값 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS factor_values (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        factor_id INTEGER,
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        value REAL,
                        FOREIGN KEY (factor_id) REFERENCES factor_definitions (id),
                        UNIQUE(factor_id, symbol, date)
                    )
                """)
                
                # 백테스팅 결과 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        factor_id INTEGER,
                        config TEXT,
                        performance_metrics TEXT,
                        portfolio_returns TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (factor_id) REFERENCES factor_definitions (id)
                    )
                """)
                
                # 포트폴리오 최적화 결과 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        factor_ids TEXT,
                        method TEXT NOT NULL,
                        weights TEXT NOT NULL,
                        expected_return REAL,
                        expected_volatility REAL,
                        sharpe_ratio REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 시장 데이터 캐시 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume INTEGER,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, date)
                    )
                """)
                
                conn.commit()
                logger.info("데이터베이스 초기화 완료")
                
        except Exception as e:
            logger.error(f"데이터베이스 초기화 오류: {str(e)}")
            raise
    
    def save_factor_definition(self, name: str, category: str, description: str = None, 
                             parameters: Dict = None) -> int:
        """팩터 정의 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                params_json = json.dumps(parameters) if parameters else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO factor_definitions 
                    (name, category, description, parameters)
                    VALUES (?, ?, ?, ?)
                """, (name, category, description, params_json))
                
                factor_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"팩터 정의 저장 완료: {name} (ID: {factor_id})")
                return factor_id
                
        except Exception as e:
            logger.error(f"팩터 정의 저장 오류: {str(e)}")
            raise
    
    def save_factor_values(self, factor_id: int, factor_data: pd.DataFrame):
        """팩터 값 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 기존 데이터 삭제
                cursor.execute("DELETE FROM factor_values WHERE factor_id = ?", (factor_id,))
                
                # 새 데이터 삽입
                for date, row in factor_data.iterrows():
                    for symbol in factor_data.columns:
                        value = row[symbol]
                        if pd.notna(value) and np.isfinite(value):
                            cursor.execute("""
                                INSERT INTO factor_values (factor_id, symbol, date, value)
                                VALUES (?, ?, ?, ?)
                            """, (factor_id, symbol, date.strftime('%Y-%m-%d'), float(value)))
                
                conn.commit()
                logger.info(f"팩터 값 저장 완료: Factor ID {factor_id}")
                
        except Exception as e:
            logger.error(f"팩터 값 저장 오류: {str(e)}")
            raise
    
    def load_factor_values(self, factor_id: int) -> Optional[pd.DataFrame]:
        """팩터 값 로드"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT symbol, date, value
                    FROM factor_values
                    WHERE factor_id = ?
                    ORDER BY date, symbol
                """
                
                df = pd.read_sql_query(query, conn, params=(factor_id,))
                
                if df.empty:
                    return None
                
                # 피벗 테이블로 변환
                factor_df = df.pivot(index='date', columns='symbol', values='value')
                factor_df.index = pd.to_datetime(factor_df.index)
                
                return factor_df
                
        except Exception as e:
            logger.error(f"팩터 값 로드 오류: {str(e)}")
            return None
    
    def save_backtest_results(self, factor_id: int, config: Dict, 
                            performance_metrics: Dict, portfolio_returns: pd.Series) -> int:
        """백테스팅 결과 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                config_json = json.dumps(config, default=str)
                metrics_json = json.dumps(performance_metrics, default=str)
                returns_json = portfolio_returns.to_json(date_format='iso')
                
                cursor.execute("""
                    INSERT INTO backtest_results 
                    (factor_id, config, performance_metrics, portfolio_returns)
                    VALUES (?, ?, ?, ?)
                """, (factor_id, config_json, metrics_json, returns_json))
                
                result_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"백테스팅 결과 저장 완료: Result ID {result_id}")
                return result_id
                
        except Exception as e:
            logger.error(f"백테스팅 결과 저장 오류: {str(e)}")
            raise
    
    def save_optimization_results(self, factor_ids: List[int], method: str, 
                                weights: pd.Series, expected_return: float,
                                expected_volatility: float, sharpe_ratio: float) -> int:
        """포트폴리오 최적화 결과 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                factor_ids_json = json.dumps(factor_ids)
                weights_json = weights.to_json()
                
                cursor.execute("""
                    INSERT INTO optimization_results 
                    (factor_ids, method, weights, expected_return, expected_volatility, sharpe_ratio)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (factor_ids_json, method, weights_json, expected_return, 
                      expected_volatility, sharpe_ratio))
                
                result_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"최적화 결과 저장 완료: Result ID {result_id}")
                return result_id
                
        except Exception as e:
            logger.error(f"최적화 결과 저장 오류: {str(e)}")
            raise
    
    def cache_market_data(self, data: pd.DataFrame, symbol: str):
        """시장 데이터 캐시"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 기존 데이터 삭제
                cursor.execute("DELETE FROM market_data WHERE symbol = ?", (symbol,))
                
                # 새 데이터 삽입
                for date, row in data.iterrows():
                    cursor.execute("""
                        INSERT INTO market_data 
                        (symbol, date, open_price, high_price, low_price, close_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, date.strftime('%Y-%m-%d'), 
                          row.get('Open'), row.get('High'), row.get('Low'), 
                          row.get('Close'), row.get('Volume')))
                
                conn.commit()
                logger.info(f"시장 데이터 캐시 완료: {symbol}")
                
        except Exception as e:
            logger.error(f"시장 데이터 캐시 오류: {str(e)}")
    
    def get_cached_market_data(self, symbol: str, start_date: str = None, 
                              end_date: str = None) -> Optional[pd.DataFrame]:
        """캐시된 시장 데이터 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT date, open_price, high_price, low_price, close_price, volume
                    FROM market_data
                    WHERE symbol = ?
                """
                params = [symbol]
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY date"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return None
                
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                return df
                
        except Exception as e:
            logger.error(f"캐시된 시장 데이터 조회 오류: {str(e)}")
            return None
    
    def get_factor_list(self) -> List[Dict]:
        """저장된 팩터 목록 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT id, name, category, description, created_at
                    FROM factor_definitions
                    ORDER BY created_at DESC
                """
                
                df = pd.read_sql_query(query, conn)
                return df.to_dict('records')
                
        except Exception as e:
            logger.error(f"팩터 목록 조회 오류: {str(e)}")
            return []
    
    def get_backtest_history(self, factor_id: int = None) -> List[Dict]:
        """백테스팅 기록 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT br.id, br.factor_id, fd.name as factor_name, 
                           br.performance_metrics, br.created_at
                    FROM backtest_results br
                    JOIN factor_definitions fd ON br.factor_id = fd.id
                """
                params = []
                
                if factor_id:
                    query += " WHERE br.factor_id = ?"
                    params.append(factor_id)
                
                query += " ORDER BY br.created_at DESC"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                # JSON 문자열을 딕셔너리로 변환
                for idx, row in df.iterrows():
                    try:
                        df.at[idx, 'performance_metrics'] = json.loads(row['performance_metrics'])
                    except:
                        df.at[idx, 'performance_metrics'] = {}
                
                return df.to_dict('records')
                
        except Exception as e:
            logger.error(f"백테스팅 기록 조회 오류: {str(e)}")
            return []
    
    def cleanup_old_data(self, days: int = 30):
        """오래된 데이터 정리"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - pd.Timedelta(days=days)
                cutoff_str = cutoff_date.strftime('%Y-%m-%d')
                
                # 오래된 백테스팅 결과 삭제
                cursor.execute("""
                    DELETE FROM backtest_results 
                    WHERE created_at < ?
                """, (cutoff_str,))
                
                # 오래된 최적화 결과 삭제
                cursor.execute("""
                    DELETE FROM optimization_results 
                    WHERE created_at < ?
                """, (cutoff_str,))
                
                conn.commit()
                logger.info(f"{days}일 이전 데이터 정리 완료")
                
        except Exception as e:
            logger.error(f"데이터 정리 오류: {str(e)}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # 테이블별 레코드 수
                tables = ['factor_definitions', 'factor_values', 'backtest_results', 
                         'optimization_results', 'market_data']
                
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # 데이터베이스 크기
                cursor = conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                stats['database_size_mb'] = round(db_size / (1024 * 1024), 2)
                
                # 최근 활동
                cursor = conn.execute("""
                    SELECT MAX(created_at) FROM factor_definitions
                """)
                last_factor = cursor.fetchone()[0]
                stats['last_factor_created'] = last_factor
                
                cursor = conn.execute("""
                    SELECT MAX(created_at) FROM backtest_results
                """)
                last_backtest = cursor.fetchone()[0]
                stats['last_backtest'] = last_backtest
                
                return stats
                
        except Exception as e:
            logger.error(f"데이터베이스 통계 조회 오류: {str(e)}")
            return {}
    
    def export_data(self, table_name: str, output_format: str = 'csv') -> Optional[str]:
        """데이터 내보내기"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                
                if output_format == 'csv':
                    filename = f"{table_name}_export.csv"
                    df.to_csv(filename, index=False)
                elif output_format == 'json':
                    filename = f"{table_name}_export.json"
                    df.to_json(filename, orient='records', date_format='iso')
                else:
                    raise ValueError(f"지원하지 않는 형식: {output_format}")
                
                logger.info(f"데이터 내보내기 완료: {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"데이터 내보내기 오류: {str(e)}")
            return None
    
    def vacuum_database(self):
        """데이터베이스 최적화 (VACUUM)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
                logger.info("데이터베이스 최적화 완료")
        except Exception as e:
            logger.error(f"데이터베이스 최적화 오류: {str(e)}")
    
    def backup_database(self, backup_path: str = None):
        """데이터베이스 백업"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"alpha_factors_backup_{timestamp}.db"
            
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"데이터베이스 백업 완료: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"데이터베이스 백업 오류: {str(e)}")
            return None
    
    def get_market_data_count(self) -> int:
        """시장 데이터 개수 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM market_data")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"시장 데이터 개수 조회 오류: {str(e)}")
            return 0
