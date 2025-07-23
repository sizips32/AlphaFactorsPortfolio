# Alpha Factor Generator

## Overview

Alpha Factor Generator is a comprehensive web application for creating, testing, and optimizing alpha factors for financial investment portfolios. The system provides a complete pipeline from factor generation through backtesting to portfolio optimization, built on a Streamlit web interface with SQLite database integration.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Core Framework**: Python-based modular architecture
- **Web Framework**: Streamlit for interactive web interface
- **Database**: SQLite for local data persistence
- **Data Processing**: Pandas and NumPy for financial data manipulation
- **Optimization**: SciPy and CVXPY for portfolio optimization algorithms
- **Machine Learning**: Scikit-learn and XGBoost for factor generation

### Frontend Architecture
- **Interface**: Streamlit-based single-page application
- **Visualization**: Plotly for interactive charts and graphs
- **Layout**: Wide layout with sidebar navigation
- **Styling**: Custom CSS for enhanced user experience

### Data Storage
- **Primary Database**: SQLite (`alpha_factors.db`)
- **Schema**: Tables for factor definitions, factor values, backtest results, and portfolio configurations
- **Data Sources**: Yahoo Finance integration via yfinance library
- **Caching**: Built-in data caching for performance optimization

## Key Components

### 1. Factor Generation Engine (`alpha_factor_library.py`)
- **Technical Factors**: Momentum, mean reversion, volatility, and volume-based indicators
- **Fundamental Factors**: Valuation, profitability, growth, and financial health metrics
- **Machine Learning Factors**: Random Forest, XGBoost, and PCA-based factors
- **Risk Factors**: Volatility, beta, and risk-adjusted return calculations
- **Validation**: Built-in factor validation and quality checks

### 2. Database Management (`database.py`)
- **Connection Management**: SQLite connection handling with automatic initialization
- **Schema Management**: Automated table creation and data persistence
- **Data Operations**: CRUD operations for factors, backtests, and portfolios
- **Error Handling**: Robust error handling with logging

### 3. Backtesting Engine (`backtesting_engine.py`)
- **Configuration**: Flexible backtesting parameters (rebalancing frequency, transaction costs, etc.)
- **Performance Metrics**: Comprehensive performance analysis including Sharpe ratio, maximum drawdown
- **Visualization**: Interactive performance charts and analytics
- **Risk Analysis**: Value-at-Risk, tracking error, and risk attribution

### 4. Portfolio Optimizer (`portfolio_optimizer.py`)
- **Optimization Methods**: Mean-variance optimization, risk parity, and custom objectives
- **Constraints**: Position size limits, sector constraints, turnover limits
- **Risk Models**: Covariance estimation with shrinkage methods
- **Integration**: Seamless integration with factor scores and backtesting

### 5. Web Application (`app.py`)
- **Multi-page Interface**: Factor generation, backtesting, portfolio optimization, and analysis pages
- **Interactive Controls**: Dynamic parameter adjustment with real-time updates
- **Data Upload**: Support for custom data uploads and external data sources
- **Export Functionality**: Results export in multiple formats

## Data Flow

1. **Data Ingestion**: Market data fetched from Yahoo Finance or uploaded by users
2. **Factor Generation**: Raw data processed through factor calculation engines
3. **Data Validation**: Automatic data type conversion and missing value handling
4. **Database Storage**: Calculated factors stored in SQLite with metadata
5. **Backtesting**: Historical performance simulation using stored factors
6. **Optimization**: Portfolio weight calculation based on factor scores and constraints
7. **Visualization**: Results presented through interactive Plotly charts
8. **Export**: Final results exported for external use

## External Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing and optimization
- **scikit-learn**: Machine learning algorithms
- **plotly**: Interactive visualization
- **streamlit**: Web application framework

### Financial Libraries
- **yfinance**: Yahoo Finance data integration
- **cvxpy**: Convex optimization
- **xgboost**: Gradient boosting algorithms

### Database
- **sqlite3**: Built-in Python SQLite interface
- **No external database server required**

### Optional Dependencies
- **tensorflow**: Deep learning models (optional)
- **talib**: Technical analysis library (optional)

## Deployment Strategy

### Local Development
- **Setup**: Simple pip install of requirements
- **Execution**: `streamlit run app.py`
- **Database**: Automatic SQLite database creation
- **Configuration**: Environment-based configuration support

### Production Deployment
- **Platform**: Compatible with Replit, Heroku, AWS, or local servers
- **Database**: SQLite for development, easily upgradeable to PostgreSQL
- **Scaling**: Modular architecture supports horizontal scaling
- **Monitoring**: Built-in logging and error tracking

### Key Architectural Decisions

1. **SQLite Choice**: Selected for simplicity and zero-configuration deployment, easily upgradeable to PostgreSQL for production scaling

2. **Streamlit Framework**: Chosen for rapid development and built-in interactivity, suitable for financial analysis workflows

3. **Modular Design**: Each component (factors, backtesting, optimization) is independent, allowing for easy maintenance and feature additions

4. **Data Type Safety**: Implemented robust data validation functions (`ensure_numeric_series`, `ensure_numeric_dataframe`) to prevent common pandas dtype errors

5. **Error Handling**: Comprehensive error handling throughout the application with graceful degradation and user feedback

6. **Performance Optimization**: Caching strategies and efficient data processing to handle large financial datasets

## Recent Changes (July 23, 2025)

### GitHub Integration Completed
- Successfully uploaded all 7 core files to GitHub repository: https://github.com/sizips32/AlphaFactorsPortfolio
- Verified file integrity and module loading compatibility
- All files synchronized between Replit and GitHub environments

### Factor Library Testing and Fixes
- **Complete Factor Testing**: All 26 factors across 8 categories tested and validated (100% success rate)
- **Parameter Compatibility**: Fixed RSI and other advanced technical factors parameter mismatch (`window` vs `period`)
- **Machine Learning Factors**: Resolved data type handling issues in Random Forest, PCA, and XGBoost factors
- **Data Validation**: Enhanced numeric data conversion and error handling throughout the factor library

### System Validation Results
- **Technical Factors**: 4/4 working (momentum, mean reversion, volatility, volume)
- **Advanced Technical**: 4/4 working (RSI, Bollinger bands, Z-score, correlation)
- **Fundamental**: 2/2 working (valuation, profitability)
- **Advanced Fundamental**: 3/3 working (PBR, PER, ROE)
- **Machine Learning**: 3/3 working (Random Forest, PCA, XGBoost)
- **Risk**: 2/2 working (beta, downside risk)
- **Advanced Math**: 5/5 working (Kalman filter, Hurst exponent, wavelet, regime detection, isolation forest)
- **Deep Learning**: 3/3 working (LSTM, attention, ensemble)

### Bug Fixes and Improvements (July 23, 2025)
- **Database Integration**: Fixed DatabaseManager.save_factor_values() parameter mismatch
- **Backtesting Engine**: Added missing scipy.stats import for correlation calculations
- **Data Validation**: Enhanced infinite value and NaN handling in factor calculations
- **Chart Rendering**: Resolved "Infinite extent for field" warnings by cleaning data before visualization
- **Error Handling**: Improved robustness of factor calculation and storage pipeline

The architecture prioritizes ease of use, rapid prototyping, and extensibility while maintaining professional-grade functionality for financial analysis and portfolio management.