from typing import Dict, List, Optional, Any
import pandas as pd
import requests
import structlog
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import numpy as np
import time

logger = structlog.get_logger()


class FinancialMetrics(BaseModel):
    """Validated financial metrics for a company."""
    
    ticker: str
    company_name: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    
    @validator('*', pre=True)
    def replace_inf_with_none(cls, v):
        """Replace infinite values with None."""
        if isinstance(v, float) and np.isinf(v):
            return None
        return v


class FinancialDataCollector:
    """Collects and processes financial data using Alpha Vantage API."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.api_key = self._get_api_key()
        self.base_url = "https://www.alphavantage.co/query"
        
    def _get_api_key(self) -> str:
        """Get Alpha Vantage API key from config."""
        from src.utils.config import config
        api_key = getattr(config, 'alpha_vantage_api_key', None)
        if not api_key:
            # Try environment variable
            import os
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                raise ValueError("Alpha Vantage API key required. Set ALPHA_VANTAGE_API_KEY in .env file")
        return api_key
        
    def get_company_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch comprehensive financial data for a company using Alpha Vantage.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing all financial data
        """
        logger.info("Fetching financial data from Alpha Vantage", ticker=ticker)
        
        try:
            # Get company overview
            overview = self._get_company_overview(ticker)
            
            # Get historical prices
            historical_prices = self._get_historical_prices(ticker)
            
            # Get income statement
            income_statement = self._get_income_statement(ticker)
            
            # Get balance sheet
            balance_sheet = self._get_balance_sheet(ticker)
            
            data = {
                "info": overview,
                "historical_prices": historical_prices,
                "financials": income_statement,
                "balance_sheet": balance_sheet,
                "cash_flow": pd.DataFrame(),  # Alpha Vantage cash flow requires separate call
                "metrics": self._calculate_metrics_from_overview(overview)
            }
            
            self.cache[ticker] = data
            logger.info("Successfully fetched financial data from Alpha Vantage", ticker=ticker)
            return data
            
        except Exception as e:
            logger.error("Error fetching financial data", ticker=ticker, error=str(e))
            raise
    
    def _get_company_overview(self, ticker: str) -> Dict:
        """Get company overview from Alpha Vantage."""
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        
        if 'Note' in data:
            logger.warning("Alpha Vantage rate limit warning", message=data['Note'])
            time.sleep(12)  # Wait 12 seconds for rate limit
            return self._get_company_overview(ticker)  # Retry
        
        return data
    
    def _get_historical_prices(self, ticker: str) -> pd.DataFrame:
        """Get historical price data from Alpha Vantage."""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': 'full',
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'Error Message' in data:
            logger.warning("Could not fetch historical prices", error=data['Error Message'])
            return pd.DataFrame()
        
        if 'Note' in data:
            logger.warning("Alpha Vantage rate limit warning", message=data['Note'])
            time.sleep(12)
            return self._get_historical_prices(ticker)
        
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(time_series).T
        df.index = pd.to_datetime(df.index)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date ascending
        df = df.sort_index()
        
        # Limit to last 5 years for performance
        df = df.last('5Y')
        
        return df
    
    def _get_income_statement(self, ticker: str) -> pd.DataFrame:
        """Get income statement from Alpha Vantage."""
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'Error Message' in data:
            logger.warning("Could not fetch income statement", error=data['Error Message'])
            return pd.DataFrame()
        
        if 'Note' in data:
            time.sleep(12)
            return self._get_income_statement(ticker)
        
        annual_reports = data.get('annualReports', [])
        if not annual_reports:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(annual_reports)
        df.set_index('fiscalDateEnding', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # Convert numeric columns
        numeric_cols = ['totalRevenue', 'netIncome', 'grossProfit', 'operatingIncome']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _get_balance_sheet(self, ticker: str) -> pd.DataFrame:
        """Get balance sheet from Alpha Vantage."""
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'Error Message' in data:
            logger.warning("Could not fetch balance sheet", error=data['Error Message'])
            return pd.DataFrame()
        
        if 'Note' in data:
            time.sleep(12)
            return self._get_balance_sheet(ticker)
        
        annual_reports = data.get('annualReports', [])
        if not annual_reports:
            return pd.DataFrame()
        
        df = pd.DataFrame(annual_reports)
        df.set_index('fiscalDateEnding', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # Convert numeric columns
        numeric_cols = ['totalAssets', 'totalLiabilities', 'totalShareholderEquity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _calculate_metrics_from_overview(self, overview: Dict) -> FinancialMetrics:
        """Calculate standardized financial metrics from Alpha Vantage overview."""
        
        def safe_float(value):
            """Safely convert string to float, return None if invalid."""
            if value is None or value == 'None' or value == '-':
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        metrics = FinancialMetrics(
            ticker=overview.get('Symbol', ''),
            company_name=overview.get('Name', ''),
            market_cap=safe_float(overview.get('MarketCapitalization')),
            pe_ratio=safe_float(overview.get('PERatio')),
            forward_pe=safe_float(overview.get('ForwardPE')),
            peg_ratio=safe_float(overview.get('PEGRatio')),
            price_to_book=safe_float(overview.get('PriceToBookRatio')),
            debt_to_equity=safe_float(overview.get('DebtToEquityRatio')),
            roe=safe_float(overview.get('ReturnOnEquityTTM')),
            roa=safe_float(overview.get('ReturnOnAssetsTTM')),
            profit_margin=safe_float(overview.get('ProfitMargin')),
            revenue_growth=safe_float(overview.get('QuarterlyRevenueGrowthYOY')),
            earnings_growth=safe_float(overview.get('QuarterlyEarningsGrowthYOY')),
            beta=safe_float(overview.get('Beta')),
            dividend_yield=safe_float(overview.get('DividendYield'))
        )
        
        return metrics
    
    def get_peer_comparison(self, tickers: List[str]) -> pd.DataFrame:
        """
        Compare financial metrics across multiple companies.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with comparative metrics
        """
        logger.info("Performing peer comparison", tickers=tickers)
        
        metrics_list = []
        for ticker in tickers:
            try:
                data = self.get_company_data(ticker)
                metrics = data['metrics'].model_dump()
                metrics_list.append(metrics)
                time.sleep(12)  # Rate limiting for Alpha Vantage
            except Exception as e:
                logger.error("Error fetching peer data", ticker=ticker, error=str(e))
        
        df = pd.DataFrame(metrics_list)
        df.set_index('ticker', inplace=True)
        
        # Calculate percentile ranks for each metric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[f'{col}_percentile'] = df[col].rank(pct=True) * 100
        
        return df
    
    def get_historical_fundamentals(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """
        Get historical fundamental data over multiple years using Alpha Vantage.
        
        Args:
            ticker: Stock ticker symbol
            years: Number of years of history
            
        Returns:
            DataFrame with historical fundamentals
        """
        try:
            # Get income statement and balance sheet
            income_statement = self._get_income_statement(ticker)
            time.sleep(12)  # Rate limiting
            balance_sheet = self._get_balance_sheet(ticker)
            
            # Combine relevant metrics
            historical_data = pd.DataFrame()
            
            if not income_statement.empty:
                historical_data['Revenue'] = income_statement.get('totalRevenue')
                historical_data['Net Income'] = income_statement.get('netIncome')
                historical_data['Gross Profit'] = income_statement.get('grossProfit')
            
            if not balance_sheet.empty:
                historical_data['Total Assets'] = balance_sheet.get('totalAssets')
                historical_data['Total Liabilities'] = balance_sheet.get('totalLiabilities')
                historical_data['Shareholder Equity'] = balance_sheet.get('totalShareholderEquity')
            
            # Calculate growth rates
            for col in ['Revenue', 'Net Income']:
                if col in historical_data.columns:
                    historical_data[f'{col} Growth'] = historical_data[col].pct_change()
            
            return historical_data.head(years)
            
        except Exception as e:
            logger.error("Error fetching historical fundamentals", ticker=ticker, error=str(e))
            return pd.DataFrame()