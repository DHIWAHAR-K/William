from typing import Dict, List, Optional, Any
import pandas as pd
import yfinance as yf
import structlog
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import numpy as np

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
    """Collects and processes financial data from various sources."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        
    def get_company_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch comprehensive financial data for a company.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing all financial data
        """
        logger.info("Fetching financial data", ticker=ticker)
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get historical data
            hist_data = stock.history(period="5y")
            
            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            data = {
                "info": info,
                "historical_prices": hist_data,
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
                "metrics": self._calculate_metrics(info, financials, balance_sheet, cash_flow)
            }
            
            self.cache[ticker] = data
            logger.info("Successfully fetched financial data", ticker=ticker)
            return data
            
        except Exception as e:
            logger.error("Error fetching financial data", ticker=ticker, error=str(e))
            raise
    
    def _calculate_metrics(self, info: Dict, financials: pd.DataFrame, 
                          balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> FinancialMetrics:
        """Calculate standardized financial metrics."""
        
        metrics = FinancialMetrics(
            ticker=info.get('symbol', ''),
            company_name=info.get('longName', ''),
            market_cap=info.get('marketCap'),
            pe_ratio=info.get('trailingPE'),
            forward_pe=info.get('forwardPE'),
            peg_ratio=info.get('pegRatio'),
            price_to_book=info.get('priceToBook'),
            debt_to_equity=info.get('debtToEquity'),
            roe=info.get('returnOnEquity'),
            roa=info.get('returnOnAssets'),
            profit_margin=info.get('profitMargins'),
            beta=info.get('beta'),
            dividend_yield=info.get('dividendYield')
        )
        
        # Calculate additional metrics from statements
        if not financials.empty and not balance_sheet.empty:
            try:
                # Revenue growth (YoY)
                if 'Total Revenue' in financials.index:
                    revenues = financials.loc['Total Revenue']
                    if len(revenues) >= 2:
                        metrics.revenue_growth = (revenues.iloc[0] - revenues.iloc[1]) / abs(revenues.iloc[1])
                
                # Free cash flow
                if not cash_flow.empty and 'Free Cash Flow' in cash_flow.index:
                    metrics.free_cash_flow = cash_flow.loc['Free Cash Flow'].iloc[0]
                    
            except Exception as e:
                logger.warning("Error calculating additional metrics", error=str(e))
        
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
                metrics = data['metrics'].dict()
                metrics_list.append(metrics)
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
        Get historical fundamental data over multiple years.
        
        Args:
            ticker: Stock ticker symbol
            years: Number of years of history
            
        Returns:
            DataFrame with historical fundamentals
        """
        stock = yf.Ticker(ticker)
        
        # Get annual financials
        financials = stock.financials.T  # Transpose to have years as rows
        balance_sheet = stock.balance_sheet.T
        cash_flow = stock.cashflow.T
        
        # Combine relevant metrics
        historical_data = pd.DataFrame(index=financials.index)
        
        if 'Total Revenue' in financials.columns:
            historical_data['Revenue'] = financials['Total Revenue']
        if 'Net Income' in financials.columns:
            historical_data['Net Income'] = financials['Net Income']
        if 'Total Assets' in balance_sheet.columns:
            historical_data['Total Assets'] = balance_sheet['Total Assets']
        if 'Total Debt' in balance_sheet.columns:
            historical_data['Total Debt'] = balance_sheet['Total Debt']
        if 'Free Cash Flow' in cash_flow.columns:
            historical_data['Free Cash Flow'] = cash_flow['Free Cash Flow']
        
        # Calculate growth rates
        for col in ['Revenue', 'Net Income']:
            if col in historical_data.columns:
                historical_data[f'{col} Growth'] = historical_data[col].pct_change()
        
        return historical_data.head(years)