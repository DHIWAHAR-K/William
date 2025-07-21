"""
Company Fundamental Analysis System

A production-grade system that combines financial data APIs, LLM analysis, 
and machine learning models to perform comprehensive company valuation.
"""

from typing import List, Dict
import pandas as pd
import structlog
from pathlib import Path
import json
from datetime import datetime

from src.data_collection.financial_data import FinancialDataCollector
from src.analysis.llm_analyzer import CompanyLLMAnalyzer
from src.models.valuation_model import FundamentalValuationModel
from src.utils.config import config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class FundamentalAnalysisSystem:
    """Main system orchestrator for fundamental analysis."""
    
    def __init__(self):
        logger.info("Initializing Fundamental Analysis System")
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid configuration. Please check API keys.")
        
        # Initialize components
        self.data_collector = FinancialDataCollector()
        self.llm_analyzer = CompanyLLMAnalyzer()
        self.valuation_model = FundamentalValuationModel()
        
        # Storage
        self.analysis_results = {}
        
    def analyze_company(self, ticker: str) -> Dict:
        """
        Perform complete fundamental analysis for a single company.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting company analysis", ticker=ticker)
        
        try:
            # Step 1: Collect financial data
            logger.info("Collecting financial data", ticker=ticker)
            financial_data = self.data_collector.get_company_data(ticker)
            
            # Step 2: LLM Analysis
            logger.info("Performing LLM analysis", ticker=ticker)
            llm_analysis = self.llm_analyzer.analyze_with_verification(
                ticker, 
                financial_data,
                additional_sources=[f"Recent financial performance shows strong fundamentals"]
            )
            
            # Step 3: Prepare results
            results = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'financial_metrics': financial_data['metrics'].model_dump(),
                'llm_analysis': llm_analysis.model_dump(),
                'data_quality': {
                    'has_financials': bool(financial_data.get('financials') is not None),
                    'has_balance_sheet': bool(financial_data.get('balance_sheet') is not None),
                    'confidence_score': llm_analysis.confidence_score
                }
            }
            
            # Store results
            self.analysis_results[ticker] = results
            
            logger.info("Company analysis complete", ticker=ticker)
            return results
            
        except Exception as e:
            logger.error("Analysis failed", ticker=ticker, error=str(e))
            raise
    
    def analyze_portfolio(self, tickers: List[str]) -> pd.DataFrame:
        """
        Analyze multiple companies and create comparative analysis.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with comparative analysis
        """
        logger.info("Starting portfolio analysis", tickers=tickers)
        
        # Collect data for all companies
        all_financial_data = {}
        for ticker in tickers:
            try:
                all_financial_data[ticker] = self.data_collector.get_company_data(ticker)
            except Exception as e:
                logger.error(f"Failed to get data for {ticker}", error=str(e))
        
        # Perform peer comparison
        comparison_df = self.data_collector.get_peer_comparison(list(all_financial_data.keys()))
        
        # Add LLM insights
        llm_insights = []
        for ticker in all_financial_data.keys():
            try:
                analysis = self.llm_analyzer.analyze_with_verification(
                    ticker, 
                    all_financial_data[ticker]
                )
                llm_insights.append({
                    'ticker': ticker,
                    'business_description': analysis.business_description,
                    'competitive_advantages': ', '.join(analysis.competitive_advantages[:3]),
                    'growth_drivers': ', '.join(analysis.growth_drivers[:3]),
                    'confidence': analysis.confidence_score
                })
            except Exception as e:
                logger.error(f"LLM analysis failed for {ticker}", error=str(e))
        
        llm_df = pd.DataFrame(llm_insights).set_index('ticker')
        
        # Combine results
        final_df = comparison_df.join(llm_df, how='left')
        
        return final_df
    
    def train_valuation_model(self, training_tickers: List[str]) -> Dict:
        """
        Train the ML valuation model on a set of companies.
        
        Args:
            training_tickers: List of tickers for training data
            
        Returns:
            Model performance metrics
        """
        logger.info("Training valuation model", n_companies=len(training_tickers))
        
        # Collect training data
        training_data = []
        for ticker in training_tickers:
            try:
                data = self.data_collector.get_company_data(ticker)
                training_data.append(data)
            except Exception as e:
                logger.warning(f"Skipping {ticker} due to error", error=str(e))
        
        if len(training_data) < 10:
            raise ValueError("Insufficient training data. Need at least 10 companies.")
        
        # Prepare features and train
        X, y = self.valuation_model.prepare_features(training_data)
        results = self.valuation_model.train_and_evaluate(X, y)
        
        # Save model
        model_path = config.output_dir / "valuation_model.pkl"
        self.valuation_model.save_model(model_path)
        
        return results
    
    def predict_valuation(self, ticker: str) -> Dict:
        """
        Predict valuation for a company using the trained model.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Valuation predictions
        """
        if self.valuation_model.best_model is None:
            # Try to load saved model
            model_path = config.output_dir / "valuation_model.pkl"
            if model_path.exists():
                self.valuation_model.load_model(model_path)
            else:
                raise ValueError("No trained model available")
        
        # Get company data
        company_data = self.data_collector.get_company_data(ticker)
        
        # Make prediction
        valuation = self.valuation_model.predict_valuation(company_data)
        
        return valuation
    
    def generate_report(self, ticker: str, output_format: str = "json") -> Path:
        """
        Generate a comprehensive report for a company.
        
        Args:
            ticker: Stock ticker symbol
            output_format: Format for output (json, csv, html)
            
        Returns:
            Path to generated report
        """
        logger.info("Generating report", ticker=ticker, format=output_format)
        
        # Ensure we have analysis results
        if ticker not in self.analysis_results:
            self.analyze_company(ticker)
        
        results = self.analysis_results[ticker]
        
        # Add valuation if model is available
        try:
            valuation = self.predict_valuation(ticker)
            results['ml_valuation'] = valuation
        except Exception as e:
            logger.warning("Could not add ML valuation", error=str(e))
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_analysis_{timestamp}.{output_format}"
        output_path = config.output_dir / filename
        
        if output_format == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif output_format == "csv":
            # Flatten the nested structure for CSV
            flat_data = self._flatten_dict(results)
            df = pd.DataFrame([flat_data])
            df.to_csv(output_path, index=False)
        
        logger.info("Report generated", path=str(output_path))
        return output_path
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, ', '.join(map(str, v))))
            else:
                items.append((new_key, v))
        return dict(items)


def main():
    """Example usage of the Fundamental Analysis System."""
    
    # Initialize system
    system = FundamentalAnalysisSystem()
    
    # Example 1: Analyze a single company
    print("\n=== Analyzing Apple (AAPL) ===")
    apple_analysis = system.analyze_company("AAPL")
    
    print(f"\nCompany: {apple_analysis['llm_analysis']['company_name']}")
    print(f"Business: {apple_analysis['llm_analysis']['business_description']}")
    print(f"\nKey Metrics:")
    metrics = apple_analysis['financial_metrics']
    print(f"  P/E Ratio: {metrics.get('pe_ratio', 'N/A')}")
    print(f"  ROE: {metrics.get('roe', 'N/A')}")
    print(f"  Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}")
    
    # Example 2: Compare multiple companies
    print("\n=== Comparing Tech Companies ===")
    tech_companies = ["AAPL", "MSFT", "GOOGL"]
    comparison = system.analyze_portfolio(tech_companies)
    print("\nComparison Results:")
    print(comparison[['pe_ratio', 'roe', 'revenue_growth', 'competitive_advantages']].head())
    
    # Example 3: Train valuation model
    print("\n=== Training Valuation Model ===")
    training_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", 
                       "NVDA", "TSLA", "JPM", "V", "JNJ", 
                       "WMT", "PG", "XOM", "CVX", "ABBV"]
    
    model_results = system.train_valuation_model(training_tickers)
    
    print("\nModel Performance:")
    for model_name, metrics in model_results.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  MAPE: {metrics['mape']:.2%}")
    
    # Example 4: Predict valuation
    print("\n=== Predicting Valuation for Tesla ===")
    tesla_valuation = system.predict_valuation("TSLA")
    print(f"Current P/E: {tesla_valuation['current_pe']:.2f}")
    print(f"Predicted P/E: {tesla_valuation['predicted_pe']:.2f}")
    print(f"Upside Potential: {tesla_valuation['upside_potential']:.2%}")
    print(f"Recommendation: {tesla_valuation['recommendation']}")
    
    # Example 5: Generate report
    print("\n=== Generating Reports ===")
    report_path = system.generate_report("AAPL", "json")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()