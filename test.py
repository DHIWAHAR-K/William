#!/usr/bin/env python3
"""
Test script for analyzing a single company with Alpha Vantage API.
This minimizes API calls to stay within the 25/day limit.
"""

from src.utils.config import config
from src.data_collection.financial_data import FinancialDataCollector
from src.analysis.llm_analyzer import CompanyLLMAnalyzer
import json
from datetime import datetime

def test_single_company(ticker: str = "MSFT"):
    """Test analyzing a single company to conserve API calls."""
    
    print("🚀 SINGLE COMPANY ANALYSIS")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using Alpha Vantage API (25 calls/day limit)")
    print(f"Target Company: {ticker}\n")
    
    if not config.validate():
        print("❌ Configuration invalid. Please check API keys in .env file.")
        return False
    
    if not config.alpha_vantage_api_key:
        print("❌ Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY in .env")
        return False
    
    try:
        # Initialize components
        print("📦 Initializing components...")
        data_collector = FinancialDataCollector()
        llm_analyzer = CompanyLLMAnalyzer()
        
        # Get financial data (uses ~4 API calls)
        print(f"\n📊 Fetching financial data for {ticker}...")
        print("   ⏳ This will use ~4 Alpha Vantage API calls")
        financial_data = data_collector.get_company_data(ticker)
        print(f"   ✅ Retrieved financial data successfully")
        
        # Display company info
        info = financial_data['info']
        print(f"\n🏢 COMPANY OVERVIEW")
        print(f"{'─' * 40}")
        print(f"Name: {info.get('Name', 'N/A')}")
        print(f"Symbol: {info.get('Symbol', 'N/A')}")
        print(f"Exchange: {info.get('Exchange', 'N/A')}")
        print(f"Industry: {info.get('Industry', 'N/A')}")
        print(f"Sector: {info.get('Sector', 'N/A')}")
        print(f"Description: {info.get('Description', 'N/A')[:200]}...")
        
        # Get LLM analysis
        print(f"\n🤖 Performing LLM analysis...")
        llm_analysis = llm_analyzer.analyze_with_verification(ticker, financial_data)
        print(f"   ✅ LLM analysis completed")
        
        # Display financial metrics
        print(f"\n💰 KEY FINANCIAL METRICS")
        print(f"{'─' * 40}")
        metrics = financial_data['metrics']
        print(f"Market Cap: ${metrics.market_cap:,.0f}" if metrics.market_cap else "Market Cap: N/A")
        print(f"P/E Ratio: {metrics.pe_ratio:.2f}" if metrics.pe_ratio else "P/E Ratio: N/A")
        print(f"Forward P/E: {metrics.forward_pe:.2f}" if metrics.forward_pe else "Forward P/E: N/A")
        print(f"PEG Ratio: {metrics.peg_ratio:.2f}" if metrics.peg_ratio else "PEG Ratio: N/A")
        print(f"ROE: {metrics.roe:.1%}" if metrics.roe else "ROE: N/A")
        print(f"ROA: {metrics.roa:.1%}" if metrics.roa else "ROA: N/A")
        print(f"Profit Margin: {metrics.profit_margin:.1%}" if metrics.profit_margin else "Profit Margin: N/A")
        print(f"Debt/Equity: {metrics.debt_to_equity:.2f}" if metrics.debt_to_equity else "Debt/Equity: N/A")
        print(f"Beta: {metrics.beta:.2f}" if metrics.beta else "Beta: N/A")
        print(f"Dividend Yield: {metrics.dividend_yield:.2%}" if metrics.dividend_yield else "Dividend Yield: N/A")
        
        # Display LLM insights
        print(f"\n🎯 AI-POWERED INSIGHTS")
        print(f"{'─' * 40}")
        print(f"Confidence Score: {llm_analysis.confidence_score:.2f}")
        
        print(f"\n📈 Growth Drivers:")
        for i, driver in enumerate(llm_analysis.growth_drivers[:5], 1):
            print(f"   {i}. {driver}")
        
        print(f"\n⚠️ Key Risks:")
        for i, risk in enumerate(llm_analysis.risks[:5], 1):
            print(f"   {i}. {risk}")
        
        print(f"\n🏆 Competitive Advantages:")
        for i, advantage in enumerate(llm_analysis.competitive_advantages[:5], 1):
            print(f"   {i}. {advantage}")
        
        print(f"\n📍 Market Position:")
        print(f"   {llm_analysis.market_position}")
        
        # Save detailed results
        results = {
            'analysis_date': datetime.now().isoformat(),
            'ticker': ticker,
            'company_info': info,
            'financial_metrics': metrics.model_dump(),
            'llm_analysis': llm_analysis.model_dump(),
            'data_quality': {
                'has_overview': bool(info),
                'has_historical_prices': not financial_data['historical_prices'].empty,
                'has_financials': not financial_data['financials'].empty,
                'has_balance_sheet': not financial_data['balance_sheet'].empty
            }
        }
        
        # Save to output directory
        output_dir = config.output_dir
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{ticker.lower()}_analysis_alphavantage.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Full analysis saved to: {output_file}")
        
        # Show API usage estimate
        print(f"\n📊 API USAGE SUMMARY")
        print(f"{'─' * 40}")
        print(f"Estimated API calls used: ~4")
        print(f"Remaining daily calls: ~{25 - 4}")
        print(f"You can analyze ~{(25 - 4) // 4} more companies today")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    # Allow user to specify ticker as command line argument
    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    
    success = test_single_company(ticker)
    
    if success:
        print(f"\n✅ Analysis completed successfully!")
    else:
        print(f"\n❌ Analysis failed. Check error messages above.")