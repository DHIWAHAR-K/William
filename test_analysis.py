#!/usr/bin/env python3
"""
Quick test script to demonstrate the system without full training.
Run this to see the analysis in action with minimal setup.
"""

from main import FundamentalAnalysisSystem
import json
from datetime import datetime

def test_single_company():
    """Test analyzing a single company."""
    print("=" * 80)
    print("COMPANY FUNDAMENTAL ANALYSIS - QUICK TEST")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Initialize system
        print("Initializing system...")
        system = FundamentalAnalysisSystem()
        
        # Analyze Microsoft
        print("\nAnalyzing Microsoft (MSFT)...")
        print("-" * 40)
        
        result = system.analyze_company("MSFT")
        
        # Print financial metrics
        print("\nFINANCIAL METRICS:")
        metrics = result['financial_metrics']
        print(f"  Company: {metrics.get('company_name', 'N/A')}")
        print(f"  P/E Ratio: {metrics.get('pe_ratio', 'N/A')}")
        print(f"  Forward P/E: {metrics.get('forward_pe', 'N/A')}")
        print(f"  ROE: {metrics.get('roe', 'N/A')}")
        print(f"  Profit Margin: {metrics.get('profit_margin', 'N/A')}")
        print(f"  Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}")
        print(f"  Revenue Growth: {metrics.get('revenue_growth', 'N/A')}")
        
        # Print LLM analysis
        print("\nLLM ANALYSIS:")
        llm = result['llm_analysis']
        print(f"  Business: {llm.get('business_description', 'N/A')}")
        print(f"\n  Products/Services:")
        for product in llm.get('products_services', [])[:3]:
            print(f"    - {product}")
        print(f"\n  Competitive Advantages:")
        for advantage in llm.get('competitive_advantages', [])[:3]:
            print(f"    - {advantage}")
        print(f"\n  Key Risks:")
        for risk in llm.get('risks', [])[:3]:
            print(f"    - {risk}")
        print(f"\n  Confidence Score: {llm.get('confidence_score', 'N/A')}")
        
        # Save full results
        output_file = "msft_analysis_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull results saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed requirements: pip install -r requirements.txt")
        print("2. Set up .env file with OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("3. Have internet connection for Yahoo Finance API")

def test_portfolio_comparison():
    """Test comparing multiple companies."""
    print("\n\n" + "=" * 80)
    print("PORTFOLIO COMPARISON TEST")
    print("=" * 80)
    
    try:
        system = FundamentalAnalysisSystem()
        
        # Compare tech giants
        print("\nComparing: Apple, Microsoft, Google")
        print("-" * 40)
        
        comparison = system.analyze_portfolio(["AAPL", "MSFT", "GOOGL"])
        
        # Print key metrics
        print("\nKey Metrics Comparison:")
        print(comparison[['pe_ratio', 'roe', 'profit_margin', 'debt_to_equity']].to_string())
        
        # Save results
        comparison.to_csv("tech_comparison.csv")
        print("\nComparison saved to: tech_comparison.csv")
        
    except Exception as e:
        print(f"\nError in portfolio comparison: {str(e)}")

if __name__ == "__main__":
    # Run single company test
    test_single_company()
    
    # Optionally run portfolio comparison
    # Uncomment the line below to compare multiple companies
    # test_portfolio_comparison()