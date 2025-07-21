# Company Fundamental Analysis System

A production-grade Python system that combines financial data APIs, LLM analysis, and machine learning models to perform comprehensive company valuation and fundamental analysis.

## Key Features

- **Financial Data Collection**: Automated collection from Yahoo Finance API with robust error handling
- **LLM-Powered Analysis**: Uses OpenAI/Anthropic APIs for intelligent company analysis with hallucination reduction
- **ML Valuation Models**: Multiple algorithms (XGBoost, Ridge, Random Forest) for valuation prediction
- **Production-Quality Code**: Type hints, structured logging, comprehensive error handling

## Technical Skills Demonstrated

1. **Strong Python Skills**
   - Clean, modular architecture with proper separation of concerns
   - Type hints and Pydantic models for data validation
   - Comprehensive error handling and structured logging

2. **LLM API Integration**
   - Multiple provider support (OpenAI and Anthropic)
   - Structured output generation with JSON schemas
   - Hallucination reduction techniques
   - Fact verification against financial data

3. **Machine Learning Implementation**
   - Feature engineering from financial metrics
   - Multiple model comparison (XGBoost, Ridge, Lasso, Random Forest)
   - Cross-validation and hyperparameter tuning
   - Model persistence and loading

4. **Statistical Analysis**
   - Understanding of financial ratios and their relationships
   - Proper handling of outliers and missing data
   - Model evaluation metrics (RMSE, R², MAPE)

## Project Structure

```
company-fundamental-analysis/
├── src/
│   ├── data_collection/
│   │   └── financial_data.py      # Yahoo Finance API integration
│   ├── analysis/
│   │   └── llm_analyzer.py        # LLM-based company analysis
│   ├── models/
│   │   └── valuation_model.py     # ML valuation models
│   └── utils/
│       └── config.py              # Configuration management
├── main.py                        # Main system orchestrator
├── requirements.txt               # Project dependencies
└── .env.example                   # Environment variables template
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   # or
   ANTHROPIC_API_KEY=your_key_here
   ```

## Usage Example

```python
from main import FundamentalAnalysisSystem

# Initialize system
system = FundamentalAnalysisSystem()

# Analyze a single company
analysis = system.analyze_company("AAPL")

# Compare multiple companies
comparison = system.analyze_portfolio(["AAPL", "MSFT", "GOOGL"])

# Train valuation model
training_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
model_results = system.train_valuation_model(training_tickers)

# Predict valuation
valuation = system.predict_valuation("TSLA")
```

## Key Components

### 1. Financial Data Collection
- Fetches comprehensive financial data: income statements, balance sheets, cash flows
- Calculates derived metrics (ROE, debt-to-equity, profit margins)
- Handles missing data and API errors gracefully

### 2. LLM Analysis with Verification
- Generates structured company analysis (products, competitive advantages, risks)
- Implements hallucination reduction:
  - Cross-references against actual financial data
  - Adjusts confidence scores based on data quality
  - Maintains source attribution

### 3. ML Valuation Models
- Feature engineering from 15+ financial metrics
- Model comparison with cross-validation
- Produces actionable recommendations (BUY/HOLD/SELL)

## Output Example

```json
{
  "ticker": "AAPL",
  "financial_metrics": {
    "pe_ratio": 28.5,
    "roe": 0.147,
    "debt_to_equity": 1.95
  },
  "llm_analysis": {
    "business_description": "Apple Inc. designs, manufactures, and markets smartphones, tablets, personal computers, and wearables globally",
    "competitive_advantages": [
      "Strong brand loyalty and ecosystem lock-in",
      "Premium pricing power in consumer electronics",
      "Integrated hardware and software design"
    ],
    "confidence_score": 0.92
  },
  "ml_valuation": {
    "predicted_pe": 26.3,
    "upside_potential": -0.077,
    "recommendation": "HOLD"
  }
}
```

## Production Considerations

- **Error Handling**: All API calls wrapped with retry logic
- **Rate Limiting**: Built-in delays for API compliance  
- **Caching**: Results cached to minimize API calls
- **Logging**: Structured logging for monitoring and debugging
- **Type Safety**: Full type hints for better IDE support

## Contact

This project demonstrates the technical skills required for the Rational AI position, combining financial analysis, LLM integration, and machine learning in a production-ready system.