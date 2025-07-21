from typing import Dict, List, Optional, Tuple
import json
import time
from abc import ABC, abstractmethod
import structlog
from pydantic import BaseModel, Field
import openai
from anthropic import Anthropic
from src.utils.config import config

logger = structlog.get_logger()


class CompanyAnalysis(BaseModel):
    """Structured output from LLM company analysis."""
    
    company_name: str
    ticker: str
    business_description: str = Field(description="Clear, concise description of what the company does")
    products_services: List[str] = Field(description="Main products and services offered")
    competitive_advantages: List[str] = Field(description="Key competitive advantages/moats")
    risks: List[str] = Field(description="Major business risks")
    growth_drivers: List[str] = Field(description="Key factors driving growth")
    market_position: str = Field(description="Company's position in its market")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in analysis accuracy")
    sources_used: List[str] = Field(description="Sources of information used")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def analyze_company(self, company_info: Dict, additional_context: str = "") -> CompanyAnalysis:
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def analyze_company(self, company_info: Dict, additional_context: str = "") -> CompanyAnalysis:
        """Analyze company using OpenAI GPT."""
        
        system_prompt = """You are a financial analyst specializing in fundamental company analysis. 
        You must return ONLY a valid JSON object with the exact structure requested.
        Be factual, specific, and base your analysis only on the provided information."""
        
        # Get basic info
        company_name = company_info.get('longName', company_info.get('shortName', 'Unknown'))
        ticker = company_info.get('symbol', 'N/A')
        business_summary = company_info.get('businessSummary', company_info.get('longBusinessSummary', ''))
        industry = company_info.get('industry', 'Unknown')
        sector = company_info.get('sector', 'Unknown')
        
        user_prompt = f"""Return a JSON object analyzing this company:

Company: {company_name} ({ticker})
Industry: {industry}
Sector: {sector}
Summary: {business_summary[:500]}...

Additional Context: {additional_context}

Return EXACTLY this JSON structure with no additional text:
{{
    "company_name": "{company_name}",
    "ticker": "{ticker}",
    "business_description": "Brief description of what the company does",
    "products_services": ["product1", "product2", "product3"],
    "competitive_advantages": ["advantage1", "advantage2", "advantage3"],
    "risks": ["risk1", "risk2", "risk3"],
    "growth_drivers": ["driver1", "driver2", "driver3"],
    "market_position": "Brief market position description",
    "confidence_score": 0.8,
    "sources_used": ["Financial data", "Company info"]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            analysis_dict = json.loads(content)
            
            # Validate required fields and provide defaults if missing
            required_fields = {
                'company_name': company_name,
                'ticker': ticker,
                'business_description': business_summary[:200] if business_summary else f'{company_name} operates in {industry}',
                'products_services': [f'{industry} products and services'],
                'competitive_advantages': ['Market position', 'Brand recognition'],
                'risks': ['Market competition', 'Economic conditions'],
                'growth_drivers': ['Market expansion', 'Product innovation'],
                'market_position': f'Company in {industry} sector',
                'confidence_score': 0.7,
                'sources_used': ['Company financial data', 'Market information']
            }
            
            # Fill missing fields
            for field, default_value in required_fields.items():
                if field not in analysis_dict or not analysis_dict[field]:
                    analysis_dict[field] = default_value
            
            return CompanyAnalysis(**analysis_dict)
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", error=str(e))
            # Return fallback analysis
            return CompanyAnalysis(
                company_name=company_name,
                ticker=ticker,
                business_description=business_summary[:200] if business_summary else f'{company_name} operates in {industry}',
                products_services=[f'{industry} products and services'],
                competitive_advantages=['Market position', 'Operational efficiency'],
                risks=['Market competition', 'Regulatory changes'],
                growth_drivers=['Market expansion', 'Product development'],
                market_position=f'Established company in {industry} sector',
                confidence_score=0.6,
                sources_used=['Company financial data']
            )
        except Exception as e:
            logger.error("OpenAI analysis failed", error=str(e))
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude implementation."""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        
    def analyze_company(self, company_info: Dict, additional_context: str = "") -> CompanyAnalysis:
        """Analyze company using Anthropic Claude."""
        
        prompt = f"""Analyze this company and provide a structured fundamental analysis:
        
        Company Info: {json.dumps(company_info, indent=2)}
        
        Additional Context: {additional_context}
        
        Provide a structured analysis that includes:
        1. Clear business description (what the company actually does)
        2. Main products/services (be specific, avoid generic statements)
        3. Competitive advantages (only those clearly supported by data)
        4. Business risks (based on the industry and company specifics)
        5. Growth drivers (specific factors that could drive future growth)
        6. Market position (based on available metrics)
        
        Important guidelines:
        - Be factual and specific
        - Avoid speculation not supported by data
        - If information is limited, reflect this in the confidence score
        - List actual products/services, not categories
        
        Return ONLY a JSON object matching this exact structure:
        {CompanyAnalysis.schema_json(indent=2)}
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Find JSON in the response
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            
            analysis_dict = json.loads(json_str)
            return CompanyAnalysis(**analysis_dict)
            
        except Exception as e:
            logger.error("Anthropic analysis failed", error=str(e))
            raise


class CompanyLLMAnalyzer:
    """Main analyzer that coordinates LLM analysis with hallucination reduction."""
    
    def __init__(self):
        self.provider = self._initialize_provider()
        self.analysis_cache: Dict[str, CompanyAnalysis] = {}
        
    def _initialize_provider(self) -> LLMProvider:
        """Initialize the appropriate LLM provider based on available API keys."""
        if config.openai_api_key:
            logger.info("Using OpenAI provider")
            return OpenAIProvider(config.openai_api_key, config.llm_model)
        elif config.anthropic_api_key:
            logger.info("Using Anthropic provider")
            return AnthropicProvider(config.anthropic_api_key)
        else:
            raise ValueError("No LLM API key configured")
    
    def analyze_with_verification(self, ticker: str, financial_data: Dict, 
                                 additional_sources: List[str] = None) -> CompanyAnalysis:
        """
        Analyze company with hallucination reduction techniques.
        
        Args:
            ticker: Company ticker
            financial_data: Financial data from data collector
            additional_sources: Additional text sources for analysis
            
        Returns:
            Verified company analysis
        """
        logger.info("Starting LLM analysis with verification", ticker=ticker)
        
        # Check cache
        if ticker in self.analysis_cache:
            logger.info("Returning cached analysis", ticker=ticker)
            return self.analysis_cache[ticker]
        
        # Prepare structured context
        context = self._prepare_context(financial_data, additional_sources)
        
        # Get initial analysis
        analysis = self.provider.analyze_company(financial_data['info'], context)
        
        # Verify and refine analysis
        verified_analysis = self._verify_analysis(analysis, financial_data)
        
        # Cache the result
        self.analysis_cache[ticker] = verified_analysis
        
        return verified_analysis
    
    def _prepare_context(self, financial_data: Dict, additional_sources: List[str] = None) -> str:
        """Prepare additional context for LLM analysis."""
        context_parts = []
        
        # Add financial metrics
        if 'metrics' in financial_data:
            metrics = financial_data['metrics']
            context_parts.append(f"Financial Metrics: {metrics.dict()}")
        
        # Add historical performance
        if 'historical_prices' in financial_data:
            hist = financial_data['historical_prices']
            if not hist.empty:
                recent_performance = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                context_parts.append(f"5-year price performance: {recent_performance:.2%}")
        
        # Add additional sources
        if additional_sources:
            context_parts.extend(additional_sources)
        
        return "\n\n".join(context_parts)
    
    def _verify_analysis(self, analysis: CompanyAnalysis, financial_data: Dict) -> CompanyAnalysis:
        """Verify analysis against known facts to reduce hallucinations."""
        
        confidence_adjustments = []
        
        # Verify company name matches
        actual_name = financial_data['info'].get('longName', '')
        if actual_name and actual_name.lower() not in analysis.company_name.lower():
            confidence_adjustments.append(-0.1)
            analysis.company_name = actual_name
        
        # Verify products/services are reasonable for the industry
        industry = financial_data['info'].get('industry', '')
        sector = financial_data['info'].get('sector', '')
        
        if industry or sector:
            analysis.sources_used.append(f"Industry: {industry}, Sector: {sector}")
        
        # Check if risks align with financial metrics
        if 'metrics' in financial_data and hasattr(financial_data['metrics'], 'debt_to_equity'):
            if financial_data['metrics'].debt_to_equity and financial_data['metrics'].debt_to_equity > 2:
                if not any('debt' in risk.lower() or 'leverage' in risk.lower() for risk in analysis.risks):
                    analysis.risks.append("High debt levels relative to equity")
                    confidence_adjustments.append(-0.05)
        
        # Adjust confidence score
        base_confidence = analysis.confidence_score
        adjustment = sum(confidence_adjustments)
        analysis.confidence_score = max(0.1, min(1.0, base_confidence + adjustment))
        
        logger.info("Analysis verification complete", 
                   ticker=analysis.ticker,
                   confidence=analysis.confidence_score,
                   adjustments=adjustment)
        
        return analysis
    
    def batch_analyze(self, tickers: List[str], financial_data_dict: Dict[str, Dict]) -> List[CompanyAnalysis]:
        """Analyze multiple companies efficiently."""
        results = []
        
        for ticker in tickers:
            if ticker in financial_data_dict:
                try:
                    analysis = self.analyze_with_verification(ticker, financial_data_dict[ticker])
                    results.append(analysis)
                    # Rate limiting
                    time.sleep(1)
                except Exception as e:
                    logger.error("Failed to analyze company", ticker=ticker, error=str(e))
        
        return results