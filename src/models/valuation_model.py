from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import structlog
import joblib
from pathlib import Path

logger = structlog.get_logger()


class FundamentalValuationModel:
    """
    ML model for company valuation based on fundamental metrics.
    Uses multiple algorithms and selects the best performer.
    """
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.best_model = None
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_selector = SelectKBest(f_regression, k=15)
        self.feature_names = None
        self.selected_features = None
        
    def prepare_features(self, financial_data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features from financial data for modeling.
        
        Args:
            financial_data: List of company financial data dictionaries
            
        Returns:
            Features DataFrame and target Series
        """
        logger.info("Preparing features for modeling", n_companies=len(financial_data))
        
        features_list = []
        targets = []
        
        for company_data in financial_data:
            metrics = company_data['metrics']
            info = company_data['info']
            
            # Extract features
            features = {
                # Valuation metrics
                'pe_ratio': metrics.pe_ratio or 0,
                'forward_pe': metrics.forward_pe or 0,
                'peg_ratio': metrics.peg_ratio or 0,
                'price_to_book': metrics.price_to_book or 0,
                
                # Profitability metrics
                'roe': metrics.roe or 0,
                'roa': metrics.roa or 0,
                'profit_margin': metrics.profit_margin or 0,
                
                # Growth metrics
                'revenue_growth': metrics.revenue_growth or 0,
                'earnings_growth': metrics.earnings_growth or 0,
                
                # Financial health
                'debt_to_equity': metrics.debt_to_equity or 0,
                'free_cash_flow_scaled': (metrics.free_cash_flow or 0) / (info.get('marketCap', 1)),
                
                # Market metrics
                'beta': metrics.beta or 1,
                'dividend_yield': metrics.dividend_yield or 0,
                
                # Size factor
                'log_market_cap': np.log(info.get('marketCap', 1)),
                
                # Sector encoding (simplified)
                'is_tech': int(info.get('sector', '') == 'Technology'),
                'is_finance': int(info.get('sector', '') == 'Financial Services'),
                'is_healthcare': int(info.get('sector', '') == 'Healthcare'),
            }
            
            # Create derived features
            features['pe_to_growth'] = features['pe_ratio'] / (features['revenue_growth'] + 0.01)
            features['quality_score'] = features['roe'] * features['profit_margin']
            features['financial_health_score'] = 1 / (1 + features['debt_to_equity'])
            
            # Target: forward P/E as proxy for valuation
            target = info.get('forwardPE', info.get('trailingPE', 0))
            
            if target > 0 and target < 100:  # Filter outliers
                features_list.append(features)
                targets.append(target)
        
        X = pd.DataFrame(features_list)
        y = pd.Series(targets)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info("Feature preparation complete", 
                   n_features=len(self.feature_names),
                   n_samples=len(X))
        
        return X, y
    
    def build_models(self) -> Dict[str, Pipeline]:
        """Build multiple model pipelines."""
        
        models = {
            'ridge': Pipeline([
                ('scaler', self.scaler),
                ('selector', self.feature_selector),
                ('model', Ridge(alpha=1.0))
            ]),
            
            'lasso': Pipeline([
                ('scaler', self.scaler),
                ('selector', self.feature_selector),
                ('model', Lasso(alpha=0.1))
            ]),
            
            'elastic_net': Pipeline([
                ('scaler', self.scaler),
                ('selector', self.feature_selector),
                ('model', ElasticNet(alpha=0.1, l1_ratio=0.5))
            ]),
            
            'xgboost': Pipeline([
                ('scaler', self.scaler),
                ('selector', self.feature_selector),
                ('model', xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ))
            ]),
            
            'random_forest': Pipeline([
                ('scaler', self.scaler),
                ('selector', self.feature_selector),
                ('model', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                ))
            ])
        }
        
        return models
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """
        Train multiple models and evaluate their performance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary of model performances
        """
        logger.info("Starting model training and evaluation")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        models = self.build_models()
        results = {}
        
        for name, pipeline in models.items():
            logger.info(f"Training {name} model")
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': pipeline,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'cv_rmse': cv_rmse,
                'predictions': y_pred,
                'actuals': y_test
            }
            
            logger.info(f"{name} performance", 
                       rmse=f"{rmse:.2f}",
                       r2=f"{r2:.3f}",
                       mape=f"{mape:.2%}")
        
        # Select best model based on cross-validation RMSE
        best_model_name = min(results.keys(), key=lambda k: results[k]['cv_rmse'])
        self.best_model = results[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name}")
        
        # Get selected features
        selector = self.best_model.named_steps['selector']
        selected_indices = selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        logger.info("Selected features", features=self.selected_features)
        
        return results
    
    def predict_valuation(self, company_data: Dict) -> Dict[str, float]:
        """
        Predict valuation for a single company.
        
        Args:
            company_data: Company financial data
            
        Returns:
            Dictionary with predicted metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features for single company
        X, _ = self.prepare_features([company_data])
        
        # Make prediction
        predicted_pe = self.best_model.predict(X)[0]
        
        # Calculate implied price
        current_price = company_data['info'].get('currentPrice', 0)
        eps = company_data['info'].get('trailingEps', 0)
        
        implied_price = predicted_pe * eps if eps > 0 else 0
        upside_potential = (implied_price - current_price) / current_price if current_price > 0 else 0
        
        return {
            'predicted_pe': predicted_pe,
            'current_pe': company_data['metrics'].pe_ratio,
            'implied_price': implied_price,
            'current_price': current_price,
            'upside_potential': upside_potential,
            'recommendation': self._generate_recommendation(upside_potential)
        }
    
    def _generate_recommendation(self, upside_potential: float) -> str:
        """Generate investment recommendation based on upside potential."""
        if upside_potential > 0.20:
            return "STRONG BUY"
        elif upside_potential > 0.10:
            return "BUY"
        elif upside_potential > -0.10:
            return "HOLD"
        elif upside_potential > -0.20:
            return "SELL"
        else:
            return "STRONG SELL"
    
    def feature_importance_analysis(self) -> pd.DataFrame:
        """Analyze feature importance across models."""
        if 'xgboost' in self.models:
            # Get XGBoost feature importance
            xgb_model = self.models['xgboost']['model'].named_steps['model']
            importance = xgb_model.feature_importances_
            
            # Map to selected features
            selector = self.models['xgboost']['model'].named_steps['selector']
            selected_indices = selector.get_support(indices=True)
            
            importance_df = pd.DataFrame({
                'feature': [self.feature_names[i] for i in selected_indices],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()
    
    def save_model(self, path: Path):
        """Save the trained model."""
        if self.best_model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.best_model,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features
        }
        
        joblib.dump(model_data, path)
        logger.info("Model saved", path=str(path))
    
    def load_model(self, path: Path):
        """Load a trained model."""
        model_data = joblib.load(path)
        self.best_model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.selected_features = model_data['selected_features']
        logger.info("Model loaded", path=str(path))