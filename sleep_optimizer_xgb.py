#!/usr/bin/env python3
"""
Sleep Optimizer with XGBoost and Residualization
Isolates environmental effects from lifestyle factors using partial dependence
"""

import os
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.inspection import partial_dependence
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class OptimalConditions:
    """Optimal environmental conditions for sleep"""
    temperature: float = 18.5  # Celsius
    humidity: float = 45.0  # Percentage  
    voc_level: float = 100.0  # PPB
    light_level: float = 0.1  # Lux
    noise_level: float = 30.0  # dB
    co2_level: float = 600.0  # PPM

class SleepOptimizer:
    """XGBoost-based optimizer with environmental isolation"""
    
    def __init__(self, data_dir: str = "~/SomnaApp/data"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.data_dir / "models" / "xgb_model.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Environmental factors (what we want to optimize)
        self.environmental_features = [
            'temperature', 'humidity', 'voc_level', 
            'light_level', 'noise_level', 'co2_level'
        ]
        
        # Lifestyle confounders (what we need to control for)
        self.lifestyle_features = [
            'bedtime_hour', 'waketime_hour', 'sleep_debt',
            'stress_level', 'activity_minutes', 'caffeine_mg',
            'alcohol_units', 'screen_time_min', 'meal_timing_hours',
            'weekend', 'nap_duration'
        ]
        
        # Initialize models
        self.xgb_model = None
        self.lifestyle_model = None
        self.scaler = StandardScaler()
        self.optimal_conditions = OptimalConditions()
        self.is_trained = False
        
        # Load existing model if available
        self.load_model()
    
    def generate_synthetic_data(self, n_samples: int = 365) -> pd.DataFrame:
        """Generate realistic sleep data with lifestyle and environmental factors"""
        np.random.seed(42)
        data = []
        
        base_date = datetime.now() - timedelta(days=n_samples)
        sleep_debt = 0
        
        for i in range(n_samples):
            date = base_date + timedelta(days=i)
            
            # Lifestyle factors (confounders)
            bedtime = np.random.normal(23, 1.5)  # 11pm average
            waketime = np.random.normal(7, 1)  # 7am average
            stress = np.random.uniform(1, 10)
            activity = np.random.exponential(30)
            caffeine = np.random.exponential(100) if np.random.random() > 0.3 else 0
            alcohol = np.random.exponential(1) if np.random.random() > 0.7 else 0
            screen_time = np.random.exponential(60)
            meal_timing = np.random.normal(3, 1)
            weekend = 1 if date.weekday() >= 5 else 0
            nap = np.random.exponential(15) if np.random.random() > 0.8 else 0
            
            # Environmental factors with realistic correlations
            temp = np.random.normal(20, 3)
            humidity = np.random.normal(50, 15)
            voc = np.random.exponential(150)
            light = np.random.exponential(5) if bedtime < 22 else np.random.exponential(0.5)
            noise = np.random.exponential(35)
            co2 = np.random.normal(800, 200)
            
            # Calculate sleep quality with complex interactions
            # Environmental impact
            env_score = (
                -2 * abs(temp - 18.5) +  # Optimal around 18.5°C
                -0.5 * abs(humidity - 45) +  # Optimal around 45%
                -0.01 * voc +  # Lower VOC better
                -2 * np.log1p(light) +  # Darkness is crucial
                -0.1 * noise +  # Quiet is better
                -0.005 * (co2 - 600)  # Lower CO2 better
            )
            
            # Lifestyle impact (stronger effects)
            lifestyle_score = (
                -3 * abs(bedtime - 23) +  # Consistency matters
                -2 * abs(waketime - 7) +  # Regular wake time
                -2 * stress +  # Stress hurts sleep
                0.02 * min(activity, 60) +  # Activity helps to a point
                -0.02 * caffeine +  # Caffeine disrupts
                -3 * alcohol +  # Alcohol disrupts REM
                -0.02 * screen_time +  # Blue light exposure
                2 * max(0, meal_timing - 2) +  # Early dinner better
                3 * weekend +  # Weekend recovery
                -0.5 * nap +  # Naps can disrupt night sleep
                -3 * sleep_debt  # Accumulated debt
            )
            
            # Combined with noise and interactions
            sleep_quality = 70 + env_score + lifestyle_score + np.random.normal(0, 5)
            sleep_quality = np.clip(sleep_quality, 0, 100)
            
            # Update sleep debt
            actual_sleep = 8 - (100 - sleep_quality) / 20
            sleep_debt = max(0, sleep_debt + (7 - actual_sleep) * 0.3)
            
            data.append({
                'date': date.isoformat(),
                'sleep_quality': sleep_quality,
                'temperature': temp,
                'humidity': humidity,
                'voc_level': voc,
                'light_level': light,
                'noise_level': noise,
                'co2_level': co2,
                'bedtime_hour': bedtime,
                'waketime_hour': waketime,
                'sleep_debt': sleep_debt,
                'stress_level': stress,
                'activity_minutes': activity,
                'caffeine_mg': caffeine,
                'alcohol_units': alcohol,
                'screen_time_min': screen_time,
                'meal_timing_hours': meal_timing,
                'weekend': weekend,
                'nap_duration': nap
            })
        
        return pd.DataFrame(data)
    
    def residualize_environmental_effects(self, X: pd.DataFrame, y: np.ndarray, 
                                         train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Remove lifestyle effects to isolate environmental impact"""
        
        X_lifestyle = X[self.lifestyle_features]
        X_env = X[self.environmental_features]
        
        if train:
            # Train model to predict sleep from lifestyle factors only
            self.lifestyle_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.lifestyle_model.fit(X_lifestyle, y)
        
        # Get predictions from lifestyle factors
        lifestyle_predictions = self.lifestyle_model.predict(X_lifestyle)
        
        # Calculate residuals (environmental effect)
        residuals = y - lifestyle_predictions
        
        return residuals, lifestyle_predictions
    
    def train(self, force_retrain: bool = False) -> Dict[str, float]:
        """Train XGBoost model with residualization"""
        
        if self.is_trained and not force_retrain:
            logger.info("Model already trained. Use force_retrain=True to retrain.")
            return self.get_metrics()
        
        logger.info("Training XGBoost model with environmental isolation...")
        
        # Load or generate data
        data = self.load_or_generate_data()
        
        # Prepare features
        X = data[self.environmental_features + self.lifestyle_features]
        y = data['sleep_quality'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Step 1: Residualize to isolate environmental effects
        logger.info("Isolating environmental effects from lifestyle factors...")
        residuals_train, lifestyle_pred_train = self.residualize_environmental_effects(
            X_train, y_train, train=True
        )
        residuals_test, lifestyle_pred_test = self.residualize_environmental_effects(
            X_test, y_test, train=False
        )
        
        # Step 2: Train XGBoost on environmental features to predict residuals
        logger.info("Training XGBoost on environmental factors...")
        X_env_train = X_train[self.environmental_features]
        X_env_test = X_test[self.environmental_features]
        
        # Scale environmental features
        X_env_train_scaled = self.scaler.fit_transform(X_env_train)
        X_env_test_scaled = self.scaler.transform(X_env_test)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.xgb_model.fit(
            X_env_train_scaled, 
            residuals_train,
            eval_set=[(X_env_test_scaled, residuals_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Step 3: Calculate optimal conditions using partial dependence
        logger.info("Calculating optimal environmental conditions...")
        self.calculate_optimal_conditions(X_env_train_scaled)
        
        # Evaluate
        env_predictions = self.xgb_model.predict(X_env_test_scaled)
        total_predictions = lifestyle_pred_test + env_predictions
        
        # Metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        self.metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, total_predictions))),
            'mae': float(mean_absolute_error(y_test, total_predictions)),
            'r2': float(r2_score(y_test, total_predictions)),
            'env_r2': float(r2_score(residuals_test, env_predictions)),
            'lifestyle_r2': float(r2_score(y_test, lifestyle_pred_test)),
            'feature_importance': self.get_feature_importance(),
            'optimal_conditions': asdict(self.optimal_conditions)
        }
        
        self.is_trained = True
        self.save_model()
        
        logger.info(f"Training complete! Overall R²: {self.metrics['r2']:.3f}")
        logger.info(f"Environmental R²: {self.metrics['env_r2']:.3f}")
        logger.info(f"Lifestyle R²: {self.metrics['lifestyle_r2']:.3f}")
        
        return self.metrics
    
    def calculate_optimal_conditions(self, X_scaled: np.ndarray) -> None:
        """Calculate optimal conditions using partial dependence"""
        
        optimal = {}
        feature_ranges = {}
        
        for i, feature in enumerate(self.environmental_features):
            # Get partial dependence
            pd_result = partial_dependence(
                self.xgb_model, X_scaled, [i],
                percentiles=(5, 95), grid_resolution=50
            )
            
            # Find value that maximizes sleep quality
            values = pd_result['values'][0]
            pd_values = pd_result['average'][0]
            optimal_idx = np.argmax(pd_values)
            
            # Inverse transform to get original scale
            dummy = np.zeros((1, len(self.environmental_features)))
            dummy[0, i] = values[optimal_idx]
            original_value = self.scaler.inverse_transform(dummy)[0, i]
            
            optimal[feature] = float(original_value)
            feature_ranges[feature] = (float(values.min()), float(values.max()))
        
        # Update optimal conditions
        self.optimal_conditions = OptimalConditions(
            temperature=optimal.get('temperature', 18.5),
            humidity=optimal.get('humidity', 45),
            voc_level=optimal.get('voc_level', 100),
            light_level=optimal.get('light_level', 0.1),
            noise_level=optimal.get('noise_level', 30),
            co2_level=optimal.get('co2_level', 600)
        )
    
    def predict(self, conditions: Dict[str, float]) -> Dict[str, Any]:
        """Predict sleep quality and provide insights"""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create DataFrame
        df = pd.DataFrame([conditions])
        
        # Ensure all features present
        for feature in self.environmental_features + self.lifestyle_features:
            if feature not in df.columns:
                if feature == 'weekend':
                    df[feature] = 1 if datetime.now().weekday() >= 5 else 0
                else:
                    df[feature] = 0
        
        # Get lifestyle prediction
        X_lifestyle = df[self.lifestyle_features]
        lifestyle_impact = self.lifestyle_model.predict(X_lifestyle)[0]
        
        # Get environmental prediction
        X_env = df[self.environmental_features]
        X_env_scaled = self.scaler.transform(X_env)
        env_impact = self.xgb_model.predict(X_env_scaled)[0]
        
        # Total prediction
        total_quality = lifestyle_impact + env_impact
        total_quality = np.clip(total_quality, 0, 100)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(conditions)
        
        return {
            'predicted_quality': float(total_quality),
            'environmental_impact': float(env_impact),
            'lifestyle_impact': float(lifestyle_impact),
            'optimal_conditions': asdict(self.optimal_conditions),
            'recommendations': recommendations
        }
    
    def generate_recommendations(self, current: Dict[str, float]) -> List[Dict[str, str]]:
        """Generate specific recommendations based on current conditions"""
        
        recommendations = []
        optimal = asdict(self.optimal_conditions)
        
        # Check each environmental factor
        thresholds = {
            'temperature': 2,
            'humidity': 10,
            'voc_level': 100,
            'light_level': 2,
            'noise_level': 10,
            'co2_level': 200
        }
        
        for feature in self.environmental_features:
            if feature not in current:
                continue
                
            current_val = current[feature]
            optimal_val = optimal[feature]
            threshold = thresholds[feature]
            
            if abs(current_val - optimal_val) > threshold:
                if feature == 'temperature':
                    if current_val > optimal_val:
                        recommendations.append({
                            'name': 'Temperature',
                            'message': f'Lower to {optimal_val:.1f}°C (currently {current_val:.1f}°C)'
                        })
                    else:
                        recommendations.append({
                            'name': 'Temperature',
                            'message': f'Increase to {optimal_val:.1f}°C (currently {current_val:.1f}°C)'
                        })
                
                elif feature == 'humidity':
                    if current_val > optimal_val:
                        recommendations.append({
                            'name': 'Humidity',
                            'message': f'Reduce to {optimal_val:.0f}% (currently {current_val:.0f}%)'
                        })
                    else:
                        recommendations.append({
                            'name': 'Humidity',
                            'message': f'Increase to {optimal_val:.0f}% (currently {current_val:.0f}%)'
                        })
                
                elif feature == 'voc_level' and current_val > optimal_val:
                    recommendations.append({
                        'name': 'Air Quality',
                        'message': f'Improve ventilation - VOC at {current_val:.0f}ppb (target: {optimal_val:.0f}ppb)'
                    })
                
                elif feature == 'light_level' and current_val > optimal_val:
                    recommendations.append({
                        'name': 'Light',
                        'message': f'Reduce to {optimal_val:.1f} lux (currently {current_val:.1f} lux)'
                    })
                
                elif feature == 'noise_level' and current_val > optimal_val:
                    recommendations.append({
                        'name': 'Noise',
                        'message': f'Reduce to {optimal_val:.0f}dB (currently {current_val:.0f}dB)'
                    })
                
                elif feature == 'co2_level' and current_val > optimal_val:
                    recommendations.append({
                        'name': 'CO2',
                        'message': f'Ventilate - CO2 at {current_val:.0f}ppm (target: {optimal_val:.0f}ppm)'
                    })
        
        return recommendations[:3]  # Top 3 most important
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost model"""
        
        if self.xgb_model is None:
            return {}
        
        importance = self.xgb_model.feature_importances_
        return dict(zip(self.environmental_features, importance.tolist()))
    
    def load_or_generate_data(self) -> pd.DataFrame:
        """Load existing data or generate synthetic data"""
        
        data_file = self.data_dir / "sleep_data.csv"
        
        if data_file.exists():
            logger.info(f"Loading data from {data_file}")
            return pd.read_csv(data_file)
        else:
            logger.info("Generating synthetic data...")
            data = self.generate_synthetic_data()
            data.to_csv(data_file, index=False)
            return data
    
    def save_model(self) -> None:
        """Save trained models"""
        
        model_data = {
            'xgb_model': self.xgb_model,
            'lifestyle_model': self.lifestyle_model,
            'scaler': self.scaler,
            'optimal_conditions': self.optimal_conditions,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load model from disk"""
        
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.xgb_model = model_data['xgb_model']
                self.lifestyle_model = model_data['lifestyle_model']
                self.scaler = model_data['scaler']
                self.optimal_conditions = model_data['optimal_conditions']
                self.metrics = model_data.get('metrics', {})
                self.is_trained = model_data['is_trained']
                
                logger.info("Model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current model metrics"""
        return self.metrics if hasattr(self, 'metrics') else {}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Sleep Optimizer with XGBoost')
    parser.add_argument('action', nargs='?', default='train',
                       choices=['train', 'force-retrain', 'predict'],
                       help='Action to perform')
    parser.add_argument('--data-dir', default='~/SomnaApp/data',
                       help='Data directory path')
    
    args = parser.parse_args()
    
    optimizer = SleepOptimizer(data_dir=args.data_dir)
    
    if args.action == 'force-retrain':
        metrics = optimizer.train(force_retrain=True)
        print(f"\n✓ Model retrained")
        print(f"  R² Score: {metrics['r2']:.3f}")
        print(f"  Environmental R²: {metrics['env_r2']:.3f}")
        print(f"\n✓ Optimal Conditions:")
        for key, value in metrics['optimal_conditions'].items():
            print(f"  {key}: {value:.1f}")
    
    elif args.action == 'train':
        metrics = optimizer.train(force_retrain=False)
        if metrics:
            print(f"\n✓ Model ready")
    
    elif args.action == 'predict':
        # Example prediction
        result = optimizer.predict({
            'temperature': 22,
            'humidity': 55,
            'voc_level': 200,
            'light_level': 5,
            'noise_level': 40,
            'co2_level': 900,
            'stress_level': 6,
            'bedtime_hour': 23.5,
            'waketime_hour': 7
        })
        
        print(f"\nPredicted Quality: {result['predicted_quality']:.0f}/100")
        print(f"Environmental Impact: {result['environmental_impact']:+.1f}")
        print(f"Lifestyle Impact: {result['lifestyle_impact']:+.1f}")

if __name__ == "__main__":
    main()
