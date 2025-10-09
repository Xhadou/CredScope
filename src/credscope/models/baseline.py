import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import numpy as np
import pandas as pd

class BaselineModel:
    """Baseline logistic regression model"""
    
    def __init__(self, config: dict):
        self.config = config
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
    def create_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simple baseline features"""
        features = df.copy()
        
        # Basic feature engineering
        features['CREDIT_TO_ANNUITY_RATIO'] = features['AMT_CREDIT'] / (features['AMT_ANNUITY'] + 1)
        features['CREDIT_TO_GOODS_RATIO'] = features['AMT_CREDIT'] / (features['AMT_GOODS_PRICE'] + 1)
        features['AGE_YEARS'] = -features['DAYS_BIRTH'] / 365.25
        features['EMPLOYED_YEARS'] = -features['DAYS_EMPLOYED'] / 365.25
        features['EMPLOYED_YEARS'] = features['EMPLOYED_YEARS'].replace([np.inf, -np.inf], 0)
        
        # Select only numeric columns for baseline
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        return features[numeric_features]
    
    def build_pipeline(self) -> Pipeline:
        """Build preprocessing and model pipeline"""
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
        return pipeline
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train baseline model with MLflow tracking"""
        
        with mlflow.start_run(run_name="baseline_logistic_regression"):
            # Create baseline features
            X_train_feat = self.create_baseline_features(X_train)
            X_val_feat = self.create_baseline_features(X_val)
            
            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("n_features", X_train_feat.shape[1])
            mlflow.log_param("n_train_samples", X_train_feat.shape[0])
            
            # Build and train pipeline
            pipeline = self.build_pipeline()
            pipeline.fit(X_train_feat, y_train)
            
            # Make predictions
            y_train_pred_proba = pipeline.predict_proba(X_train_feat)[:, 1]
            y_val_pred_proba = pipeline.predict_proba(X_val_feat)[:, 1]
            
            # Calculate metrics
            train_auc = roc_auc_score(y_train, y_train_pred_proba)
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            
            # Log metrics (convert to Python float for MLflow)
            mlflow.log_metric("train_auc", float(train_auc))
            mlflow.log_metric("val_auc", float(val_auc))
            
            print(f"\n📊 Baseline Model Performance:")
            print(f"Train AUC: {train_auc:.4f}")
            print(f"Validation AUC: {val_auc:.4f}")
            
            # Save model
            model_path = "models/baseline_model.pkl"
            joblib.dump(pipeline, model_path)
            mlflow.sklearn.log_model(pipeline, "model")
            
            # Log feature importance (coefficients for logistic regression)
            feature_names = X_train_feat.columns
            coefficients = pipeline.named_steps['model'].coef_[0]
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coefficients)
            }).sort_values('importance', ascending=False).head(20)
            
            print("\n🔝 Top 20 Important Features:")
            print(feature_importance.to_string(index=False))
            
            return pipeline, val_auc