"""Model Explanation with SHAP

This module provides SHAP-based model interpretation for credit risk models.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based model explainer for LightGBM and XGBoost"""
    
    def __init__(self, model, X_sample, feature_names):
        """Initialize SHAP explainer
        
        Args:
            model: Trained model (LightGBM or XGBoost)
            X_sample: Sample data for background (use ~1000 rows)
            feature_names: List of feature names
        """
        self.model = model
        self.X_sample = X_sample
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def compute_shap_values(self, X_explain):
        """Compute SHAP values for given data
        
        Args:
            X_explain: Data to explain
            
        Returns:
            SHAP values array
        """
        logger.info(f"Computing SHAP values for {len(X_explain)} samples...")
        
        # Create explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_explain)
        
        # For binary classification, extract positive class SHAP values
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        logger.info("✓ SHAP values computed")
        return self.shap_values
    
    def plot_summary(self, output_path: Optional[str] = None, max_display: int = 20):
        """Create SHAP summary plot
        
        Args:
            output_path: Path to save plot (optional)
            max_display: Number of features to display
        """
        if self.shap_values is None:
            raise ValueError("Must compute SHAP values first")
        
        logger.info("Creating SHAP summary plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Summary plot saved to {output_path}")
        
        plt.close()
    
    def plot_feature_importance(self, output_path: Optional[str] = None, max_display: int = 20):
        """Create SHAP feature importance plot
        
        Args:
            output_path: Path to save plot (optional)
            max_display: Number of features to display
        """
        if self.shap_values is None:
            raise ValueError("Must compute SHAP values first")
        
        logger.info("Creating SHAP feature importance plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Feature importance plot saved to {output_path}")
        
        plt.close()
    
    def get_feature_importance_df(self, top_n: int = 30) -> pd.DataFrame:
        """Get feature importance as DataFrame
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("Must compute SHAP values first")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def explain_prediction(self, X_single, output_path: Optional[str] = None):
        """Explain a single prediction with force plot
        
        Args:
            X_single: Single sample to explain (1D array or Series)
            output_path: Path to save plot (optional)
        """
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values for single prediction
        shap_values_single = self.explainer.shap_values(X_single.reshape(1, -1))
        
        # For binary classification
        if isinstance(shap_values_single, list):
            shap_values_single = shap_values_single[1][0]
        else:
            shap_values_single = shap_values_single[0]
        
        # Create force plot
        shap.force_plot(
            self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
            else self.explainer.expected_value,
            shap_values_single,
            X_single,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Force plot saved to {output_path}")
            plt.close()


def analyze_model_with_shap(model, X_train, X_val, feature_names, output_dir: str = 'analysis'):
    """Complete SHAP analysis for a model
    
    Args:
        model: Trained model
        X_train: Training data (for background)
        X_val: Validation data (to explain)
        feature_names: List of feature names
        output_dir: Directory to save plots
        
    Returns:
        Feature importance DataFrame
    """
    logger.info("Starting complete SHAP analysis...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Sample data for efficiency
    sample_size = min(1000, len(X_train))
    explain_size = min(500, len(X_val))
    
    X_background = X_train.sample(n=sample_size, random_state=42)
    X_explain = X_val.sample(n=explain_size, random_state=42)
    
    # Create explainer
    explainer = SHAPExplainer(model, X_background, feature_names)
    
    # Compute SHAP values
    explainer.compute_shap_values(X_explain)
    
    # Generate plots
    explainer.plot_summary(
        output_path=str(output_path / 'shap_summary.png'),
        max_display=20
    )
    
    explainer.plot_feature_importance(
        output_path=str(output_path / 'shap_importance.png'),
        max_display=20
    )
    
    # Get feature importance
    importance_df = explainer.get_feature_importance_df(top_n=30)
    importance_df.to_csv(output_path / 'shap_feature_importance.csv', index=False)
    
    logger.info(f"✓ SHAP analysis complete! Plots saved to {output_path}")
    
    return importance_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("SHAP Model Explainer")
    print("=" * 50)
    print("\nUsage:")
    print("""
    from src.credscope.evaluation.explainer import analyze_model_with_shap
    
    # Run complete SHAP analysis
    importance_df = analyze_model_with_shap(
        model=trained_model,
        X_train=X_train,
        X_val=X_val,
        feature_names=feature_names,
        output_dir='shap_analysis'
    )
    
    print(importance_df)
    """)