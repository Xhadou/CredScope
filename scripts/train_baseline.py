#!/usr/bin/env python
"""Train baseline model for CredScope"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.credscope.data.loader import DataLoader
from src.credscope.models.baseline import BaselineModel
from src.credscope.utils.config import load_config, setup_paths
import mlflow
import warnings
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def main():
    """Main training function"""
    print("🚀 Starting CredScope Phase 1: Baseline Model Training\n")
    print("=" * 60)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    setup_paths(config)
    
    # Initialize MLflow
    logger.info("Initializing MLflow...")
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # Load data
    print("\n📁 Loading data...")
    print("-" * 40)
    loader = DataLoader(config['data']['raw_path'])
    
    try:
        df_train = loader.load_application_data(train=True)
        print(f"✓ Data loaded successfully: {df_train.shape}")
        print(f"✓ Memory usage: {df_train.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Basic data checks
        print(f"\n📊 Data Overview:")
        print(f"  - Missing values: {df_train.isnull().sum().sum():,}")
        print(f"  - Default rate: {df_train['TARGET'].mean():.3%}")
        print(f"  - Features: {df_train.shape[1] - 2} (excluding TARGET and SK_ID_CURR)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please ensure you have downloaded the data:")
        print("   1. Go to https://www.kaggle.com/c/home-credit-default-risk/data")
        print("   2. Download all files to data/raw/")
        print("   3. Extract if necessary")
        return
    
    # Split data
    print("\n✂️ Splitting data...")
    print("-" * 40)
    (X_train, y_train), (X_val, y_val) = loader.get_train_test_split(
        df_train, 
        test_size=config['model']['test_size']
    )
    
    # Train baseline model
    print("\n🎯 Training baseline model...")
    print("-" * 40)
    baseline = BaselineModel(config)
    
    try:
        model, val_auc = baseline.train(X_train, y_train, X_val, y_val)
        
        # Success summary
        print("\n" + "=" * 60)
        print("✅ PHASE 1 COMPLETE!")
        print("=" * 60)
        print(f"\n📈 Results Summary:")
        print(f"  - Baseline AUC: {val_auc:.4f}")
        print(f"  - Model saved to: models/baseline_model.pkl")
        print(f"  - MLflow experiment: {config['mlflow']['experiment_name']}")
        
        print(f"\n💡 Next Steps:")
        print(f"  1. View results: mlflow ui --host 0.0.0.0")
        print(f"  2. Open browser: http://localhost:5000")
        print(f"  3. Continue to Phase 2: Feature Engineering")
        
        # Performance benchmark
        if val_auc < 0.70:
            print(f"\n⚠️ Warning: AUC ({val_auc:.4f}) is below expected range (0.74-0.76)")
            print("  Consider checking data loading and feature engineering")
        elif val_auc >= 0.74:
            print(f"\n🎉 Excellent! AUC ({val_auc:.4f}) meets expectations!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        print(f"\n❌ Training failed: {e}")
        print("\n💡 Common issues:")
        print("  - Missing required packages (pip install -r requirements.txt)")
        print("  - Insufficient memory (try reducing data size)")
        print("  - MLflow issues (check mlruns directory permissions)")
        raise

if __name__ == "__main__":
    main()