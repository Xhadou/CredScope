"""Geographic and Regional Risk Features

This module creates features based on geographic indicators and regional patterns
that are strong predictors of credit risk in alternative credit scoring.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class GeographicFeatureEngineer:
    """Engineer geographic risk features from application data"""
    
    def __init__(self):
        self.regional_stats = {}
        self.city_stats = {}
        
    def fit(self, df: pd.DataFrame, target: str = 'TARGET') -> 'GeographicFeatureEngineer':
        """Calculate regional statistics from training data
        
        Args:
            df: Training dataframe with geographic columns
            target: Target column name
            
        Returns:
            self for chaining
        """
        logger.info("Calculating regional statistics...")
        
        # Regional default rates
        if 'REGION_RATING_CLIENT' in df.columns:
            regional_agg = df.groupby('REGION_RATING_CLIENT').agg({
                target: ['mean', 'std', 'count']
            })[target]
            self.regional_stats = regional_agg.to_dict('index')  # type: ignore[call-overload]
            
        # City ratings
        if 'REGION_RATING_CLIENT_W_CITY' in df.columns:
            city_agg = df.groupby('REGION_RATING_CLIENT_W_CITY').agg({
                target: ['mean', 'std', 'count']
            })[target]
            self.city_stats = city_agg.to_dict('index')  # type: ignore[call-overload]
            
        logger.info(f"✓ Regional statistics calculated for {len(self.regional_stats)} regions")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate geographic features
        
        Args:
            df: Dataframe to transform
            
        Returns:
            DataFrame with new geographic features
        """
        logger.info("Engineering geographic features...")
        df = df.copy()
        
        # === REGION-BASED FEATURES ===
        
        # 1. Region risk score (weighted combination)
        if 'REGION_RATING_CLIENT' in df.columns:
            df['GEO_REGION_RISK_SCORE'] = df['REGION_RATING_CLIENT']
            
        if 'REGION_RATING_CLIENT_W_CITY' in df.columns:
            df['GEO_CITY_RISK_SCORE'] = df['REGION_RATING_CLIENT_W_CITY']
            
        # 2. Combined geographic risk
        if all(col in df.columns for col in ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']):
            df['GEO_COMBINED_RISK'] = (
                0.6 * df['REGION_RATING_CLIENT'] + 
                0.4 * df['REGION_RATING_CLIENT_W_CITY']
            )
            
        # 3. Region-City consistency
        if all(col in df.columns for col in ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']):
            df['GEO_REGION_CITY_DIFF'] = (
                df['REGION_RATING_CLIENT'] - df['REGION_RATING_CLIENT_W_CITY']
            ).abs()
            df['GEO_REGION_CITY_MATCH'] = (
                df['REGION_RATING_CLIENT'] == df['REGION_RATING_CLIENT_W_CITY']
            ).astype(int)
        
        # === POPULATION DENSITY FEATURES ===
        
        # 4. Population density indicators
        population_cols = [
            'REGION_POPULATION_RELATIVE',
            'POPULATION_REGIONAL',  
        ]
        
        for col in population_cols:
            if col in df.columns:
                # Log transform to handle skewness
                df[f'GEO_{col}_LOG'] = np.log1p(df[col].fillna(0))
                
                # Binned categories
                df[f'GEO_{col}_CATEGORY'] = pd.qcut(
                    df[col].fillna(0), 
                    q=5, 
                    labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                    duplicates='drop'
                )
        
        # === LIVING AREA FEATURES ===
        
        # 5. Average living area features
        living_area_cols = ['LIVINGAREA_AVG', 'LIVINGAREA_MEDI', 'LIVINGAREA_MODE']
        if all(col in df.columns for col in living_area_cols):
            # Mean of different area measures
            df['GEO_LIVING_AREA_MEAN'] = df[living_area_cols].mean(axis=1)
            
            # Standard deviation (consistency)
            df['GEO_LIVING_AREA_STD'] = df[living_area_cols].std(axis=1)
            
            # Coefficient of variation
            df['GEO_LIVING_AREA_CV'] = (
                df['GEO_LIVING_AREA_STD'] / (df['GEO_LIVING_AREA_MEAN'] + 1e-6)
            )
        
        # === BUILDING CHARACTERISTICS ===
        
        # 6. Building age features
        building_cols = ['YEARS_BUILD_AVG', 'YEARS_BUILD_MEDI', 'YEARS_BUILD_MODE']
        if all(col in df.columns for col in building_cols):
            df['GEO_BUILDING_AGE_MEAN'] = df[building_cols].mean(axis=1)
            df['GEO_BUILDING_AGE_STD'] = df[building_cols].std(axis=1)
            
            # Old vs new building binary
            df['GEO_BUILDING_OLD'] = (df['GEO_BUILDING_AGE_MEAN'] > 0.5).astype(int)
        
        # 7. Apartment features
        apartment_cols = ['APARTMENTS_AVG', 'APARTMENTS_MEDI', 'APARTMENTS_MODE']
        if all(col in df.columns for col in apartment_cols):
            df['GEO_APARTMENTS_MEAN'] = df[apartment_cols].mean(axis=1)
            df['GEO_APARTMENTS_STD'] = df[apartment_cols].std(axis=1)
        
        # === INFRASTRUCTURE QUALITY ===
        
        # 8. Common area features (indicator of building quality)
        common_area_cols = ['COMMONAREA_AVG', 'COMMONAREA_MEDI', 'COMMONAREA_MODE']
        if all(col in df.columns for col in common_area_cols):
            df['GEO_COMMON_AREA_MEAN'] = df[common_area_cols].mean(axis=1)
            
            # Has common area (binary)
            df['GEO_HAS_COMMON_AREA'] = (df['GEO_COMMON_AREA_MEAN'] > 0).astype(int)
        
        # 9. Basement features
        basement_cols = ['BASEMENTAREA_AVG', 'BASEMENTAREA_MEDI', 'BASEMENTAREA_MODE']
        if all(col in df.columns for col in basement_cols):
            df['GEO_BASEMENT_MEAN'] = df[basement_cols].mean(axis=1)
            df['GEO_HAS_BASEMENT'] = (df['GEO_BASEMENT_MEAN'] > 0).astype(int)
        
        # === GEOGRAPHIC ISOLATION FEATURES ===
        
        # 10. Distance-based features (if available)
        if 'HOUR_APPR_PROCESS_START' in df.columns:
            # Application time can indicate location (time zones, work hours)
            df['GEO_APP_HOUR_CATEGORY'] = pd.cut(
                df['HOUR_APPR_PROCESS_START'],
                bins=[0, 9, 17, 24],
                labels=['early', 'business', 'evening']
            )
        
        # Count new features created
        new_cols = [col for col in df.columns if col.startswith('GEO_')]
        logger.info(f"✓ Created {len(new_cols)} geographic features")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target: str = 'TARGET') -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df, target).transform(df)


def create_geographic_features(df: pd.DataFrame, 
                               target: str = 'TARGET',
                               is_train: bool = True) -> pd.DataFrame:
    """Convenience function to create geographic features
    
    Args:
        df: Input dataframe
        target: Target column name (only used if is_train=True)
        is_train: Whether this is training data
        
    Returns:
        DataFrame with geographic features
    """
    engineer = GeographicFeatureEngineer()
    
    if is_train:
        return engineer.fit_transform(df, target)
    else:
        return engineer.transform(df)


if __name__ == "__main__":
    # Test the feature engineering
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("Geographic Feature Engineer Test")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'REGION_RATING_CLIENT': [1, 2, 3, 1, 2],
        'REGION_RATING_CLIENT_W_CITY': [1, 2, 2, 1, 3],
        'REGION_POPULATION_RELATIVE': [0.01, 0.05, 0.02, 0.01, 0.08],
        'LIVINGAREA_AVG': [0.05, 0.08, 0.06, 0.05, 0.09],
        'LIVINGAREA_MEDI': [0.05, 0.08, 0.06, 0.05, 0.09],
        'LIVINGAREA_MODE': [0.05, 0.08, 0.06, 0.05, 0.09],
        'TARGET': [0, 1, 0, 0, 1]
    })
    
    engineer = GeographicFeatureEngineer()
    result = engineer.fit_transform(sample_data)
    
    # Show new features
    geo_features = [col for col in result.columns if col.startswith('GEO_')]
    print(f"\nCreated {len(geo_features)} features:")
    for feat in geo_features:
        print(f"  - {feat}")
    
    print(f"\nSample output:\n{result[geo_features].head()}")