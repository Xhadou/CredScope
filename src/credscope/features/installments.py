"""Installments Payment Features

This module creates features from payment history showing:
- Payment punctuality (early/late patterns)
- Payment amount consistency
- Payment behavior trends over time

These are STRONG predictors of default risk.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class InstallmentsFeatureEngineer:
    """Engineer features from installments_payments table"""
    
    def transform(self, installments: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from installments payments
        
        Args:
            installments: Installments payments dataframe
            
        Returns:
            DataFrame with aggregated payment behavior features
        """
        logger.info("Engineering installments payment features...")
        
        if installments is None or installments.empty:
            logger.warning("Installments data is empty")
            return pd.DataFrame()
        
        # ===== DERIVED COLUMNS =====
        
        # Days past due (DPD) - positive means late
        installments['INST_DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
        
        # Payment difference (actual - expected)
        installments['INST_PAYMENT_DIFF'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
        
        # Payment ratio (actual / expected)
        installments['INST_PAYMENT_RATIO'] = installments['AMT_PAYMENT'] / (installments['AMT_INSTALMENT'] + 1)
        
        # Binary flags
        installments['INST_PAID_LATE'] = (installments['INST_DPD'] > 0).astype(int)
        installments['INST_PAID_EARLY'] = (installments['INST_DPD'] < 0).astype(int)
        installments['INST_PAID_ON_TIME'] = (installments['INST_DPD'] == 0).astype(int)
        
        installments['INST_UNDERPAID'] = (installments['INST_PAYMENT_DIFF'] < -1).astype(int)
        installments['INST_OVERPAID'] = (installments['INST_PAYMENT_DIFF'] > 1).astype(int)
        installments['INST_EXACT_PAYMENT'] = (installments['INST_PAYMENT_DIFF'].abs() <= 1).astype(int)
        
        # Severity of lateness
        installments['INST_DPD_SEVERITY'] = installments['INST_DPD'].apply(
            lambda x: 0 if x <= 0 else (1 if x <= 5 else (2 if x <= 15 else 3))
        )
        
        # ===== AGGREGATIONS =====
        
        agg_dict = {
            # Days past due statistics
            'INST_DPD': ['mean', 'max', 'min', 'std', 'sum'],
            
            # Payment differences
            'INST_PAYMENT_DIFF': ['mean', 'max', 'min', 'std', 'sum'],
            'INST_PAYMENT_RATIO': ['mean', 'max', 'min', 'std'],
            
            # Binary flags - count and rate
            'INST_PAID_LATE': ['sum', 'mean', 'max'],
            'INST_PAID_EARLY': ['sum', 'mean'],
            'INST_PAID_ON_TIME': ['sum', 'mean'],
            'INST_UNDERPAID': ['sum', 'mean'],
            'INST_OVERPAID': ['sum', 'mean'],
            'INST_EXACT_PAYMENT': ['sum', 'mean'],
            
            # Original amounts
            'AMT_INSTALMENT': ['sum', 'mean', 'max', 'min', 'std'],
            'AMT_PAYMENT': ['sum', 'mean', 'max', 'min', 'std'],
            
            # Counting
            'NUM_INSTALMENT_VERSION': ['max', 'nunique'],
            'NUM_INSTALMENT_NUMBER': ['max', 'count'],
        }
        
        inst_agg = installments.groupby('SK_ID_CURR').agg(agg_dict)  # type: ignore[call-overload]
        inst_agg.columns = ['INST_' + '_'.join(col).upper() for col in inst_agg.columns]
        inst_agg.reset_index(inplace=True)
        
        # ===== DERIVED FEATURES =====
        
        # 1. Late payment rate (most important!)
        if 'INST_INST_PAID_LATE_MEAN' in inst_agg.columns:
            inst_agg['INST_LATE_PAYMENT_RATE'] = inst_agg['INST_INST_PAID_LATE_MEAN']
        
        # 2. Severe lateness rate (>15 days)
        if 'INST_INST_DPD_MAX' in inst_agg.columns:
            inst_agg['INST_SEVERE_LATE'] = (inst_agg['INST_INST_DPD_MAX'] > 15).astype(int)
        
        # 3. Payment consistency (lower CV = more consistent)
        if all(col in inst_agg.columns for col in ['INST_AMT_PAYMENT_STD', 'INST_AMT_PAYMENT_MEAN']):
            inst_agg['INST_PAYMENT_CONSISTENCY'] = (
                inst_agg['INST_AMT_PAYMENT_STD'] / (inst_agg['INST_AMT_PAYMENT_MEAN'] + 1)
            )
        
        # 4. Average days early/late
        if 'INST_INST_DPD_MEAN' in inst_agg.columns:
            inst_agg['INST_AVG_DAYS_LATE'] = inst_agg['INST_INST_DPD_MEAN'].clip(lower=0)
            inst_agg['INST_AVG_DAYS_EARLY'] = (-inst_agg['INST_INST_DPD_MEAN']).clip(lower=0)
        
        # 5. Total amount underpaid
        if 'INST_INST_PAYMENT_DIFF_SUM' in inst_agg.columns:
            inst_agg['INST_TOTAL_UNDERPAID'] = (-inst_agg['INST_INST_PAYMENT_DIFF_SUM']).clip(lower=0)
        
        # 6. Payment completion rate
        if all(col in inst_agg.columns for col in ['INST_AMT_PAYMENT_SUM', 'INST_AMT_INSTALMENT_SUM']):
            inst_agg['INST_PAYMENT_COMPLETION_RATE'] = (
                inst_agg['INST_AMT_PAYMENT_SUM'] / (inst_agg['INST_AMT_INSTALMENT_SUM'] + 1)
            )
        
        # 7. Number of versions (payment plan changes - instability indicator)
        if 'INST_NUM_INSTALMENT_VERSION_MAX' in inst_agg.columns:
            inst_agg['INST_PAYMENT_PLAN_CHANGES'] = inst_agg['INST_NUM_INSTALMENT_VERSION_MAX']
        
        # 8. Has ever been late
        if 'INST_INST_PAID_LATE_SUM' in inst_agg.columns:
            inst_agg['INST_HAS_BEEN_LATE'] = (inst_agg['INST_INST_PAID_LATE_SUM'] > 0).astype(int)
        
        # ===== RECENT vs OLD BEHAVIOR =====
        
        # Get recent payments (last 12 installments)
        installments_sorted = installments.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'], ascending=[True, False])
        recent = installments_sorted.groupby('SK_ID_CURR').head(12)
        
        recent_agg = recent.groupby('SK_ID_CURR').agg({
            'INST_PAID_LATE': 'mean',
            'INST_DPD': 'mean',
            'INST_PAYMENT_RATIO': 'mean'
        })
        recent_agg.columns = ['INST_RECENT_' + col.upper() for col in recent_agg.columns]
        
        # Merge recent stats
        inst_agg = inst_agg.merge(recent_agg, on='SK_ID_CURR', how='left')
        
        # Trend: Recent vs Overall
        if all(col in inst_agg.columns for col in ['INST_RECENT_INST_PAID_LATE', 'INST_LATE_PAYMENT_RATE']):
            inst_agg['INST_LATE_PAYMENT_TREND'] = (
                inst_agg['INST_RECENT_INST_PAID_LATE'] - inst_agg['INST_LATE_PAYMENT_RATE']
            )
            inst_agg['INST_GETTING_WORSE'] = (inst_agg['INST_LATE_PAYMENT_TREND'] > 0.1).astype(int)
        
        logger.info(f"✓ Created {len(inst_agg.columns) - 1} installments features")
        
        return inst_agg


def create_installments_features(installments: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to create installments features
    
    Args:
        installments: Installments payments dataframe
        
    Returns:
        DataFrame with aggregated features
    """
    engineer = InstallmentsFeatureEngineer()
    return engineer.transform(installments)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("Installments Feature Engineer")
    print("=" * 50)
    print("Creates features from payment behavior:")
    print("  - Payment punctuality (late/early/on-time)")
    print("  - Payment amount consistency")
    print("  - Payment trends over time")
    print("\nUsage:")
    print("""
from credscope.features.installments import create_installments_features

installments = loader.load_installments_payments()
inst_features = create_installments_features(installments)
    """)