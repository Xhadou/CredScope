"""Credit Card Balance Features

This module creates features from credit card usage showing:
- Credit utilization patterns
- Spending behavior
- Cash advance usage
- Payment discipline on credit cards
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CreditCardFeatureEngineer:
    """Engineer features from credit_card_balance table"""
    
    def transform(self, cc_balance: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from credit card balance
        
        Args:
            cc_balance: Credit card balance dataframe
            
        Returns:
            DataFrame with aggregated credit card features
        """
        logger.info("Engineering credit card balance features...")
        
        if cc_balance is None or cc_balance.empty:
            logger.warning("Credit card balance data is empty")
            return pd.DataFrame()
        
        # ===== DERIVED COLUMNS =====
        
        # Credit utilization rate
        cc_balance['CC_UTILIZATION'] = cc_balance['AMT_BALANCE'] / (cc_balance['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
        
        # Drawings ratios
        cc_balance['CC_DRAWING_RATIO'] = cc_balance['AMT_DRAWINGS_CURRENT'] / (cc_balance['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
        cc_balance['CC_CASH_DRAWING_RATIO'] = cc_balance['AMT_DRAWINGS_ATM_CURRENT'] / (cc_balance['AMT_DRAWINGS_CURRENT'] + 1)
        cc_balance['CC_POS_DRAWING_RATIO'] = cc_balance['AMT_DRAWINGS_POS_CURRENT'] / (cc_balance['AMT_DRAWINGS_CURRENT'] + 1)
        
        # Payment ratios
        cc_balance['CC_PAYMENT_RATIO'] = cc_balance['AMT_PAYMENT_CURRENT'] / (cc_balance['AMT_INST_MIN_REGULARITY'] + 1)
        cc_balance['CC_RECEIVABLE_RATIO'] = cc_balance['AMT_RECEIVABLE_PRINCIPAL'] / (cc_balance['AMT_BALANCE'] + 1)
        
        # Binary flags
        cc_balance['CC_OVER_LIMIT'] = (cc_balance['CC_UTILIZATION'] > 1.0).astype(int)
        cc_balance['CC_FULL_BALANCE'] = (cc_balance['AMT_BALANCE'] <= 0).astype(int)
        cc_balance['CC_USING_CASH_ADVANCE'] = (cc_balance['AMT_DRAWINGS_ATM_CURRENT'] > 0).astype(int)
        cc_balance['CC_UNDERPAYING'] = (cc_balance['CC_PAYMENT_RATIO'] < 1.0).astype(int)
        
        # DPD severity
        cc_balance['CC_DPD_SEVERITY'] = cc_balance['SK_DPD'].apply(
            lambda x: 0 if x == 0 else (1 if x <= 5 else (2 if x <= 30 else 3))
        )
        
        # ===== AGGREGATIONS =====
        
        agg_dict = {
            # Balances
            'AMT_BALANCE': ['mean', 'max', 'min', 'std', 'sum'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max', 'min'],
            
            # Utilization
            'CC_UTILIZATION': ['mean', 'max', 'min', 'std'],
            
            # Drawings
            'AMT_DRAWINGS_CURRENT': ['mean', 'max', 'sum', 'std'],
            'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['mean', 'max', 'sum'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['mean', 'max', 'sum'],
            
            'CC_DRAWING_RATIO': ['mean', 'max'],
            'CC_CASH_DRAWING_RATIO': ['mean', 'max'],
            'CC_POS_DRAWING_RATIO': ['mean', 'max'],
            
            # Payments
            'AMT_PAYMENT_CURRENT': ['mean', 'max', 'sum', 'std'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'max', 'sum'],
            'AMT_INST_MIN_REGULARITY': ['mean', 'max', 'sum'],
            
            'CC_PAYMENT_RATIO': ['mean', 'min', 'max'],
            
            # Receivables
            'AMT_RECEIVABLE_PRINCIPAL': ['mean', 'max', 'sum'],
            'AMT_RECIVABLE': ['mean', 'max', 'sum'],
            'AMT_TOTAL_RECEIVABLE': ['mean', 'max', 'sum'],
            
            'CC_RECEIVABLE_RATIO': ['mean', 'max'],
            
            # DPD (Days Past Due)
            'SK_DPD': ['mean', 'max', 'sum'],
            'SK_DPD_DEF': ['mean', 'max', 'sum'],
            'CC_DPD_SEVERITY': ['mean', 'max'],
            
            # Binary flags
            'CC_OVER_LIMIT': ['sum', 'mean', 'max'],
            'CC_FULL_BALANCE': ['sum', 'mean'],
            'CC_USING_CASH_ADVANCE': ['sum', 'mean'],
            'CC_UNDERPAYING': ['sum', 'mean'],
            
            # Counting
            'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'sum', 'max'],
            'CNT_DRAWINGS_CURRENT': ['mean', 'sum', 'max'],
            'CNT_DRAWINGS_POS_CURRENT': ['mean', 'sum', 'max'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['mean', 'sum', 'max'],
            'CNT_INSTALMENT_MATURE_CUM': ['mean', 'max', 'sum'],
            
            # Record count
            'SK_ID_PREV': ['count'],
        }
        
        cc_agg = cc_balance.groupby('SK_ID_CURR').agg(agg_dict)  # type: ignore[call-overload]
        cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
        cc_agg.reset_index(inplace=True)
        
        # ===== DERIVED FEATURES =====
        
        # 1. Average credit utilization (key feature!)
        if 'CC_CC_UTILIZATION_MEAN' in cc_agg.columns:
            cc_agg['CC_AVG_UTILIZATION'] = cc_agg['CC_CC_UTILIZATION_MEAN']
            cc_agg['CC_HIGH_UTILIZATION'] = (cc_agg['CC_AVG_UTILIZATION'] > 0.8).astype(int)
        
        # 2. Cash advance dependency
        if all(col in cc_agg.columns for col in ['CC_AMT_DRAWINGS_ATM_CURRENT_SUM', 'CC_AMT_DRAWINGS_CURRENT_SUM']):
            cc_agg['CC_CASH_ADVANCE_DEPENDENCY'] = (
                cc_agg['CC_AMT_DRAWINGS_ATM_CURRENT_SUM'] / (cc_agg['CC_AMT_DRAWINGS_CURRENT_SUM'] + 1)
            )
        
        # 3. Payment discipline
        if 'CC_CC_PAYMENT_RATIO_MEAN' in cc_agg.columns:
            cc_agg['CC_GOOD_PAYER'] = (cc_agg['CC_CC_PAYMENT_RATIO_MEAN'] >= 1.0).astype(int)
            cc_agg['CC_MIN_PAYER'] = (
                (cc_agg['CC_CC_PAYMENT_RATIO_MEAN'] < 1.0) & 
                (cc_agg['CC_CC_PAYMENT_RATIO_MEAN'] >= 0.9)
            ).astype(int)
        
        # 4. Credit line usage stability
        if 'CC_CC_UTILIZATION_STD' in cc_agg.columns:
            cc_agg['CC_UTILIZATION_STABILITY'] = 1 / (cc_agg['CC_CC_UTILIZATION_STD'] + 0.01)
        
        # 5. Has DPD history
        if 'CC_SK_DPD_SUM' in cc_agg.columns:
            cc_agg['CC_HAS_DPD_HISTORY'] = (cc_agg['CC_SK_DPD_SUM'] > 0).astype(int)
        
        # 6. Severe DPD flag
        if 'CC_SK_DPD_MAX' in cc_agg.columns:
            cc_agg['CC_SEVERE_DPD'] = (cc_agg['CC_SK_DPD_MAX'] > 30).astype(int)
        
        # 7. Over-limit frequency
        if 'CC_CC_OVER_LIMIT_MEAN' in cc_agg.columns:
            cc_agg['CC_FREQUENT_OVER_LIMIT'] = (cc_agg['CC_CC_OVER_LIMIT_MEAN'] > 0.2).astype(int)
        
        # 8. Drawing to payment ratio
        if all(col in cc_agg.columns for col in ['CC_AMT_DRAWINGS_CURRENT_SUM', 'CC_AMT_PAYMENT_CURRENT_SUM']):
            cc_agg['CC_DRAWING_TO_PAYMENT_RATIO'] = (
                cc_agg['CC_AMT_DRAWINGS_CURRENT_SUM'] / (cc_agg['CC_AMT_PAYMENT_CURRENT_SUM'] + 1)
            )
        
        # 9. Number of active cards
        if 'CC_SK_ID_PREV_COUNT' in cc_agg.columns:
            cc_agg['CC_NUM_ACTIVE_CARDS'] = cc_agg['CC_SK_ID_PREV_COUNT']
        
        # 10. Total credit available
        if 'CC_AMT_CREDIT_LIMIT_ACTUAL_SUM' in cc_agg.columns:
            cc_agg['CC_TOTAL_CREDIT_LIMIT'] = cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_SUM']
        
        # ===== RECENT BEHAVIOR =====
        
        # Get recent months (last 6 months)
        cc_recent = cc_balance[cc_balance['MONTHS_BALANCE'] >= -6]
        
        if not cc_recent.empty:
            recent_agg = cc_recent.groupby('SK_ID_CURR').agg({
                'CC_UTILIZATION': 'mean',
                'SK_DPD': 'max',
                'CC_USING_CASH_ADVANCE': 'mean',
                'CC_UNDERPAYING': 'mean'
            })
            recent_agg.columns = ['CC_RECENT_' + col.upper() for col in recent_agg.columns]
            
            # Merge
            cc_agg = cc_agg.merge(recent_agg, on='SK_ID_CURR', how='left')
            
            # Trend analysis
            if all(col in cc_agg.columns for col in ['CC_RECENT_CC_UTILIZATION', 'CC_AVG_UTILIZATION']):
                cc_agg['CC_UTILIZATION_TREND'] = (
                    cc_agg['CC_RECENT_CC_UTILIZATION'] - cc_agg['CC_AVG_UTILIZATION']
                )
                cc_agg['CC_UTILIZATION_INCREASING'] = (cc_agg['CC_UTILIZATION_TREND'] > 0.1).astype(int)
        
        # ===== TREND ANALYSIS: Recent vs Historical =====
        
        # Compare last 3 months vs overall average
        cc_very_recent = cc_balance[cc_balance['MONTHS_BALANCE'] >= -3]
        
        if not cc_very_recent.empty:
            very_recent_agg = cc_very_recent.groupby('SK_ID_CURR').agg({
                'CC_UTILIZATION': 'mean',
                'CC_PAYMENT_RATIO': 'mean',
                'SK_DPD': 'max'
            })
            very_recent_agg.columns = ['CC_VERY_RECENT_' + col.upper() for col in very_recent_agg.columns]
            
            # Merge
            cc_agg = cc_agg.merge(very_recent_agg, on='SK_ID_CURR', how='left')
        
        logger.info(f"✓ Created {len(cc_agg.columns) - 1} credit card features")
        
        return cc_agg


def create_creditcard_features(cc_balance: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to create credit card features
    
    Args:
        cc_balance: Credit card balance dataframe
        
    Returns:
        DataFrame with aggregated features
    """
    engineer = CreditCardFeatureEngineer()
    return engineer.transform(cc_balance)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("Credit Card Feature Engineer")
    print("=" * 50)
    print("Creates features from credit card behavior:")
    print("  - Credit utilization patterns")
    print("  - Cash advance usage")
    print("  - Payment discipline")
    print("  - DPD history")
    print("\nUsage:")
    print("""
from credscope.features.creditcard import create_creditcard_features

cc_balance = loader.load_credit_card_balance()
cc_features = create_creditcard_features(cc_balance)
    """)