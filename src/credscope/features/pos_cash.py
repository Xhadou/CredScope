"""POS and Cash Balance Features

This module creates features from point-of-sale credit and cash loans:
- Short-term credit behavior
- Installment payment patterns
- Contract status history
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class POSCashFeatureEngineer:
    """Engineer features from POS_CASH_balance table"""
    
    def transform(self, pos_cash: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from POS/Cash balance
        
        Args:
            pos_cash: POS/Cash balance dataframe
            
        Returns:
            DataFrame with aggregated features
        """
        logger.info("Engineering POS/Cash balance features...")
        
        if pos_cash is None or pos_cash.empty:
            logger.warning("POS/Cash balance data is empty")
            return pd.DataFrame()
        
        # ===== DERIVED COLUMNS =====
        
        # Binary flags for contract status
        pos_cash['POS_STATUS_ACTIVE'] = (pos_cash['NAME_CONTRACT_STATUS'] == 'Active').astype(int)
        pos_cash['POS_STATUS_COMPLETED'] = (pos_cash['NAME_CONTRACT_STATUS'] == 'Completed').astype(int)
        pos_cash['POS_STATUS_AMORT_DEBT'] = (pos_cash['NAME_CONTRACT_STATUS'] == 'Amortized debt').astype(int)
        pos_cash['POS_STATUS_CANCELLED'] = (pos_cash['NAME_CONTRACT_STATUS'] == 'Canceled').astype(int)
        pos_cash['POS_STATUS_APPROVED'] = (pos_cash['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
        pos_cash['POS_STATUS_DEMAND'] = (pos_cash['NAME_CONTRACT_STATUS'] == 'Demand').astype(int)
        pos_cash['POS_STATUS_SIGNED'] = (pos_cash['NAME_CONTRACT_STATUS'] == 'Signed').astype(int)
        pos_cash['POS_STATUS_XNA'] = (pos_cash['NAME_CONTRACT_STATUS'] == 'XNA').astype(int)
        
        # DPD indicators
        pos_cash['POS_HAS_DPD'] = (pos_cash['SK_DPD'] > 0).astype(int)
        pos_cash['POS_HAS_DPD_DEF'] = (pos_cash['SK_DPD_DEF'] > 0).astype(int)
        
        # Installment completion ratio
        pos_cash['POS_INSTALLMENT_RATIO'] = (
            pos_cash['CNT_INSTALMENT'] / (pos_cash['CNT_INSTALMENT_FUTURE'] + pos_cash['CNT_INSTALMENT'] + 1)
        )
        
        # ===== AGGREGATIONS =====
        
        agg_dict = {
            # DPD metrics
            'SK_DPD': ['mean', 'max', 'sum', 'std'],
            'SK_DPD_DEF': ['mean', 'max', 'sum'],
            'POS_HAS_DPD': ['sum', 'mean', 'max'],
            'POS_HAS_DPD_DEF': ['sum', 'mean', 'max'],
            
            # Installment counts
            'CNT_INSTALMENT': ['mean', 'max', 'sum'],
            'CNT_INSTALMENT_FUTURE': ['mean', 'max', 'sum'],
            'POS_INSTALLMENT_RATIO': ['mean', 'min', 'max'],
            
            # Contract statuses
            'POS_STATUS_ACTIVE': ['sum', 'mean'],
            'POS_STATUS_COMPLETED': ['sum', 'mean'],
            'POS_STATUS_AMORT_DEBT': ['sum', 'mean'],
            'POS_STATUS_CANCELLED': ['sum', 'mean'],
            'POS_STATUS_DEMAND': ['sum', 'mean'],
            
            # Months balance
            'MONTHS_BALANCE': ['mean', 'max', 'min', 'count'],
            
            # Counting
            'SK_ID_PREV': ['count', 'nunique'],
        }
        
        pos_agg = pos_cash.groupby('SK_ID_CURR').agg(agg_dict)  # type: ignore[call-overload]
        pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
        pos_agg.reset_index(inplace=True)
        
        # ===== DERIVED FEATURES =====
        
        # 1. Has DPD history
        if 'POS_SK_DPD_SUM' in pos_agg.columns:
            pos_agg['POS_HAS_ANY_DPD'] = (pos_agg['POS_SK_DPD_SUM'] > 0).astype(int)
        
        # 2. Severe DPD (>30 days)
        if 'POS_SK_DPD_MAX' in pos_agg.columns:
            pos_agg['POS_SEVERE_DPD'] = (pos_agg['POS_SK_DPD_MAX'] > 30).astype(int)
        
        # 3. DPD rate
        if all(col in pos_agg.columns for col in ['POS_SK_DPD_SUM', 'POS_MONTHS_BALANCE_COUNT']):
            pos_agg['POS_DPD_RATE'] = pos_agg['POS_SK_DPD_SUM'] / (pos_agg['POS_MONTHS_BALANCE_COUNT'] + 1)
        
        # 4. Completion rate
        if 'POS_POS_STATUS_COMPLETED_MEAN' in pos_agg.columns:
            pos_agg['POS_GOOD_COMPLETION_RATE'] = pos_agg['POS_POS_STATUS_COMPLETED_MEAN']
        
        # 5. Cancellation rate
        if 'POS_POS_STATUS_CANCELLED_MEAN' in pos_agg.columns:
            pos_agg['POS_CANCELLATION_RATE'] = pos_agg['POS_POS_STATUS_CANCELLED_MEAN']
            pos_agg['POS_HAS_CANCELLATIONS'] = (pos_agg['POS_POS_STATUS_CANCELLED_SUM'] > 0).astype(int)
        
        # 6. Active loans count
        if 'POS_POS_STATUS_ACTIVE_SUM' in pos_agg.columns:
            pos_agg['POS_NUM_ACTIVE_LOANS'] = pos_agg['POS_POS_STATUS_ACTIVE_SUM']
        
        # 7. Total completed loans
        if 'POS_POS_STATUS_COMPLETED_SUM' in pos_agg.columns:
            pos_agg['POS_NUM_COMPLETED_LOANS'] = pos_agg['POS_POS_STATUS_COMPLETED_SUM']
        
        # 8. Demand/problem loans
        if 'POS_POS_STATUS_DEMAND_SUM' in pos_agg.columns:
            pos_agg['POS_HAS_DEMAND_STATUS'] = (pos_agg['POS_POS_STATUS_DEMAND_SUM'] > 0).astype(int)
        
        # 9. Average installment progress
        if 'POS_POS_INSTALLMENT_RATIO_MEAN' in pos_agg.columns:
            pos_agg['POS_AVG_COMPLETION'] = pos_agg['POS_POS_INSTALLMENT_RATIO_MEAN']
        
        # 10. Total number of POS/Cash contracts
        if 'POS_SK_ID_PREV_NUNIQUE' in pos_agg.columns:
            pos_agg['POS_TOTAL_CONTRACTS'] = pos_agg['POS_SK_ID_PREV_NUNIQUE']
        
        # ===== RECENT BEHAVIOR =====
        
        # Get recent months (last 12 months)
        pos_recent = pos_cash[pos_cash['MONTHS_BALANCE'] >= -12]
        
        if not pos_recent.empty:
            recent_agg = pos_recent.groupby('SK_ID_CURR').agg({
                'SK_DPD': ['mean', 'max'],
                'POS_STATUS_ACTIVE': 'sum',
                'POS_STATUS_COMPLETED': 'sum',
                'POS_HAS_DPD': 'mean'
            })
            recent_agg.columns = ['POS_RECENT_' + '_'.join(col).upper() for col in recent_agg.columns]
            
            # Merge
            pos_agg = pos_agg.merge(recent_agg, on='SK_ID_CURR', how='left')
            
            # Trend: Getting better or worse?
            if all(col in pos_agg.columns for col in ['POS_RECENT_SK_DPD_MEAN', 'POS_SK_DPD_MEAN']):
                pos_agg['POS_DPD_TREND'] = pos_agg['POS_RECENT_SK_DPD_MEAN'] - pos_agg['POS_SK_DPD_MEAN']
                pos_agg['POS_DPD_GETTING_WORSE'] = (pos_agg['POS_DPD_TREND'] > 1).astype(int)
        
        logger.info(f"✓ Created {len(pos_agg.columns) - 1} POS/Cash features")
        
        return pos_agg


def create_pos_cash_features(pos_cash: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to create POS/Cash features
    
    Args:
        pos_cash: POS/Cash balance dataframe
        
    Returns:
        DataFrame with aggregated features
    """
    engineer = POSCashFeatureEngineer()
    return engineer.transform(pos_cash)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("POS/Cash Feature Engineer")
    print("=" * 50)
    print("Creates features from POS and cash loan behavior:")
    print("  - Contract status history")
    print("  - DPD patterns")
    print("  - Completion rates")
    print("  - Active/cancelled loans")
    print("\nUsage:")
    print("""
from credscope.features.pos_cash import create_pos_cash_features

pos_cash = loader.load_pos_cash_balance()
pos_features = create_pos_cash_features(pos_cash)
    """)