"""Bureau Credit History Features

This module creates aggregated features from credit bureau data showing
historical credit behavior patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class BureauFeatureEngineer:
    """Engineer features from bureau and bureau_balance tables"""
    
    def __init__(self):
        self.aggregations = {}
        
    def transform_bureau(self, bureau: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from bureau table
        
        Args:
            bureau: Bureau dataframe
            
        Returns:
            DataFrame with aggregated features by SK_ID_CURR
        """
        logger.info("Engineering bureau credit history features...")
        
        if bureau is None or bureau.empty:
            logger.warning("Bureau data is empty")
            return pd.DataFrame()
        
        # Define aggregations for different feature types
        aggregations = {
            # Credit amounts
            'AMT_CREDIT_SUM': ['sum', 'mean', 'max', 'min', 'std'],
            'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max', 'std'],
            'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
            
            # Credit duration
            'DAYS_CREDIT': ['min', 'max', 'mean', 'std'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
            
            # Counts
            'CREDIT_DAY_OVERDUE': ['max', 'mean', 'sum'],
            'CNT_CREDIT_PROLONG': ['sum', 'mean', 'max'],
            
            # Credit type (will count unique)
            'CREDIT_TYPE': ['count', 'nunique'],
        }
        
        # Apply aggregations
        bureau_agg = bureau.groupby('SK_ID_CURR').agg(aggregations)  # type: ignore[call-overload]
        
        # Flatten column names
        bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
        bureau_agg.reset_index(inplace=True)
        
        # === DERIVED FEATURES ===
        
        # 1. Credit utilization rate
        if all(col in bureau_agg.columns for col in ['BUREAU_AMT_CREDIT_SUM_DEBT_SUM', 
                                                       'BUREAU_AMT_CREDIT_SUM_LIMIT_SUM']):
            bureau_agg['BUREAU_CREDIT_UTILIZATION'] = (
                bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / 
                (bureau_agg['BUREAU_AMT_CREDIT_SUM_LIMIT_SUM'] + 1)
            )
        
        # 2. Overdue ratio
        if all(col in bureau_agg.columns for col in ['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM',
                                                       'BUREAU_AMT_CREDIT_SUM_SUM']):
            bureau_agg['BUREAU_OVERDUE_RATIO'] = (
                bureau_agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM'] / 
                (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1)
            )
        
        # 3. Debt to credit ratio
        if all(col in bureau_agg.columns for col in ['BUREAU_AMT_CREDIT_SUM_DEBT_SUM',
                                                       'BUREAU_AMT_CREDIT_SUM_SUM']):
            bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = (
                bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / 
                (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1)
            )
        
        # 4. Active credit lines
        bureau_agg['BUREAU_ACTIVE_CREDITS'] = bureau_agg['BUREAU_CREDIT_TYPE_COUNT']
        
        # 5. Credit diversity (more types = potentially better management)
        if 'BUREAU_CREDIT_TYPE_NUNIQUE' in bureau_agg.columns:
            bureau_agg['BUREAU_CREDIT_DIVERSITY'] = (
                bureau_agg['BUREAU_CREDIT_TYPE_NUNIQUE'] / 
                (bureau_agg['BUREAU_CREDIT_TYPE_COUNT'] + 1)
            )
        
        # 6. Average credit age (in years)
        if 'BUREAU_DAYS_CREDIT_MEAN' in bureau_agg.columns:
            bureau_agg['BUREAU_AVG_CREDIT_AGE_YEARS'] = (
                bureau_agg['BUREAU_DAYS_CREDIT_MEAN'] / -365
            )
        
        # 7. Credit recency (days since last credit update)
        if 'BUREAU_DAYS_CREDIT_UPDATE_MIN' in bureau_agg.columns:
            bureau_agg['BUREAU_CREDIT_RECENCY_DAYS'] = abs(
                bureau_agg['BUREAU_DAYS_CREDIT_UPDATE_MIN']
            )
        
        # 8. Prolongation rate (how often credits are extended)
        if all(col in bureau_agg.columns for col in ['BUREAU_CNT_CREDIT_PROLONG_SUM',
                                                       'BUREAU_CREDIT_TYPE_COUNT']):
            bureau_agg['BUREAU_PROLONG_RATE'] = (
                bureau_agg['BUREAU_CNT_CREDIT_PROLONG_SUM'] / 
                (bureau_agg['BUREAU_CREDIT_TYPE_COUNT'] + 1)
            )
        
        # 9. Has any overdue
        if 'BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX' in bureau_agg.columns:
            bureau_agg['BUREAU_HAS_OVERDUE'] = (
                bureau_agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX'] > 0
            ).astype(int)
        
        logger.info(f"✓ Created {len(bureau_agg.columns) - 1} bureau features")
        return bureau_agg
    
    def transform_bureau_balance(self, bureau: pd.DataFrame, 
                                 bureau_balance: pd.DataFrame) -> pd.DataFrame:
        """Create features from bureau balance (monthly status history)
        
        Args:
            bureau: Bureau dataframe
            bureau_balance: Bureau balance dataframe
            
        Returns:
            DataFrame with aggregated balance features
        """
        logger.info("Engineering bureau balance payment behavior features...")
        
        if bureau_balance is None or bureau_balance.empty:
            logger.warning("Bureau balance data is empty")
            return pd.DataFrame()
        
        # Merge to get SK_ID_CURR
        bb = bureau_balance.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], 
                                  on='SK_ID_BUREAU', how='left')
        
        # === STATUS ENCODING ===
        # Status codes: C=closed, X=status unknown, 0-5=DPD levels, etc.
        
        # Create binary flags for different statuses
        bb['BB_STATUS_C'] = (bb['STATUS'] == 'C').astype(int)  # Closed
        bb['BB_STATUS_X'] = (bb['STATUS'] == 'X').astype(int)  # Unknown
        bb['BB_STATUS_0'] = (bb['STATUS'] == '0').astype(int)  # Current
        bb['BB_STATUS_1'] = (bb['STATUS'] == '1').astype(int)  # 1-29 DPD
        bb['BB_STATUS_2'] = (bb['STATUS'] == '2').astype(int)  # 30-59 DPD
        bb['BB_STATUS_3'] = (bb['STATUS'] == '3').astype(int)  # 60-89 DPD
        bb['BB_STATUS_4'] = (bb['STATUS'] == '4').astype(int)  # 90-119 DPD
        bb['BB_STATUS_5'] = (bb['STATUS'] == '5').astype(int)  # 120+ DPD
        
        # Any delinquency flag
        bb['BB_STATUS_DPD'] = bb[['BB_STATUS_1', 'BB_STATUS_2', 'BB_STATUS_3', 
                                   'BB_STATUS_4', 'BB_STATUS_5']].max(axis=1)
        
        # === AGGREGATIONS ===
        
        agg_dict = {
            'MONTHS_BALANCE': ['min', 'max', 'size'],  # size = count of records
            'BB_STATUS_C': ['sum', 'mean'],
            'BB_STATUS_X': ['sum', 'mean'],
            'BB_STATUS_0': ['sum', 'mean'],
            'BB_STATUS_1': ['sum', 'mean'],
            'BB_STATUS_2': ['sum', 'mean'],
            'BB_STATUS_3': ['sum', 'mean'],
            'BB_STATUS_4': ['sum', 'mean'],
            'BB_STATUS_5': ['sum', 'mean'],
            'BB_STATUS_DPD': ['sum', 'mean', 'max'],
        }
        
        bb_agg = bb.groupby('SK_ID_CURR').agg(agg_dict)  # type: ignore[call-overload]
        bb_agg.columns = ['BB_' + '_'.join(col).upper() for col in bb_agg.columns]
        bb_agg.reset_index(inplace=True)
        
        # === DERIVED FEATURES ===
        
        # 1. Payment consistency score (higher = more consistent)
        if 'BB_BB_STATUS_0_MEAN' in bb_agg.columns:
            bb_agg['BB_PAYMENT_CONSISTENCY'] = bb_agg['BB_BB_STATUS_0_MEAN']
        
        # 2. Delinquency rate
        if 'BB_BB_STATUS_DPD_MEAN' in bb_agg.columns:
            bb_agg['BB_DELINQUENCY_RATE'] = bb_agg['BB_BB_STATUS_DPD_MEAN']
        
        # 3. Serious delinquency rate (60+ days)
        if all(col in bb_agg.columns for col in ['BB_BB_STATUS_3_SUM', 
                                                   'BB_BB_STATUS_4_SUM',
                                                   'BB_BB_STATUS_5_SUM',
                                                   'BB_MONTHS_BALANCE_SIZE']):
            bb_agg['BB_SERIOUS_DPD_RATE'] = (
                (bb_agg['BB_BB_STATUS_3_SUM'] + 
                 bb_agg['BB_BB_STATUS_4_SUM'] + 
                 bb_agg['BB_BB_STATUS_5_SUM']) / 
                (bb_agg['BB_MONTHS_BALANCE_SIZE'] + 1)
            )
        
        # 4. Has ever been seriously delinquent
        if all(col in bb_agg.columns for col in ['BB_BB_STATUS_3_SUM',
                                                   'BB_BB_STATUS_4_SUM', 
                                                   'BB_BB_STATUS_5_SUM']):
            bb_agg['BB_HAS_SERIOUS_DPD'] = (
                (bb_agg['BB_BB_STATUS_3_SUM'] > 0) |
                (bb_agg['BB_BB_STATUS_4_SUM'] > 0) |
                (bb_agg['BB_BB_STATUS_5_SUM'] > 0)
            ).astype(int)
        
        # 5. Recent payment behavior (last 12 months)
        bb_recent = bb[bb['MONTHS_BALANCE'] >= -12]
        if not bb_recent.empty:
            recent_agg = bb_recent.groupby('SK_ID_CURR').agg({
                'BB_STATUS_0': 'mean',
                'BB_STATUS_DPD': 'mean',
            })
            recent_agg.columns = ['BB_RECENT_12M_CURRENT_RATE', 'BB_RECENT_12M_DPD_RATE']
            bb_agg = bb_agg.merge(recent_agg, on='SK_ID_CURR', how='left')
        
        # 6. Trend analysis - getting better or worse?
        # Compare last 6 months vs previous 6 months
        bb_last6 = bb[bb['MONTHS_BALANCE'] >= -6]
        bb_prev6 = bb[(bb['MONTHS_BALANCE'] >= -12) & (bb['MONTHS_BALANCE'] < -6)]
        
        if not bb_last6.empty and not bb_prev6.empty:
            last6_dpd = bb_last6.groupby('SK_ID_CURR')['BB_STATUS_DPD'].mean()
            prev6_dpd = bb_prev6.groupby('SK_ID_CURR')['BB_STATUS_DPD'].mean()
            
            trend = (last6_dpd - prev6_dpd).reset_index()
            trend.columns = ['SK_ID_CURR', 'BB_DPD_TREND_6M']
            bb_agg = bb_agg.merge(trend, on='SK_ID_CURR', how='left')
            
            # Getting worse flag
            bb_agg['BB_GETTING_WORSE'] = (bb_agg['BB_DPD_TREND_6M'] > 0).astype(int)
        
        logger.info(f"✓ Created {len(bb_agg.columns) - 1} bureau balance features")
        return bb_agg


class PreviousApplicationFeatureEngineer:
    """Engineer features from previous loan applications"""
    
    def transform(self, prev_app: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from previous applications
        
        Args:
            prev_app: Previous applications dataframe
            
        Returns:
            DataFrame with aggregated features
        """
        logger.info("Engineering previous application features...")
        
        if prev_app is None or prev_app.empty:
            logger.warning("Previous applications data is empty")
            return pd.DataFrame()
        
        # === STATUS-BASED FEATURES ===
        
        # Count applications by status
        prev_app['PREV_APP_APPROVED'] = (prev_app['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
        prev_app['PREV_APP_REFUSED'] = (prev_app['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
        prev_app['PREV_APP_CANCELED'] = (prev_app['NAME_CONTRACT_STATUS'] == 'Canceled').astype(int)
        
        # === AGGREGATIONS ===
        
        aggregations = {
            # Amounts
            'AMT_CREDIT': ['sum', 'mean', 'max', 'min', 'std'],
            'AMT_ANNUITY': ['mean', 'max', 'min'],
            'AMT_APPLICATION': ['sum', 'mean', 'max', 'min'],
            'AMT_GOODS_PRICE': ['sum', 'mean', 'max', 'min'],
            'AMT_DOWN_PAYMENT': ['sum', 'mean', 'max'],
            
            # Rates and terms
            'RATE_DOWN_PAYMENT': ['mean', 'max', 'min'],
            'RATE_INTEREST_PRIMARY': ['mean', 'max', 'min'],
            'RATE_INTEREST_PRIVILEGED': ['mean', 'max', 'min'],
            'CNT_PAYMENT': ['mean', 'max', 'min', 'sum'],
            
            # Days
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'DAYS_FIRST_DRAWING': ['min', 'max', 'mean'],
            'DAYS_FIRST_DUE': ['min', 'max', 'mean'],
            'DAYS_LAST_DUE': ['min', 'max', 'mean'],
            'DAYS_TERMINATION': ['min', 'max', 'mean'],
            
            # Status counts
            'PREV_APP_APPROVED': ['sum', 'mean'],
            'PREV_APP_REFUSED': ['sum', 'mean'],
            'PREV_APP_CANCELED': ['sum', 'mean'],
            
            # Application characteristics
            'NAME_CONTRACT_TYPE': ['nunique'],
            'NAME_PAYMENT_TYPE': ['nunique'],
            'NAME_PRODUCT_TYPE': ['nunique'],
            'CHANNEL_TYPE': ['nunique'],
        }
        
        prev_agg = prev_app.groupby('SK_ID_CURR').agg(aggregations)  # type: ignore[call-overload]
        prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
        prev_agg.reset_index(inplace=True)
        
        # === DERIVED FEATURES ===
        
        # 1. Total number of previous applications
        prev_agg['PREV_TOTAL_APPLICATIONS'] = (
            prev_agg['PREV_PREV_APP_APPROVED_SUM'] + 
            prev_agg['PREV_PREV_APP_REFUSED_SUM'] + 
            prev_agg['PREV_PREV_APP_CANCELED_SUM']
        )
        
        # 2. Approval rate
        prev_agg['PREV_APPROVAL_RATE'] = (
            prev_agg['PREV_PREV_APP_APPROVED_SUM'] / 
            (prev_agg['PREV_TOTAL_APPLICATIONS'] + 1)
        )
        
        # 3. Refusal rate
        prev_agg['PREV_REFUSAL_RATE'] = (
            prev_agg['PREV_PREV_APP_REFUSED_SUM'] / 
            (prev_agg['PREV_TOTAL_APPLICATIONS'] + 1)
        )
        
        # 4. Cancellation rate
        prev_agg['PREV_CANCEL_RATE'] = (
            prev_agg['PREV_PREV_APP_CANCELED_SUM'] / 
            (prev_agg['PREV_TOTAL_APPLICATIONS'] + 1)
        )
        
        # 5. Application amount vs credit ratio
        if all(col in prev_agg.columns for col in ['PREV_AMT_APPLICATION_MEAN',
                                                     'PREV_AMT_CREDIT_MEAN']):
            prev_agg['PREV_APP_CREDIT_RATIO'] = (
                prev_agg['PREV_AMT_APPLICATION_MEAN'] / 
                (prev_agg['PREV_AMT_CREDIT_MEAN'] + 1)
            )
        
        # 6. Down payment ratio
        if all(col in prev_agg.columns for col in ['PREV_AMT_DOWN_PAYMENT_MEAN',
                                                     'PREV_AMT_CREDIT_MEAN']):
            prev_agg['PREV_DOWN_PAYMENT_RATIO'] = (
                prev_agg['PREV_AMT_DOWN_PAYMENT_MEAN'] / 
                (prev_agg['PREV_AMT_CREDIT_MEAN'] + 1)
            )
        
        # 7. Interest rate range (max - min)
        if all(col in prev_agg.columns for col in ['PREV_RATE_INTEREST_PRIMARY_MAX',
                                                     'PREV_RATE_INTEREST_PRIMARY_MIN']):
            prev_agg['PREV_INTEREST_RATE_RANGE'] = (
                prev_agg['PREV_RATE_INTEREST_PRIMARY_MAX'] - 
                prev_agg['PREV_RATE_INTEREST_PRIMARY_MIN']
            )
        
        # 8. Product diversity
        if 'PREV_NAME_PRODUCT_TYPE_NUNIQUE' in prev_agg.columns:
            prev_agg['PREV_PRODUCT_DIVERSITY'] = (
                prev_agg['PREV_NAME_PRODUCT_TYPE_NUNIQUE'] / 
                (prev_agg['PREV_TOTAL_APPLICATIONS'] + 1)
            )
        
        # 9. Channel diversity
        if 'PREV_CHANNEL_TYPE_NUNIQUE' in prev_agg.columns:
            prev_agg['PREV_CHANNEL_DIVERSITY'] = (
                prev_agg['PREV_CHANNEL_TYPE_NUNIQUE'] / 
                (prev_agg['PREV_TOTAL_APPLICATIONS'] + 1)
            )
        
        logger.info(f"✓ Created {len(prev_agg.columns) - 1} previous application features")
        return prev_agg


def create_bureau_features(bureau: pd.DataFrame,
                          bureau_balance: pd.DataFrame | None = None) -> pd.DataFrame:
    """Convenience function to create all bureau features
    
    Args:
        bureau: Bureau dataframe
        bureau_balance: Bureau balance dataframe (optional)
        
    Returns:
        DataFrame with all bureau features
    """
    engineer = BureauFeatureEngineer()
    
    # Bureau features
    bureau_feats = engineer.transform_bureau(bureau)
    
    # Bureau balance features
    if bureau_balance is not None and not bureau_balance.empty:
        bb_feats = engineer.transform_bureau_balance(bureau, bureau_balance)
        
        # Merge both
        if not bureau_feats.empty and not bb_feats.empty:
            return bureau_feats.merge(bb_feats, on='SK_ID_CURR', how='outer')
        elif not bb_feats.empty:
            return bb_feats
    
    return bureau_feats


def create_previous_application_features(prev_app: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to create previous application features
    
    Args:
        prev_app: Previous applications dataframe
        
    Returns:
        DataFrame with previous application features
    """
    engineer = PreviousApplicationFeatureEngineer()
    return engineer.transform(prev_app)


if __name__ == "__main__":
    # Test the feature engineering
    logging.basicConfig(level=logging.INFO)
    
    print("Bureau Feature Engineer Test")
    print("=" * 50)
    
    # This would need actual bureau data to test properly
    print("\nNote: Requires actual bureau data to test")
    print("Usage example:")
    print("""
    from credscope.data.loader import load_data
    from credscope.features.bureau import create_bureau_features
    
    # Load data
    bureau = load_data('bureau')
    bureau_balance = load_data('bureau_balance')
    
    # Create features
    bureau_features = create_bureau_features(bureau, bureau_balance)
    print(f"Created {len(bureau_features.columns)} bureau features")
    """)