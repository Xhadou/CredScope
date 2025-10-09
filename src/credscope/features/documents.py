"""Document Submission Behavioral Features

This module creates features based on document submission patterns, which are
strong alternative credit signals indicating applicant engagement and credibility.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class DocumentFeatureEngineer:
    """Engineer document submission behavioral features"""
    
    def __init__(self):
        self.document_columns = None
        self.document_importance = {}
        
    def fit(self, df: pd.DataFrame, target: str = 'TARGET') -> 'DocumentFeatureEngineer':
        """Calculate document statistics from training data
        
        Args:
            df: Training dataframe
            target: Target column name
            
        Returns:
            self for chaining
        """
        logger.info("Analyzing document submission patterns...")
        
        # Identify all document flag columns
        self.document_columns = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
        
        if not self.document_columns:
            logger.warning("No document flag columns found!")
            return self
        
        # Calculate importance of each document type
        for doc_col in self.document_columns:
            if doc_col in df.columns and target in df.columns:
                # Default rate for those who submitted vs didn't submit
                submitted_default = df[df[doc_col] == 1][target].mean()
                not_submitted_default = df[df[doc_col] == 0][target].mean()
                
                # Importance score (higher = more predictive)
                importance = abs(submitted_default - not_submitted_default)
                self.document_importance[doc_col] = {
                    'submitted_default_rate': submitted_default,
                    'not_submitted_default_rate': not_submitted_default,
                    'importance_score': importance
                }
        
        logger.info(f"✓ Analyzed {len(self.document_columns)} document types")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate document behavioral features
        
        Args:
            df: Dataframe to transform
            
        Returns:
            DataFrame with document features
        """
        logger.info("Engineering document submission features...")
        df = df.copy()
        
        # Get all document columns
        doc_cols = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
        
        if not doc_cols:
            logger.warning("No document columns found for feature engineering")
            return df
        
        # === BASIC DOCUMENT STATISTICS ===
        
        # 1. Total documents submitted
        df['DOC_COUNT_TOTAL'] = df[doc_cols].sum(axis=1)
        
        # 2. Document submission rate (percentage)
        df['DOC_SUBMISSION_RATE'] = df['DOC_COUNT_TOTAL'] / len(doc_cols)
        
        # 3. Binary flag for complete documentation
        df['DOC_COMPLETE'] = (df['DOC_COUNT_TOTAL'] == len(doc_cols)).astype(int)
        
        # 4. Binary flag for minimal documentation
        df['DOC_MINIMAL'] = (df['DOC_COUNT_TOTAL'] <= 3).astype(int)
        
        # === DOCUMENT CATEGORIES ===
        
        # 5. Group documents by common patterns
        # Essential documents (typically required)
        essential_docs = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 
                         'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6']
        existing_essential = [doc for doc in essential_docs if doc in df.columns]
        if existing_essential:
            df['DOC_ESSENTIAL_COUNT'] = df[existing_essential].sum(axis=1)
            df['DOC_ESSENTIAL_RATE'] = df['DOC_ESSENTIAL_COUNT'] / len(existing_essential)
        
        # Optional documents
        optional_docs = ['FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
                        'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12']
        existing_optional = [doc for doc in optional_docs if doc in df.columns]
        if existing_optional:
            df['DOC_OPTIONAL_COUNT'] = df[existing_optional].sum(axis=1)
            df['DOC_OPTIONAL_RATE'] = df['DOC_OPTIONAL_COUNT'] / len(existing_optional)
        
        # Supplementary documents
        supplementary_docs = ['FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
                             'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
                             'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
        existing_supplementary = [doc for doc in supplementary_docs if doc in df.columns]
        if existing_supplementary:
            df['DOC_SUPPLEMENTARY_COUNT'] = df[existing_supplementary].sum(axis=1)
            df['DOC_SUPPLEMENTARY_RATE'] = df['DOC_SUPPLEMENTARY_COUNT'] / len(existing_supplementary)
        
        # === DOCUMENT PATTERNS ===
        
        # 6. Consecutive documents pattern
        # Check for consecutive document submissions
        doc_array = df[doc_cols].values
        consecutive_runs = []
        for row in doc_array:
            max_run = 0
            current_run = 0
            for val in row:
                if val == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            consecutive_runs.append(max_run)
        
        df['DOC_MAX_CONSECUTIVE'] = consecutive_runs
        
        # 7. Document gaps (missing documents between submitted ones)
        gaps = []
        for row in doc_array:
            if row.sum() <= 1:
                gaps.append(0)
            else:
                first_doc = np.where(row == 1)[0][0]
                last_doc = np.where(row == 1)[0][-1]
                expected_docs = last_doc - first_doc + 1
                actual_docs = row[first_doc:last_doc+1].sum()
                gaps.append(expected_docs - actual_docs)
        
        df['DOC_GAPS'] = gaps
        
        # === WEIGHTED DOCUMENT SCORE ===
        
        # 8. Weighted score based on document importance (if fitted)
        if self.document_importance:
            weighted_score = np.zeros(len(df))
            for doc_col, stats in self.document_importance.items():
                if doc_col in df.columns:
                    weight = stats['importance_score']
                    weighted_score += df[doc_col] * weight
            
            df['DOC_WEIGHTED_SCORE'] = weighted_score
            # Normalize to 0-1 range
            if weighted_score.max() > 0:
                df['DOC_WEIGHTED_SCORE_NORM'] = weighted_score / weighted_score.max()
            else:
                df['DOC_WEIGHTED_SCORE_NORM'] = 0
        
        # === SPECIFIC DOCUMENT COMBINATIONS ===
        
        # 9. Key document combinations that might indicate risk
        # ID + Income proof
        if all(doc in df.columns for doc in ['FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6']):
            df['DOC_ID_INCOME_COMBO'] = (
                (df['FLAG_DOCUMENT_3'] == 1) & (df['FLAG_DOCUMENT_6'] == 1)
            ).astype(int)
        
        # 10. Employment documents
        employment_docs = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4']
        existing_employment = [doc for doc in employment_docs if doc in df.columns]
        if existing_employment:
            df['DOC_EMPLOYMENT_COUNT'] = df[existing_employment].sum(axis=1)
            df['DOC_HAS_EMPLOYMENT'] = (df['DOC_EMPLOYMENT_COUNT'] > 0).astype(int)
        
        # === DOCUMENT BEHAVIOR RISK INDICATORS ===
        
        # 11. Suspiciously low documentation (possible fraud indicator)
        df['DOC_SUSPICIOUSLY_LOW'] = (df['DOC_COUNT_TOTAL'] <= 2).astype(int)
        
        # 12. Over-documentation (might indicate over-eagerness or trying too hard)
        df['DOC_OVER_DOCUMENTED'] = (df['DOC_COUNT_TOTAL'] >= 15).astype(int)
        
        # 13. Balanced documentation (sweet spot)
        df['DOC_BALANCED'] = (
            (df['DOC_COUNT_TOTAL'] >= 5) & (df['DOC_COUNT_TOTAL'] <= 12)
        ).astype(int)
        
        # === MISSING DOCUMENT PATTERNS ===
        
        # 14. Count of missing critical documents
        critical_docs = ['FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8']
        existing_critical = [doc for doc in critical_docs if doc in df.columns]
        if existing_critical:
            df['DOC_MISSING_CRITICAL'] = len(existing_critical) - df[existing_critical].sum(axis=1)
        
        # 15. Document diversity score (how spread out the documents are)
        # Higher is better - shows comprehensive documentation
        if len(doc_cols) > 0:
            # Calculate entropy-like measure
            doc_array = df[doc_cols].values
            diversity_scores = []
            for row in doc_array:
                if row.sum() == 0:
                    diversity_scores.append(0)
                else:
                    # Positions of submitted documents
                    positions = np.where(row == 1)[0]
                    if len(positions) <= 1:
                        diversity_scores.append(0)
                    else:
                        # Standard deviation of positions (normalized)
                        std = np.std(positions)
                        diversity_scores.append(std / len(doc_cols))
            
            df['DOC_DIVERSITY_SCORE'] = diversity_scores
        
        # === INTERACTION WITH OTHER FEATURES ===
        
        # 16. Document submission relative to income
        if 'AMT_INCOME_TOTAL' in df.columns:
            df['DOC_PER_INCOME_UNIT'] = df['DOC_COUNT_TOTAL'] / (df['AMT_INCOME_TOTAL'] / 10000 + 1)
        
        # 17. Document submission relative to credit amount
        if 'AMT_CREDIT' in df.columns:
            df['DOC_PER_CREDIT_UNIT'] = df['DOC_COUNT_TOTAL'] / (df['AMT_CREDIT'] / 100000 + 1)
        
        # Count new features
        new_cols = [col for col in df.columns if col.startswith('DOC_')]
        logger.info(f"✓ Created {len(new_cols)} document features")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target: str = 'TARGET') -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df, target).transform(df)


def create_document_features(df: pd.DataFrame,
                            target: str = 'TARGET',
                            is_train: bool = True) -> pd.DataFrame:
    """Convenience function to create document features
    
    Args:
        df: Input dataframe
        target: Target column name
        is_train: Whether this is training data
        
    Returns:
        DataFrame with document features
    """
    engineer = DocumentFeatureEngineer()
    
    if is_train:
        return engineer.fit_transform(df, target)
    else:
        return engineer.transform(df)


if __name__ == "__main__":
    # Test the feature engineering
    logging.basicConfig(level=logging.INFO)
    
    print("Document Feature Engineer Test")
    print("=" * 50)
    
    # Create sample data with document flags
    sample_data = pd.DataFrame({
        'FLAG_DOCUMENT_2': [1, 0, 1, 1, 0],
        'FLAG_DOCUMENT_3': [1, 1, 1, 0, 1],
        'FLAG_DOCUMENT_4': [1, 0, 0, 1, 1],
        'FLAG_DOCUMENT_5': [0, 0, 1, 1, 0],
        'FLAG_DOCUMENT_6': [1, 1, 1, 1, 1],
        'FLAG_DOCUMENT_8': [1, 0, 1, 0, 0],
        'AMT_INCOME_TOTAL': [100000, 150000, 200000, 120000, 180000],
        'AMT_CREDIT': [500000, 600000, 400000, 550000, 450000],
        'TARGET': [0, 1, 0, 1, 0]
    })
    
    engineer = DocumentFeatureEngineer()
    result = engineer.fit_transform(sample_data)
    
    # Show new features
    doc_features = [col for col in result.columns if col.startswith('DOC_')]
    print(f"\nCreated {len(doc_features)} features:")
    for feat in doc_features:
        print(f"  - {feat}")
    
    print(f"\nSample output:\n{result[doc_features].head()}")