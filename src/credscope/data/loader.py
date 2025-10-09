import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Memory-efficient data loader for Home Credit dataset"""
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
    def reduce_memory_usage(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Reduce memory usage by optimizing dtypes
        
        Args:
            df: DataFrame to optimize
            verbose: Whether to print memory reduction info
            
        Returns:
            Optimized DataFrame
        """
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                # Skip optimization for columns with NaN values when converting to int
                if df[col].isnull().any() and str(col_type)[:3] == 'float':
                    # Keep as float but try to reduce precision
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                elif str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                elif str(col_type)[:3] == 'float':
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        end_mem = df.memory_usage().sum() / 1024**2
        
        if verbose:
            logger.info(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
        
        return df
    
    def load_application_data(self, train: bool = True, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load application train or test data
        
        Args:
            train: Whether to load train or test data
            nrows: Number of rows to load (for testing)
            
        Returns:
            Loaded and optimized DataFrame
        """
        filename = "application_train.csv" if train else "application_test.csv"
        filepath = self.data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            # Load without specifying dtypes first to avoid conversion errors
            logger.info(f"Loading {filename}...")
            df = pd.read_csv(filepath, nrows=nrows)
            
            if df.empty:
                raise ValueError(f"Loaded DataFrame is empty: {filename}")
            
            # Now optimize memory usage after loading
            df = self.reduce_memory_usage(df)
            
            logger.info(f"✅ Loaded {filename}: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Log basic statistics
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            logger.info(f"Missing values: {missing_pct:.1f}%")
            
            if 'TARGET' in df.columns:
                logger.info(f"Target distribution - Default rate: {df['TARGET'].mean():.3%}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def load_bureau_data(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load bureau and bureau_balance data
        
        Args:
            nrows: Number of rows to load (for testing)
            
        Returns:
            Tuple of (bureau, bureau_balance) DataFrames
        """
        bureau_path = self.data_path / "bureau.csv"
        bureau_balance_path = self.data_path / "bureau_balance.csv"
        
        if not bureau_path.exists():
            raise FileNotFoundError(f"File not found: {bureau_path}")
        if not bureau_balance_path.exists():
            raise FileNotFoundError(f"File not found: {bureau_balance_path}")
        
        try:
            logger.info("Loading bureau data...")
            bureau = pd.read_csv(bureau_path, nrows=nrows)
            bureau = self.reduce_memory_usage(bureau, verbose=False)
            logger.info(f"✅ Loaded bureau.csv: {bureau.shape[0]:,} rows, {bureau.shape[1]} columns")
            
            logger.info("Loading bureau_balance data...")
            bureau_balance = pd.read_csv(bureau_balance_path, nrows=nrows)
            bureau_balance = self.reduce_memory_usage(bureau_balance, verbose=False)
            logger.info(f"✅ Loaded bureau_balance.csv: {bureau_balance.shape[0]:,} rows, {bureau_balance.shape[1]} columns")
            
            return bureau, bureau_balance
            
        except Exception as e:
            logger.error(f"Error loading bureau data: {str(e)}")
            raise
    
    def load_previous_application(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load previous application data"""
        filepath = self.data_path / "previous_application.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info("Loading previous application data...")
        df = pd.read_csv(filepath, nrows=nrows)
        df = self.reduce_memory_usage(df, verbose=False)
        logger.info(f"✅ Loaded previous_application.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def load_installments_payments(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load installments payments data"""
        filepath = self.data_path / "installments_payments.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info("Loading installments payments data...")
        df = pd.read_csv(filepath, nrows=nrows)
        df = self.reduce_memory_usage(df, verbose=False)
        logger.info(f"✅ Loaded installments_payments.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def load_credit_card_balance(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load credit card balance data"""
        filepath = self.data_path / "credit_card_balance.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info("Loading credit card balance data...")
        df = pd.read_csv(filepath, nrows=nrows)
        df = self.reduce_memory_usage(df, verbose=False)
        logger.info(f"✅ Loaded credit_card_balance.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def load_pos_cash_balance(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load POS cash balance data"""
        filepath = self.data_path / "POS_CASH_balance.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info("Loading POS cash balance data...")
        df = pd.read_csv(filepath, nrows=nrows)
        df = self.reduce_memory_usage(df, verbose=False)
        logger.info(f"✅ Loaded POS_CASH_balance.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def get_train_test_split(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Split data into train and validation sets
        
        Args:
            df: DataFrame with TARGET and SK_ID_CURR columns
            test_size: Proportion of data to use for validation
            
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val))
        """
        from sklearn.model_selection import train_test_split
        
        # Validate required columns
        required_cols = ['TARGET', 'SK_ID_CURR']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")
        
        X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = df['TARGET']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        logger.info(f"Validation set: {X_val.shape[0]:,} samples, {X_val.shape[1]} features")
        logger.info(f"Class distribution - Train: {y_train.mean():.3%}, Val: {y_val.mean():.3%}")
        
        return (X_train, y_train), (X_val, y_val)