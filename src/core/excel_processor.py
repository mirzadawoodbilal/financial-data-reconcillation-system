"""
Excel Processor Module
Handles loading, cleaning, and preparing Excel data for reconciliation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import re
from pathlib import Path
import logging

class ExcelProcessor:
    """Handles Excel file processing and data preparation for reconciliation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def load_excel_sheets(self, file_path: str, sheet1_name: str = None, sheet2_name: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load two Excel sheets for reconciliation.
        
        Args:
            file_path: Path to the Excel file
            sheet1_name: Name of the first sheet (transactions)
            sheet2_name: Name of the second sheet (targets)
            
        Returns:
            Tuple of (transactions_df, targets_df)
        """
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if sheet1_name is None:
                sheet1_name = sheet_names[0]
            if sheet2_name is None and len(sheet_names) > 1:
                sheet2_name = sheet_names[1]
            elif sheet2_name is None:
                raise ValueError("Only one sheet found. Please specify sheet names.")
                
            transactions_df = pd.read_excel(file_path, sheet_name=sheet1_name)
            targets_df = pd.read_excel(file_path, sheet_name=sheet2_name)
            
            self.logger.info(f"Loaded sheets: {sheet1_name} and {sheet2_name}")
            return transactions_df, targets_df
            
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {e}")
            raise
    
    def clean_amount_column(self, df: pd.DataFrame, amount_col: str) -> pd.DataFrame:
        """
        Clean and standardize amount columns.
        
        Args:
            df: DataFrame containing amount column
            amount_col: Name of the amount column
            
        Returns:
            DataFrame with cleaned amount column
        """
        df_clean = df.copy()
        
        # Convert to string first to handle various formats
        df_clean[amount_col] = df_clean[amount_col].astype(str)
        
        # Remove currency symbols, commas, and extra spaces
        df_clean[amount_col] = df_clean[amount_col].apply(
            lambda x: re.sub(r'[^\d.-]', '', str(x)) if pd.notna(x) else x
        )
        
        # Convert to float, handling errors
        df_clean[amount_col] = pd.to_numeric(df_clean[amount_col], errors='coerce')
        
        # Remove rows with invalid amounts
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=[amount_col])
        final_count = len(df_clean)
        
        if initial_count != final_count:
            self.logger.warning(f"Removed {initial_count - final_count} rows with invalid amounts")
            
        return df_clean
    
    def prepare_transactions_data(self, df: pd.DataFrame, amount_col: str = 'A', 
                                desc_col: str = 'B') -> pd.DataFrame:
        """
        Prepare transactions data for reconciliation.
        
        Args:
            df: Raw transactions DataFrame
            amount_col: Column name for amounts
            desc_col: Column name for descriptions
            
        Returns:
            Prepared transactions DataFrame
        """
        # Clean amount column
        df_clean = self.clean_amount_column(df, amount_col)
        
        # Create unique transaction ID
        df_clean['transaction_id'] = [f"TXN_{i:06d}" for i in range(len(df_clean))]
        
        # Standardize column names
        df_clean = df_clean.rename(columns={
            amount_col: 'amount',
            desc_col: 'description'
        })
        
        # Fill missing descriptions
        df_clean['description'] = df_clean['description'].fillna('No Description')
        
        # Add metadata
        df_clean['amount_abs'] = df_clean['amount'].abs()
        df_clean['amount_rounded'] = df_clean['amount'].round(2)
        
        self.logger.info(f"Prepared {len(df_clean)} transactions")
        return df_clean
    
    def prepare_targets_data(self, df: pd.DataFrame, amount_col: str = 'C', 
                           ref_col: str = 'D') -> pd.DataFrame:
        """
        Prepare targets data for reconciliation.
        
        Args:
            df: Raw targets DataFrame
            amount_col: Column name for target amounts
            ref_col: Column name for reference IDs
            
        Returns:
            Prepared targets DataFrame
        """
        # Clean amount column
        df_clean = self.clean_amount_column(df, amount_col)
        
        # Create unique target ID
        df_clean['target_id'] = [f"TGT_{i:06d}" for i in range(len(df_clean))]
        
        # Standardize column names
        df_clean = df_clean.rename(columns={
            amount_col: 'target_amount',
            ref_col: 'reference_id'
        })
        
        # Fill missing reference IDs
        df_clean['reference_id'] = df_clean['reference_id'].fillna('No Reference')
        
        # Add metadata
        df_clean['target_amount_abs'] = df_clean['target_amount'].abs()
        df_clean['target_amount_rounded'] = df_clean['target_amount'].round(2)
        
        self.logger.info(f"Prepared {len(df_clean)} targets")
        return df_clean
    
    def generate_sample_data(self, num_transactions: int = 100, num_targets: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate sample data for testing and demonstration.
        
        Args:
            num_transactions: Number of transactions to generate
            num_targets: Number of targets to generate
            
        Returns:
            Tuple of (transactions_df, targets_df)
        """
        np.random.seed(42)
        
        # Generate transactions
        amounts = np.random.uniform(10, 1000, num_transactions)
        descriptions = [f"Invoice #{i:04d}" for i in range(num_transactions)]
        
        transactions_df = pd.DataFrame({
            'amount': amounts,
            'description': descriptions
        })
        
        # Generate targets (some matching, some not)
        target_amounts = []
        for _ in range(num_targets):
            if np.random.random() < 0.3:  # 30% chance of exact match
                target_amounts.append(np.random.choice(amounts))
            else:
                target_amounts.append(np.random.uniform(10, 1000))
        
        reference_ids = [f"REF{i:03d}" for i in range(num_targets)]
        
        targets_df = pd.DataFrame({
            'target_amount': target_amounts,
            'reference_id': reference_ids
        })
        
        return transactions_df, targets_df
    
    def save_processed_data(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame, 
                          output_path: str) -> None:
        """
        Save processed data to Excel file.
        
        Args:
            transactions_df: Processed transactions DataFrame
            targets_df: Processed targets DataFrame
            output_path: Output file path
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            transactions_df.to_excel(writer, sheet_name='Transactions', index=False)
            targets_df.to_excel(writer, sheet_name='Targets', index=False)
        
        self.logger.info(f"Saved processed data to {output_path}")
    
    def get_data_summary(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of the data.
        
        Args:
            transactions_df: Processed transactions DataFrame
            targets_df: Processed targets DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'transactions_count': len(transactions_df),
            'targets_count': len(targets_df),
            'transactions_total': transactions_df['amount'].sum(),
            'targets_total': targets_df['target_amount'].sum(),
            'transactions_mean': transactions_df['amount'].mean(),
            'targets_mean': targets_df['target_amount'].mean(),
            'transactions_std': transactions_df['amount'].std(),
            'targets_std': targets_df['target_amount'].std(),
            'transactions_min': transactions_df['amount'].min(),
            'transactions_max': transactions_df['amount'].max(),
            'targets_min': targets_df['target_amount'].min(),
            'targets_max': targets_df['target_amount'].max(),
        }
        
        return summary
