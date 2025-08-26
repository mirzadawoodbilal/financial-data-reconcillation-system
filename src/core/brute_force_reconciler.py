"""
Brute Force Reconciliation Module
Implements direct matching and subset sum algorithms for financial reconciliation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
import time
import itertools
from collections import defaultdict
import logging

class BruteForceReconciler:
    """Implements brute force approaches for financial reconciliation."""
    
    def __init__(self, tolerance: float = 0.01, logger: Optional[logging.Logger] = None):
        self.tolerance = tolerance
        self.logger = logger or logging.getLogger(__name__)
        
    def direct_matching(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict:
        """
        Perform direct matching between individual transactions and targets.
        
        Args:
            transactions_df: DataFrame with transaction data
            targets_df: DataFrame with target data
            
        Returns:
            Dictionary containing matching results
        """
        start_time = time.time()
        
        matches = []
        matched_transactions = set()
        matched_targets = set()
        
        # Create lookup dictionaries for faster access
        transactions_dict = transactions_df.set_index('transaction_id').to_dict('index')
        targets_dict = targets_df.set_index('target_id').to_dict('index')
        
        for txn_id, txn_data in transactions_dict.items():
            for target_id, target_data in targets_dict.items():
                if target_id in matched_targets:
                    continue
                    
                # Check for exact match within tolerance
                if abs(txn_data['amount'] - target_data['target_amount']) <= self.tolerance:
                    matches.append({
                        'transaction_id': txn_id,
                        'target_id': target_id,
                        'transaction_amount': txn_data['amount'],
                        'target_amount': target_data['target_amount'],
                        'difference': txn_data['amount'] - target_data['target_amount'],
                        'match_type': 'exact'
                    })
                    matched_transactions.add(txn_id)
                    matched_targets.add(target_id)
                    break
        
        execution_time = time.time() - start_time
        
        results = {
            'matches': matches,
            'matched_transactions': list(matched_transactions),
            'matched_targets': list(matched_targets),
            'unmatched_transactions': [txn_id for txn_id in transactions_dict.keys() 
                                     if txn_id not in matched_transactions],
            'unmatched_targets': [target_id for target_id in targets_dict.keys() 
                                if target_id not in matched_targets],
            'execution_time': execution_time,
            'total_matches': len(matches)
        }
        
        self.logger.info(f"Direct matching completed: {len(matches)} matches found in {execution_time:.4f}s")
        return results
    
    def subset_sum_brute_force(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame, 
                              max_combinations: int = 1000) -> Dict:
        """
        Implement subset sum problem using brute force approach.
        
        Args:
            transactions_df: DataFrame with transaction data
            targets_df: DataFrame with target data
            max_combinations: Maximum number of combinations to try per target
            
        Returns:
            Dictionary containing subset sum results
        """
        start_time = time.time()
        
        subset_matches = []
        used_transactions = set()
        
        # Get unmatched transactions and targets
        direct_results = self.direct_matching(transactions_df, targets_df)
        matched_txns = set(direct_results['matched_transactions'])
        matched_targets = set(direct_results['matched_targets'])
        
        unmatched_txns = set(transactions_df['transaction_id']) - matched_txns
        unmatched_targets = set(targets_df['target_id']) - matched_targets
        
        if not unmatched_txns or not unmatched_targets:
            return {
                'subset_matches': [],
                'execution_time': time.time() - start_time,
                'total_subset_matches': 0,
                'used_transactions': []
            }
        
        # Create filtered DataFrames
        unmatched_txns_df = transactions_df[transactions_df['transaction_id'].isin(unmatched_txns)]
        unmatched_targets_df = targets_df[targets_df['target_id'].isin(unmatched_targets)]
        
        for _, target_row in unmatched_targets_df.iterrows():
            target_amount = target_row['target_amount']
            target_id = target_row['target_id']
            
            # Get available transactions (not used in previous matches)
            available_txns = unmatched_txns_df[~unmatched_txns_df['transaction_id'].isin(used_transactions)]
            
            if len(available_txns) == 0:
                continue
                
            # Try different combination sizes
            best_match = None
            best_difference = float('inf')
            
            for r in range(1, min(6, len(available_txns) + 1)):  # Limit to 5 transactions max
                combinations = list(itertools.combinations(range(len(available_txns)), r))
                
                # Limit number of combinations to check
                if len(combinations) > max_combinations:
                    combinations = combinations[:max_combinations]
                
                for combo_indices in combinations:
                    combo_txns = available_txns.iloc[list(combo_indices)]
                    combo_sum = combo_txns['amount'].sum()
                    
                    difference = abs(combo_sum - target_amount)
                    
                    if difference <= self.tolerance and difference < best_difference:
                        best_match = {
                            'target_id': target_id,
                            'target_amount': target_amount,
                            'transaction_ids': combo_txns['transaction_id'].tolist(),
                            'transaction_amounts': combo_txns['amount'].tolist(),
                            'combo_sum': combo_sum,
                            'difference': combo_sum - target_amount,
                            'num_transactions': len(combo_txns)
                        }
                        best_difference = difference
            
            if best_match:
                subset_matches.append(best_match)
                used_transactions.update(best_match['transaction_ids'])
        
        execution_time = time.time() - start_time
        
        results = {
            'subset_matches': subset_matches,
            'execution_time': execution_time,
            'total_subset_matches': len(subset_matches),
            'used_transactions': list(used_transactions)
        }
        
        self.logger.info(f"Subset sum brute force completed: {len(subset_matches)} matches found in {execution_time:.4f}s")
        return results
    
    def performance_analysis(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame, 
                           dataset_sizes: List[int] = [10, 25, 50, 100]) -> Dict:
        """
        Analyze performance for different dataset sizes.
        
        Args:
            transactions_df: Full transactions DataFrame
            targets_df: Full targets DataFrame
            dataset_sizes: List of dataset sizes to test
            
        Returns:
            Dictionary with performance metrics
        """
        performance_results = {}
        
        for size in dataset_sizes:
            if size > len(transactions_df) or size > len(targets_df):
                continue
                
            # Sample data
            sample_txns = transactions_df.sample(n=min(size, len(transactions_df)), random_state=42)
            sample_targets = targets_df.sample(n=min(size, len(targets_df)), random_state=42)
            
            # Test direct matching
            start_time = time.time()
            direct_results = self.direct_matching(sample_txns, sample_targets)
            direct_time = time.time() - start_time
            
            # Test subset sum
            start_time = time.time()
            subset_results = self.subset_sum_brute_force(sample_txns, sample_targets)
            subset_time = time.time() - start_time
            
            performance_results[size] = {
                'direct_matching_time': direct_time,
                'subset_sum_time': subset_time,
                'total_time': direct_time + subset_time,
                'direct_matches': direct_results['total_matches'],
                'subset_matches': subset_results['total_subset_matches'],
                'total_matches': direct_results['total_matches'] + subset_results['total_subset_matches']
            }
        
        return performance_results
    
    def get_reconciliation_summary(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict:
        """
        Get comprehensive reconciliation summary using brute force approach.
        
        Args:
            transactions_df: Processed transactions DataFrame
            targets_df: Processed targets DataFrame
            
        Returns:
            Dictionary with comprehensive reconciliation results
        """
        start_time = time.time()
        
        # Perform direct matching
        direct_results = self.direct_matching(transactions_df, targets_df)
        
        # Perform subset sum matching
        subset_results = self.subset_sum_brute_force(transactions_df, targets_df)
        
        total_time = time.time() - start_time
        
        # Calculate reconciliation metrics
        total_transactions = len(transactions_df)
        total_targets = len(targets_df)
        total_matched_txns = len(direct_results['matched_transactions']) + len(subset_results.get('used_transactions', []))
        total_matched_targets = len(direct_results['matched_targets']) + subset_results.get('total_subset_matches', 0)
        
        reconciliation_rate_txns = (total_matched_txns / total_transactions) * 100 if total_transactions > 0 else 0
        reconciliation_rate_targets = (total_matched_targets / total_targets) * 100 if total_targets > 0 else 0
        
        summary = {
            'total_execution_time': total_time,
            'direct_matching': direct_results,
            'subset_sum_matching': subset_results,
            'reconciliation_metrics': {
                'total_transactions': total_transactions,
                'total_targets': total_targets,
                'matched_transactions': total_matched_txns,
                'matched_targets': total_matched_targets,
                'reconciliation_rate_transactions': reconciliation_rate_txns,
                'reconciliation_rate_targets': reconciliation_rate_targets,
                'unmatched_transactions': total_transactions - total_matched_txns,
                'unmatched_targets': total_targets - total_matched_targets
            }
        }
        
        self.logger.info(f"Brute force reconciliation completed in {total_time:.4f}s")
        self.logger.info(f"Reconciliation rate: {reconciliation_rate_txns:.2f}% transactions, {reconciliation_rate_targets:.2f}% targets")
        
        return summary
