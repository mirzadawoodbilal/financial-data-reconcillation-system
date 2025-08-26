"""
Basic Functionality Tests
Tests core functionality of the financial reconciliation system.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.excel_processor import ExcelProcessor
from core.brute_force_reconciler import BruteForceReconciler
from core.reconciliation_orchestrator import ReconciliationOrchestrator

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the reconciliation system."""
    
    def setUp(self):
        """Set up test data."""
        self.processor = ExcelProcessor()
        self.reconciler = BruteForceReconciler(tolerance=0.01)
        self.orchestrator = ReconciliationOrchestrator(tolerance=0.01)
        
        # Create test data
        self.transactions_data = {
            'amount': [150.00, 75.50, 200.00, 125.25, 300.00],
            'description': ['Invoice #001', 'Payment XYZ', 'Invoice #002', 'Payment ABC', 'Invoice #003']
        }
        
        self.targets_data = {
            'target_amount': [150.00, 75.50, 200.00, 125.25, 300.00],
            'reference_id': ['REF001', 'REF002', 'REF003', 'REF004', 'REF005']
        }
        
        self.transactions_df = pd.DataFrame(self.transactions_data)
        self.targets_df = pd.DataFrame(self.targets_data)
    
    def test_excel_processor_initialization(self):
        """Test Excel processor initialization."""
        self.assertIsNotNone(self.processor)
        self.assertIsInstance(self.processor, ExcelProcessor)
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        # Test with messy data
        messy_data = {
            'amount': ['$150.00', '150', '150.00 USD', 'invalid', '150.00'],
            'description': ['Invoice #001', 'Payment XYZ', 'Invoice #002', 'N/A', 'Invoice #003']
        }
        messy_df = pd.DataFrame(messy_data)
        
        # Clean the data
        cleaned_df = self.processor.clean_amount_column(messy_df, 'amount')
        
        # Should have valid numeric amounts
        self.assertTrue(cleaned_df['amount'].dtype in ['float64', 'int64'])
        self.assertEqual(len(cleaned_df), 4)  # One invalid row should be removed
    
    def test_transactions_preparation(self):
        """Test transactions data preparation."""
        prepared_df = self.processor.prepare_transactions_data(
            self.transactions_df, 'amount', 'description'
        )
        
        # Check required columns exist
        required_columns = ['transaction_id', 'amount', 'description', 'amount_abs', 'amount_rounded']
        for col in required_columns:
            self.assertIn(col, prepared_df.columns)
        
        # Check data types
        self.assertTrue(prepared_df['amount'].dtype in ['float64', 'int64'])
        self.assertEqual(len(prepared_df), len(self.transactions_df))
    
    def test_targets_preparation(self):
        """Test targets data preparation."""
        prepared_df = self.processor.prepare_targets_data(
            self.targets_df, 'target_amount', 'reference_id'
        )
        
        # Check required columns exist
        required_columns = ['target_id', 'target_amount', 'reference_id', 'target_amount_abs', 'target_amount_rounded']
        for col in required_columns:
            self.assertIn(col, prepared_df.columns)
        
        # Check data types
        self.assertTrue(prepared_df['target_amount'].dtype in ['float64', 'int64'])
        self.assertEqual(len(prepared_df), len(self.targets_df))
    
    def test_direct_matching(self):
        """Test direct matching functionality."""
        # Prepare data
        prepared_txns = self.processor.prepare_transactions_data(
            self.transactions_df, 'amount', 'description'
        )
        prepared_targets = self.processor.prepare_targets_data(
            self.targets_df, 'target_amount', 'reference_id'
        )
        
        # Run direct matching
        results = self.reconciler.direct_matching(prepared_txns, prepared_targets)
        
        # Check results structure
        self.assertIn('matches', results)
        self.assertIn('matched_transactions', results)
        self.assertIn('matched_targets', results)
        self.assertIn('execution_time', results)
        self.assertIn('total_matches', results)
        
        # Should find matches since amounts are identical
        self.assertGreater(results['total_matches'], 0)
    
    def test_subset_sum_brute_force(self):
        """Test subset sum brute force functionality."""
        # Create data with subset sum possibilities
        subset_txns_data = {
            'amount': [50.00, 75.00, 100.00, 25.00, 150.00],
            'description': ['TXN1', 'TXN2', 'TXN3', 'TXN4', 'TXN5']
        }
        subset_targets_data = {
            'target_amount': [125.00, 200.00],  # 50+75=125, 100+100=200
            'reference_id': ['TGT1', 'TGT2']
        }
        
        subset_txns_df = pd.DataFrame(subset_txns_data)
        subset_targets_df = pd.DataFrame(subset_targets_data)
        
        # Prepare data
        prepared_txns = self.processor.prepare_transactions_data(
            subset_txns_df, 'amount', 'description'
        )
        prepared_targets = self.processor.prepare_targets_data(
            subset_targets_df, 'target_amount', 'reference_id'
        )
        
        # Run subset sum
        results = self.reconciler.subset_sum_brute_force(prepared_txns, prepared_targets)
        
        # Check results structure
        self.assertIn('subset_matches', results)
        self.assertIn('execution_time', results)
        self.assertIn('total_subset_matches', results)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)
        self.assertIsInstance(self.orchestrator, ReconciliationOrchestrator)
        
        # Check components are initialized
        self.assertIsNotNone(self.orchestrator.excel_processor)
        self.assertIsNotNone(self.orchestrator.brute_force_reconciler)
        self.assertIsNotNone(self.orchestrator.ml_reconciler)
        self.assertIsNotNone(self.orchestrator.advanced_reconciler)
        self.assertIsNotNone(self.orchestrator.performance_analyzer)
    
    def test_sample_data_generation(self):
        """Test sample data generation."""
        num_transactions = 50
        num_targets = 10
        
        transactions_df, targets_df = self.processor.generate_sample_data(
            num_transactions, num_targets
        )
        
        # Check data structure
        self.assertEqual(len(transactions_df), num_transactions)
        self.assertEqual(len(targets_df), num_targets)
        
        # Check required columns
        self.assertIn('amount', transactions_df.columns)
        self.assertIn('description', transactions_df.columns)
        self.assertIn('target_amount', targets_df.columns)
        self.assertIn('reference_id', targets_df.columns)
        
        # Check data types
        self.assertTrue(transactions_df['amount'].dtype in ['float64', 'int64'])
        self.assertTrue(targets_df['target_amount'].dtype in ['float64', 'int64'])
    
    def test_data_summary(self):
        """Test data summary generation."""
        # Prepare data
        prepared_txns = self.processor.prepare_transactions_data(
            self.transactions_df, 'amount', 'description'
        )
        prepared_targets = self.processor.prepare_targets_data(
            self.targets_df, 'target_amount', 'reference_id'
        )
        
        # Generate summary
        summary = self.processor.get_data_summary(prepared_txns, prepared_targets)
        
        # Check summary structure
        required_keys = [
            'transactions_count', 'targets_count', 'transactions_total', 
            'targets_total', 'transactions_mean', 'targets_mean'
        ]
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check values
        self.assertEqual(summary['transactions_count'], len(prepared_txns))
        self.assertEqual(summary['targets_count'], len(prepared_targets))
        self.assertGreater(summary['transactions_total'], 0)
        self.assertGreater(summary['targets_total'], 0)

def run_basic_tests():
    """Run basic functionality tests."""
    print("Running basic functionality tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBasicFunctionality)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
