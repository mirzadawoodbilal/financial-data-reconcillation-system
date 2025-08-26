"""
Advanced Parsing Example
Demonstrates advanced usage with real Excel files and different data formats.
"""

import sys
import os
import logging
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.reconciliation_orchestrator import ReconciliationOrchestrator
from core.excel_processor import ExcelProcessor

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('advanced_parsing.log')
        ]
    )
    return logging.getLogger(__name__)

def create_sample_excel_file():
    """Create a sample Excel file with multiple sheets for testing."""
    logger = logging.getLogger(__name__)
    
    # Create sample data with different formats
    transactions_data = {
        'Amount': [150.00, 75.50, 200.00, 125.25, 300.00, 50.75, 175.00, 225.50],
        'Description': [
            'Invoice #001 - Consulting Services',
            'Payment XYZ - Monthly Subscription',
            'Invoice #002 - Software License',
            'Payment ABC - Annual Fee',
            'Invoice #003 - Training Services',
            'Payment DEF - Support Contract',
            'Invoice #004 - Implementation',
            'Payment GHI - Maintenance'
        ],
        'Date': [
            '2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18',
            '2024-01-19', '2024-01-20', '2024-01-21', '2024-01-22'
        ],
        'Category': ['Income', 'Expense', 'Income', 'Expense', 'Income', 'Expense', 'Income', 'Expense']
    }
    
    targets_data = {
        'Target_Amount': [225.50, 300.00, 150.00, 125.25, 175.00, 50.75, 200.00, 75.50],
        'Reference_ID': [
            'REF001 - Consulting',
            'REF002 - Software',
            'REF003 - Training',
            'REF004 - Support',
            'REF005 - Implementation',
            'REF006 - Maintenance',
            'REF007 - Services',
            'REF008 - Subscription'
        ],
        'Expected_Date': [
            '2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18',
            '2024-01-19', '2024-01-20', '2024-01-21', '2024-01-22'
        ],
        'Status': ['Pending', 'Paid', 'Pending', 'Paid', 'Pending', 'Paid', 'Pending', 'Paid']
    }
    
    # Create Excel file with multiple sheets
    output_file = "data/sample/reconciliation_data.xlsx"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Transactions sheet
        transactions_df = pd.DataFrame(transactions_data)
        transactions_df.to_excel(writer, sheet_name='Transactions', index=False)
        
        # Targets sheet
        targets_df = pd.DataFrame(targets_data)
        targets_df.to_excel(writer, sheet_name='Targets', index=False)
        
        # Additional sheet with different format
        summary_data = {
            'Summary_Item': ['Total Transactions', 'Total Targets', 'Expected Matches'],
            'Value': [len(transactions_data['Amount']), len(targets_data['Target_Amount']), 5],
            'Notes': ['All transaction records', 'All target records', 'Estimated matches']
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    logger.info(f"Created sample Excel file: {output_file}")
    return output_file

def demonstrate_data_cleaning():
    """Demonstrate data cleaning capabilities."""
    logger = logging.getLogger(__name__)
    
    # Create sample data with various formats
    messy_data = {
        'Amount': [
            '$150.00', '150', '150.00 USD', '150,00', '150.0', '150.00$',
            '75.50', '75.5', '75,50', '75.50 EUR', '75.50€',
            '200.00', '200', '200,00', '200.00 GBP', '£200.00'
        ],
        'Description': [
            'Invoice #001', 'INV-001', 'Invoice 001', 'INVOICE #001',
            'Payment XYZ', 'PAY-XYZ', 'Payment_XYZ', 'PAYMENT XYZ',
            'Invoice #002', 'INV-002', 'Invoice 002', 'INVOICE #002',
            'Invoice #003', 'INV-003', 'Invoice 003', 'INVOICE #003'
        ]
    }
    
    messy_df = pd.DataFrame(messy_data)
    
    # Initialize processor
    processor = ExcelProcessor()
    
    # Clean the data
    logger.info("Demonstrating data cleaning...")
    logger.info("Original data:")
    print(messy_df.head())
    
    # Clean amount column
    cleaned_df = processor.clean_amount_column(messy_df, 'Amount')
    logger.info("Cleaned data:")
    print(cleaned_df.head())
    
    return cleaned_df

def demonstrate_excel_parsing():
    """Demonstrate Excel file parsing capabilities."""
    logger = logging.getLogger(__name__)
    
    # Create sample Excel file
    excel_file = create_sample_excel_file()
    
    # Initialize processor
    processor = ExcelProcessor()
    
    # Load and examine the Excel file
    logger.info("Loading Excel file...")
    transactions_df, targets_df = processor.load_excel_sheets(excel_file)
    
    logger.info("Original Transactions:")
    print(transactions_df.head())
    
    logger.info("Original Targets:")
    print(targets_df.head())
    
    # Prepare data with different column mappings
    logger.info("Preparing data with custom column mappings...")
    
    # Method 1: Using default column names
    prepared_txns1 = processor.prepare_transactions_data(transactions_df, 'Amount', 'Description')
    prepared_targets1 = processor.prepare_targets_data(targets_df, 'Target_Amount', 'Reference_ID')
    
    logger.info("Prepared Transactions (Method 1):")
    print(prepared_txns1.head())
    
    logger.info("Prepared Targets (Method 1):")
    print(prepared_targets1.head())
    
    # Method 2: Using different column mappings
    # Rename columns to match expected format
    transactions_df_renamed = transactions_df.rename(columns={'Amount': 'A', 'Description': 'B'})
    targets_df_renamed = targets_df.rename(columns={'Target_Amount': 'C', 'Reference_ID': 'D'})
    
    prepared_txns2 = processor.prepare_transactions_data(transactions_df_renamed, 'A', 'B')
    prepared_targets2 = processor.prepare_targets_data(targets_df_renamed, 'C', 'D')
    
    logger.info("Prepared Transactions (Method 2):")
    print(prepared_txns2.head())
    
    return prepared_txns1, prepared_targets1

def demonstrate_reconciliation_with_real_data():
    """Demonstrate reconciliation with real-like data."""
    logger = logging.getLogger(__name__)
    
    # Get prepared data
    transactions_df, targets_df = demonstrate_excel_parsing()
    
    # Initialize orchestrator
    orchestrator = ReconciliationOrchestrator(tolerance=0.01, logger=logger)
    
    # Run comprehensive reconciliation
    logger.info("Running reconciliation with real-like data...")
    results = orchestrator.run_comprehensive_reconciliation(transactions_df, targets_df)
    
    # Display results
    print("\n" + "="*60)
    print("RECONCILIATION RESULTS WITH REAL-LIKE DATA")
    print("="*60)
    print(orchestrator.get_summary_report())
    
    # Save results
    saved_files = orchestrator.save_results("results/advanced_parsing")
    logger.info(f"Results saved to: {list(saved_files.values())}")
    
    return results

def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    logger = logging.getLogger(__name__)
    
    # Create data with errors
    problematic_data = {
        'Amount': ['$150.00', 'invalid', '150.00', 'N/A', '150.00', 'abc', '150.00'],
        'Description': ['Invoice #001', 'Payment XYZ', 'Invoice #002', '', 'Invoice #003', None, 'Invoice #004']
    }
    
    problematic_df = pd.DataFrame(problematic_data)
    
    logger.info("Demonstrating error handling...")
    logger.info("Problematic data:")
    print(problematic_df)
    
    # Initialize processor
    processor = ExcelProcessor()
    
    # Clean the data (should handle errors gracefully)
    cleaned_df = processor.clean_amount_column(problematic_df, 'Amount')
    
    logger.info("Cleaned data (errors handled):")
    print(cleaned_df)
    
    logger.info(f"Original rows: {len(problematic_df)}")
    logger.info(f"Cleaned rows: {len(cleaned_df)}")
    logger.info(f"Removed rows: {len(problematic_df) - len(cleaned_df)}")

def main():
    """Main function demonstrating advanced parsing capabilities."""
    logger = setup_logging()
    logger.info("Starting Advanced Parsing Example")
    
    try:
        # Demonstrate data cleaning
        logger.info("="*60)
        logger.info("DEMONSTRATING DATA CLEANING")
        logger.info("="*60)
        demonstrate_data_cleaning()
        
        # Demonstrate Excel parsing
        logger.info("="*60)
        logger.info("DEMONSTRATING EXCEL PARSING")
        logger.info("="*60)
        demonstrate_excel_parsing()
        
        # Demonstrate reconciliation with real data
        logger.info("="*60)
        logger.info("DEMONSTRATING RECONCILIATION WITH REAL DATA")
        logger.info("="*60)
        demonstrate_reconciliation_with_real_data()
        
        # Demonstrate error handling
        logger.info("="*60)
        logger.info("DEMONSTRATING ERROR HANDLING")
        logger.info("="*60)
        demonstrate_error_handling()
        
        logger.info("Advanced parsing example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in advanced parsing example: {e}")
        raise

if __name__ == "__main__":
    main() 