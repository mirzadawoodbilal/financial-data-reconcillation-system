"""
Basic Usage Example
Demonstrates basic usage of the financial reconciliation system.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.reconciliation_orchestrator import ReconciliationOrchestrator

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('reconciliation.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main function demonstrating basic usage."""
    logger = setup_logging()
    logger.info("Starting Financial Reconciliation System - Basic Usage Example")
    
    # Initialize the orchestrator
    orchestrator = ReconciliationOrchestrator(tolerance=0.01, logger=logger)
    
    # Generate sample data
    logger.info("Generating sample data...")
    transactions_df, targets_df = orchestrator.excel_processor.generate_sample_data(
        num_transactions=50, 
        num_targets=10
    )
    
    # Process the data through the Excel processor to ensure proper formatting
    transactions_df = orchestrator.excel_processor.prepare_transactions_data(
        transactions_df, amount_col='amount', desc_col='description'
    )
    targets_df = orchestrator.excel_processor.prepare_targets_data(
        targets_df, amount_col='target_amount', ref_col='reference_id'
    )
    
    logger.info(f"Generated {len(transactions_df)} transactions and {len(targets_df)} targets")
    
    # Display sample data
    print("\nSample Transactions:")
    print(transactions_df.head())
    
    print("\nSample Targets:")
    print(targets_df.head())
    
    # Run comprehensive reconciliation
    logger.info("Running comprehensive reconciliation...")
    results = orchestrator.run_comprehensive_reconciliation(transactions_df, targets_df)
    
    # Display summary report
    print("\n" + "="*60)
    print("RECONCILIATION SUMMARY")
    print("="*60)
    print(orchestrator.get_summary_report())
    
    # Save results
    logger.info("Saving results...")
    saved_files = orchestrator.save_results("results/basic_example")
    
    print(f"\nResults saved to: {list(saved_files.values())}")
    
    # Display detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    for approach, result in results['approaches'].items():
        if 'error' in result:
            print(f"\n{approach.upper()}: ERROR - {result['error']}")
        else:
            print(f"\n{approach.upper()}:")
            if approach == 'brute_force':
                print(f"  - Direct Matches: {result['direct_matching']['total_matches']}")
                print(f"  - Subset Matches: {result['subset_sum_matching']['total_subset_matches']}")
                print(f"  - Total Matches: {result['reconciliation_metrics']['matched_transactions']}")
                print(f"  - Reconciliation Rate: {result['reconciliation_metrics']['reconciliation_rate_transactions']:.2f}%")
                print(f"  - Execution Time: {result['total_execution_time']:.3f}s")
            
            elif approach == 'machine_learning':
                print(f"  - Feature Vectors: {result['feature_engineering']['total_feature_vectors']}")
                print(f"  - High Confidence Matches: {result['predictions']['high_confidence_matches']}")
                print(f"  - Best Model: {result['predictions']['best_model']}")
                print(f"  - Execution Time: {result['total_execution_time']:.3f}s")
            
            elif approach == 'advanced':
                print(f"  - GA Matches: {result['genetic_algorithm']['total_ga_matches']}")
                print(f"  - Fuzzy Matches: {result['fuzzy_matching']['total_fuzzy_matches']}")
                print(f"  - Cluster Matches: {result['clustering']['total_cluster_matches']}")
                print(f"  - Total Matches: {result['reconciliation_metrics']['matched_transactions']}")
                print(f"  - Execution Time: {result['total_execution_time']:.3f}s")
    
    # Display recommendations
    if 'recommendations' in results:
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        recs = results['recommendations']
        if 'best_approach' in recs:
            best = recs['best_approach']
            print(f"Best for Speed: {best.get('for_speed', 'N/A')}")
            print(f"Best for Accuracy: {best.get('for_accuracy', 'N/A')}")
            print(f"Most Efficient: {best.get('for_efficiency', 'N/A')}")
    
    logger.info("Basic usage example completed successfully!")

if __name__ == "__main__":
    main()
