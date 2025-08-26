"""
Performance Demonstration Example
Demonstrates benchmarking and performance analysis capabilities.
"""

import sys
import os
import logging
import time
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
            logging.FileHandler('performance_demo.log')
        ]
    )
    return logging.getLogger(__name__)

def demonstrate_benchmarking(orchestrator):
    """Demonstrate comprehensive benchmarking capabilities."""
    logger = logging.getLogger(__name__)
    
    # Generate ONE large dataset for all benchmarking
    logger.info("Generating large dataset for benchmarking...")
    transactions_df, targets_df = orchestrator.excel_processor.generate_sample_data(
        num_transactions=200,  # Reduced for speed
        num_targets=50
    )
    
    # Process the data through the Excel processor to ensure proper formatting
    transactions_df = orchestrator.excel_processor.prepare_transactions_data(
        transactions_df, amount_col='amount', desc_col='description'
    )
    targets_df = orchestrator.excel_processor.prepare_targets_data(
        targets_df, amount_col='target_amount', ref_col='reference_id'
    )
    
    logger.info(f"Generated {len(transactions_df)} transactions and {len(targets_df)} targets for benchmarking")
    
    # Run benchmark analysis with proper dataset sizes
    logger.info("Running comprehensive benchmark analysis...")
    dataset_sizes = [10, 25, 50]  # Reduced for speed
    
    benchmark_results = {}
    
    for size in dataset_sizes:
        if size > len(transactions_df) or size > len(targets_df):
            continue
            
        logger.info(f"Benchmarking with dataset size: {size}")
        
        # Sample from the same base dataset
        sample_txns = transactions_df.sample(n=min(size, len(transactions_df)), random_state=42)
        sample_targets = targets_df.sample(n=min(size, len(targets_df)), random_state=42)
        
        size_results = {}
        
        # Brute Force Approach
        try:
            start_time = time.time()
            bf_summary = orchestrator.brute_force_reconciler.get_reconciliation_summary(sample_txns, sample_targets)
            bf_time = time.time() - start_time
            
            size_results['brute_force'] = {
                'execution_time': bf_time,
                'total_matches': bf_summary['reconciliation_metrics']['matched_transactions'],
                'reconciliation_rate': bf_summary['reconciliation_metrics']['reconciliation_rate_transactions']
            }
        except Exception as e:
            logger.warning(f"Brute force failed for size {size}: {e}")
            size_results['brute_force'] = {'execution_time': 0, 'total_matches': 0, 'reconciliation_rate': 0}
        
        # Machine Learning Approach
        try:
            start_time = time.time()
            ml_summary = orchestrator.ml_reconciler.get_ml_reconciliation_summary(sample_txns, sample_targets)
            ml_time = time.time() - start_time
            
            # Calculate reconciliation rate properly
            high_confidence_matches = ml_summary['predictions']['high_confidence_matches']
            reconciliation_rate = min((high_confidence_matches / len(sample_txns)) * 100, 100.0)
            
            size_results['machine_learning'] = {
                'execution_time': ml_time,
                'total_matches': high_confidence_matches,
                'reconciliation_rate': reconciliation_rate
            }
        except Exception as e:
            logger.warning(f"Machine learning failed for size {size}: {e}")
            size_results['machine_learning'] = {'execution_time': 0, 'total_matches': 0, 'reconciliation_rate': 0}
        
        # Advanced Approach
        try:
            start_time = time.time()
            adv_summary = orchestrator.advanced_reconciler.get_advanced_reconciliation_summary(sample_txns, sample_targets)
            adv_time = time.time() - start_time
            
            size_results['advanced'] = {
                'execution_time': adv_time,
                'total_matches': adv_summary['reconciliation_metrics']['matched_transactions'],
                'reconciliation_rate': adv_summary['reconciliation_metrics']['reconciliation_rate_transactions']
            }
        except Exception as e:
            logger.warning(f"Advanced approach failed for size {size}: {e}")
            size_results['advanced'] = {'execution_time': 0, 'total_matches': 0, 'reconciliation_rate': 0}
        
        benchmark_results[size] = size_results
    
    # Store results in performance analyzer
    orchestrator.performance_analyzer.results['benchmark'] = benchmark_results
    
    # Display benchmark summary
    print("\n" + "="*60)
    print("BENCHMARK ANALYSIS SUMMARY")
    print("="*60)
    
    performance_summary = orchestrator.performance_analyzer.get_performance_summary()
    
    print(f"Total Benchmark Runs: {performance_summary.get('total_benchmark_runs', 0)}")
    print(f"Approaches Tested: {', '.join(performance_summary.get('approaches_tested', []))}")
    print(f"Dataset Sizes Tested: {performance_summary.get('dataset_sizes_tested', [])}")
    print(f"Fastest Approach: {performance_summary.get('fastest_approach', 'N/A')}")
    print(f"Most Accurate Approach: {performance_summary.get('most_accurate_approach', 'N/A')}")
    print(f"Best Performing Approach: {performance_summary.get('best_performing_approach', 'N/A')}")
    
    # Display detailed results by dataset size
    print("\n" + "="*60)
    print("DETAILED BENCHMARK RESULTS")
    print("="*60)
    
    for size, results in benchmark_results.items():
        print(f"\nDataset Size: {size}")
        print("-" * 40)
        
        for approach, metrics in results.items():
            print(f"{approach.replace('_', ' ').title()}:")
            print(f"  - Execution Time: {metrics['execution_time']:.3f}s")
            print(f"  - Total Matches: {metrics['total_matches']}")
            print(f"  - Reconciliation Rate: {metrics['reconciliation_rate']:.2f}%")
    
    return benchmark_results

def demonstrate_visualization_capabilities(orchestrator):
    """Demonstrate visualization capabilities."""
    logger = logging.getLogger(__name__)
    
    # Create visualizations using the same orchestrator instance
    logger.info("Creating performance visualizations...")
    try:
        visualization_files = orchestrator.performance_analyzer.create_performance_visualizations("reports")
        
        print("\n" + "="*60)
        print("VISUALIZATION FILES CREATED")
        print("="*60)
        
        for viz_type, file_path in visualization_files.items():
            print(f"{viz_type.replace('_', ' ').title()}: {file_path}")
        
        print(f"\nPerformance visualizations created successfully")
        
        return visualization_files
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        return {}

def main():
    """Main function demonstrating performance analysis capabilities."""
    logger = setup_logging()
    logger.info("Starting Performance Demonstration")
    
    try:
        # Initialize orchestrator once
        orchestrator = ReconciliationOrchestrator(tolerance=0.01, logger=logger)
        
        # Demonstrate benchmarking
        logger.info("="*60)
        logger.info("DEMONSTRATING BENCHMARKING")
        logger.info("="*60)
        benchmark_results = demonstrate_benchmarking(orchestrator)
        
        # Demonstrate visualization capabilities
        logger.info("="*60)
        logger.info("DEMONSTRATING VISUALIZATION CAPABILITIES")
        logger.info("="*60)
        visualization_files = demonstrate_visualization_capabilities(orchestrator)
        
        # Create summary report
        print("\n" + "="*60)
        print("PERFORMANCE DEMONSTRATION SUMMARY")
        print("="*60)
        print("✓ Benchmarking completed")
        print("✓ Visualizations created")
        print(f"✓ {len(visualization_files)} visualization files generated")
        
        logger.info("Performance demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in performance demonstration: {e}")
        raise

if __name__ == "__main__":
    main() 