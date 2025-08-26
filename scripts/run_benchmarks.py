#!/usr/bin/env python3
"""
Benchmark Runner Script
Executes comprehensive benchmarks for the financial reconciliation system.
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.reconciliation_orchestrator import ReconciliationOrchestrator

def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark.log')
        ]
    )
    return logging.getLogger(__name__)

def run_basic_benchmark(logger, dataset_sizes=None):
    """Run basic benchmark with sample data."""
    if dataset_sizes is None:
        dataset_sizes = [10, 25, 50, 100]
    
    logger.info("Starting basic benchmark...")
    
    # Initialize orchestrator
    orchestrator = ReconciliationOrchestrator(tolerance=0.01, logger=logger)
    
    # Generate large dataset for sampling
    logger.info("Generating large dataset for sampling...")
    transactions_df, targets_df = orchestrator.excel_processor.generate_sample_data(
        num_transactions=1000, 
        num_targets=200
    )
    
    # Run benchmark analysis
    logger.info(f"Running benchmark with dataset sizes: {dataset_sizes}")
    benchmark_results = orchestrator.run_benchmark_analysis(
        transactions_df, targets_df, dataset_sizes
    )
    
    # Display results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    performance_summary = orchestrator.performance_analyzer.get_performance_summary()
    print(f"Total Benchmark Runs: {performance_summary.get('total_benchmark_runs', 0)}")
    print(f"Fastest Approach: {performance_summary.get('fastest_approach', 'N/A')}")
    print(f"Most Accurate Approach: {performance_summary.get('most_accurate_approach', 'N/A')}")
    
    return benchmark_results

def run_comprehensive_benchmark(logger, file_path=None):
    """Run comprehensive benchmark with real data if available."""
    logger.info("Starting comprehensive benchmark...")
    
    # Initialize orchestrator
    orchestrator = ReconciliationOrchestrator(tolerance=0.01, logger=logger)
    
    if file_path and os.path.exists(file_path):
        logger.info(f"Using real data from: {file_path}")
        # Load real data
        transactions_df, targets_df = orchestrator.load_and_prepare_data(file_path)
    else:
        logger.info("Using generated sample data")
        # Generate sample data
        transactions_df, targets_df = orchestrator.excel_processor.generate_sample_data(
            num_transactions=200, 
            num_targets=40
        )
    
    # Run comprehensive reconciliation
    logger.info("Running comprehensive reconciliation...")
    results = orchestrator.run_comprehensive_reconciliation(transactions_df, targets_df)
    
    # Display summary
    print("\n" + "="*60)
    print("COMPREHENSIVE RECONCILIATION RESULTS")
    print("="*60)
    print(orchestrator.get_summary_report())
    
    # Save results
    saved_files = orchestrator.save_results("results/benchmark_run")
    logger.info(f"Results saved to: {list(saved_files.values())}")
    
    return results

def run_performance_analysis(logger):
    """Run detailed performance analysis."""
    logger.info("Starting performance analysis...")
    
    # Initialize orchestrator
    orchestrator = ReconciliationOrchestrator(tolerance=0.01, logger=logger)
    
    # Generate data for analysis
    transactions_df, targets_df = orchestrator.excel_processor.generate_sample_data(
        num_transactions=300, 
        num_targets=60
    )
    
    # Run benchmark with multiple sizes
    dataset_sizes = [10, 25, 50, 100, 200, 300]
    benchmark_results = orchestrator.run_benchmark_analysis(
        transactions_df, targets_df, dataset_sizes
    )
    
    # Create visualizations
    logger.info("Creating performance visualizations...")
    try:
        visualization_files = orchestrator.performance_analyzer.create_performance_visualizations(
            "reports/benchmark_analysis"
        )
        
        print("\n" + "="*60)
        print("VISUALIZATION FILES CREATED")
        print("="*60)
        for viz_type, file_path in visualization_files.items():
            print(f"{viz_type.replace('_', ' ').title()}: {file_path}")
        
        # Performance report generation removed - only visualizations
        print(f"\nPerformance visualizations created successfully")
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
    
    return benchmark_results

def main():
    """Main function for benchmark runner."""
    parser = argparse.ArgumentParser(description='Run financial reconciliation benchmarks')
    parser.add_argument('--mode', choices=['basic', 'comprehensive', 'performance'], 
                       default='basic', help='Benchmark mode to run')
    parser.add_argument('--file', type=str, help='Path to Excel file for real data testing')
    parser.add_argument('--sizes', nargs='+', type=int, 
                       default=[10, 25, 50, 100], help='Dataset sizes to test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info(f"Starting benchmark runner in {args.mode} mode")
    
    try:
        if args.mode == 'basic':
            results = run_basic_benchmark(logger, args.sizes)
        elif args.mode == 'comprehensive':
            results = run_comprehensive_benchmark(logger, args.file)
        elif args.mode == 'performance':
            results = run_performance_analysis(logger)
        
        logger.info("Benchmark completed successfully!")
        
        # Display final summary
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Mode: {args.mode}")
        print(f"Dataset sizes tested: {args.sizes}")
        if args.file:
            print(f"Data file: {args.file}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 