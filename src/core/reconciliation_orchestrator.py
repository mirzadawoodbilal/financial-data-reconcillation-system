"""
Reconciliation Orchestrator Module
Main orchestrator that coordinates all reconciliation approaches and provides unified interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
from pathlib import Path
import json

from .excel_processor import ExcelProcessor
from .brute_force_reconciler import BruteForceReconciler
from .ml_reconciler import MLReconciler
from .advanced_reconciler import AdvancedReconciler
from .performance_analyzer import PerformanceAnalyzer

class ReconciliationOrchestrator:
    """Main orchestrator for financial reconciliation system."""
    
    def __init__(self, tolerance: float = 0.01, logger: Optional[logging.Logger] = None):
        self.tolerance = tolerance
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize all components
        self.excel_processor = ExcelProcessor(logger)
        self.brute_force_reconciler = BruteForceReconciler(tolerance, logger)
        self.ml_reconciler = MLReconciler(tolerance, logger)
        self.advanced_reconciler = AdvancedReconciler(tolerance, logger)
        self.performance_analyzer = PerformanceAnalyzer(logger)
        
        self.results = {}
        
    def load_and_prepare_data(self, file_path: str, sheet1_name: str = None, 
                            sheet2_name: str = None, amount_col1: str = 'A', 
                            desc_col1: str = 'B', amount_col2: str = 'C', 
                            ref_col2: str = 'D') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for reconciliation.
        
        Args:
            file_path: Path to Excel file
            sheet1_name: Name of first sheet (transactions)
            sheet2_name: Name of second sheet (targets)
            amount_col1: Column name for transaction amounts
            desc_col1: Column name for transaction descriptions
            amount_col2: Column name for target amounts
            ref_col2: Column name for target references
            
        Returns:
            Tuple of (transactions_df, targets_df)
        """
        self.logger.info("Loading and preparing data...")
        
        # Load raw data
        raw_transactions, raw_targets = self.excel_processor.load_excel_sheets(
            file_path, sheet1_name, sheet2_name
        )
        
        # Prepare data
        transactions_df = self.excel_processor.prepare_transactions_data(
            raw_transactions, amount_col1, desc_col1
        )
        targets_df = self.excel_processor.prepare_targets_data(
            raw_targets, amount_col2, ref_col2
        )
        
        # Get data summary
        data_summary = self.excel_processor.get_data_summary(transactions_df, targets_df)
        self.logger.info(f"Data loaded: {data_summary['transactions_count']} transactions, {data_summary['targets_count']} targets")
        
        return transactions_df, targets_df
    
    def run_comprehensive_reconciliation(self, transactions_df: pd.DataFrame, 
                                       targets_df: pd.DataFrame) -> Dict:
        """
        Run comprehensive reconciliation using all approaches.
        
        Args:
            transactions_df: Processed transactions DataFrame
            targets_df: Processed targets DataFrame
            
        Returns:
            Dictionary with comprehensive reconciliation results
        """
        self.logger.info("Starting comprehensive reconciliation...")
        start_time = time.time()
        
        comprehensive_results = {
            'data_summary': self.excel_processor.get_data_summary(transactions_df, targets_df),
            'approaches': {},
            'comparison': {},
            'recommendations': {}
        }
        
        # 1. Brute Force Approach
        self.logger.info("Running brute force reconciliation...")
        try:
            bf_results = self.brute_force_reconciler.get_reconciliation_summary(transactions_df, targets_df)
            comprehensive_results['approaches']['brute_force'] = bf_results
            self.logger.info(f"Brute force completed: {bf_results['reconciliation_metrics']['matched_transactions']} matches")
        except Exception as e:
            self.logger.error(f"Brute force failed: {e}")
            comprehensive_results['approaches']['brute_force'] = {'error': str(e)}
        
        # 2. Machine Learning Approach
        self.logger.info("Running machine learning reconciliation...")
        try:
            ml_results = self.ml_reconciler.get_ml_reconciliation_summary(transactions_df, targets_df)
            comprehensive_results['approaches']['machine_learning'] = ml_results
            self.logger.info(f"ML completed: {ml_results['predictions']['high_confidence_matches']} high-confidence matches")
        except Exception as e:
            self.logger.error(f"Machine learning failed: {e}")
            comprehensive_results['approaches']['machine_learning'] = {'error': str(e)}
        
        # 3. Advanced Approach
        self.logger.info("Running advanced reconciliation...")
        try:
            adv_results = self.advanced_reconciler.get_advanced_reconciliation_summary(transactions_df, targets_df)
            comprehensive_results['approaches']['advanced'] = adv_results
            self.logger.info(f"Advanced completed: {adv_results['reconciliation_metrics']['matched_transactions']} matches")
        except Exception as e:
            self.logger.error(f"Advanced approach failed: {e}")
            comprehensive_results['approaches']['advanced'] = {'error': str(e)}
        
        # 4. Performance Comparison
        self.logger.info("Running performance comparison...")
        try:
            comparison_results = self._compare_approaches(comprehensive_results['approaches'])
            comprehensive_results['comparison'] = comparison_results
        except Exception as e:
            self.logger.error(f"Performance comparison failed: {e}")
            comprehensive_results['comparison'] = {'error': str(e)}
        
        # 5. Generate Recommendations
        comprehensive_results['recommendations'] = self._generate_recommendations(comprehensive_results)
        
        total_time = time.time() - start_time
        comprehensive_results['total_execution_time'] = total_time
        
        self.logger.info(f"Comprehensive reconciliation completed in {total_time:.2f} seconds")
        
        self.results = comprehensive_results
        return comprehensive_results
    
    def run_benchmark_analysis(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame,
                              dataset_sizes: List[int] = [10, 25, 50, 100, 200]) -> Dict:
        """
        Run comprehensive benchmark analysis.
        
        Args:
            transactions_df: Processed transactions DataFrame
            targets_df: Processed targets DataFrame
            dataset_sizes: List of dataset sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        self.logger.info("Starting benchmark analysis...")
        
        benchmark_results = self.performance_analyzer.benchmark_all_approaches(
            transactions_df, targets_df, dataset_sizes
        )
        
        # Store visualization files separately from benchmark data
        try:
            visualization_files = self.performance_analyzer.create_performance_visualizations()
            self.results['visualization_files'] = visualization_files
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
            self.results['visualization_files'] = {'error': str(e)}
        
        self.results['benchmark'] = benchmark_results
        return benchmark_results
    
    def _compare_approaches(self, approaches_results: Dict) -> Dict:
        """
        Compare different reconciliation approaches.
        
        Args:
            approaches_results: Results from different approaches
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'execution_times': {},
            'reconciliation_rates': {},
            'total_matches': {},
            'best_approach': {},
            'trade_offs': {}
        }
        
        for approach, results in approaches_results.items():
            if 'error' in results:
                continue
                
            # Extract metrics
            if approach == 'brute_force':
                comparison['execution_times'][approach] = results['total_execution_time']
                comparison['reconciliation_rates'][approach] = results['reconciliation_metrics']['reconciliation_rate_transactions']
                comparison['total_matches'][approach] = results['reconciliation_metrics']['matched_transactions']
                
            elif approach == 'machine_learning':
                comparison['execution_times'][approach] = results['total_execution_time']
                comparison['reconciliation_rates'][approach] = (results['predictions']['high_confidence_matches'] / 
                                                             results['feature_engineering']['total_feature_vectors']) * 100
                comparison['total_matches'][approach] = results['predictions']['high_confidence_matches']
                
            elif approach == 'advanced':
                comparison['execution_times'][approach] = results['total_execution_time']
                comparison['reconciliation_rates'][approach] = results['reconciliation_metrics']['reconciliation_rate_transactions']
                comparison['total_matches'][approach] = results['reconciliation_metrics']['matched_transactions']
        
        # Find best approaches
        if comparison['execution_times']:
            comparison['best_approach']['fastest'] = min(comparison['execution_times'], key=comparison['execution_times'].get)
            comparison['best_approach']['most_accurate'] = max(comparison['reconciliation_rates'], key=comparison['reconciliation_rates'].get)
            comparison['best_approach']['most_matches'] = max(comparison['total_matches'], key=comparison['total_matches'].get)
        
        # Analyze trade-offs
        comparison['trade_offs'] = {
            'speed_vs_accuracy': self._analyze_speed_accuracy_tradeoff(comparison),
            'complexity_vs_performance': self._analyze_complexity_performance_tradeoff(comparison)
        }
        
        return comparison
    
    def _analyze_speed_accuracy_tradeoff(self, comparison: Dict) -> Dict:
        """Analyze speed vs accuracy trade-off."""
        if not comparison['execution_times'] or not comparison['reconciliation_rates']:
            return {}
        
        # Calculate efficiency score (accuracy / time)
        efficiency_scores = {}
        for approach in comparison['execution_times'].keys():
            if approach in comparison['reconciliation_rates']:
                efficiency_scores[approach] = comparison['reconciliation_rates'][approach] / comparison['execution_times'][approach]
        
        return {
            'efficiency_scores': efficiency_scores,
            'most_efficient': max(efficiency_scores, key=efficiency_scores.get) if efficiency_scores else None
        }
    
    def _analyze_complexity_performance_tradeoff(self, comparison: Dict) -> Dict:
        """Analyze complexity vs performance trade-off."""
        complexity_rankings = {
            'brute_force': 1,  # Simplest
            'machine_learning': 2,  # Medium complexity
            'advanced': 3  # Most complex
        }
        
        complexity_performance = {}
        for approach in comparison['execution_times'].keys():
            if approach in complexity_rankings:
                complexity_performance[approach] = {
                    'complexity': complexity_rankings[approach],
                    'performance': comparison['reconciliation_rates'].get(approach, 0)
                }
        
        return {
            'complexity_performance': complexity_performance,
            'best_complexity_performance_ratio': min(complexity_performance, 
                                                   key=lambda x: complexity_performance[x]['complexity'] / 
                                                                max(complexity_performance[x]['performance'], 1))
        }
    
    def _generate_recommendations(self, comprehensive_results: Dict) -> Dict:
        """
        Generate recommendations based on results.
        
        Args:
            comprehensive_results: Comprehensive reconciliation results
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'best_approach': {},
            'use_cases': {},
            'optimization_suggestions': {},
            'next_steps': []
        }
        
        comparison = comprehensive_results.get('comparison', {})
        
        if 'best_approach' in comparison:
            recommendations['best_approach'] = {
                'for_speed': comparison['best_approach'].get('fastest'),
                'for_accuracy': comparison['best_approach'].get('most_accurate'),
                'for_efficiency': comparison.get('trade_offs', {}).get('speed_vs_accuracy', {}).get('most_efficient')
            }
        
        # Use case recommendations
        recommendations['use_cases'] = {
            'small_datasets': 'brute_force',
            'large_datasets': 'machine_learning',
            'complex_matching': 'advanced',
            'real_time_processing': 'brute_force',
            'batch_processing': 'machine_learning'
        }
        
        # Optimization suggestions
        recommendations['optimization_suggestions'] = {
            'brute_force': [
                'Consider parallel processing for large datasets',
                'Implement early termination for exact matches',
                'Use hash tables for faster lookups'
            ],
            'machine_learning': [
                'Feature engineering for better accuracy',
                'Hyperparameter tuning for models',
                'Ensemble methods for improved predictions'
            ],
            'advanced': [
                'Optimize genetic algorithm parameters',
                'Fine-tune similarity thresholds',
                'Implement caching for repeated calculations'
            ]
        }
        
        # Next steps
        recommendations['next_steps'] = [
            'Run benchmark analysis for performance comparison',
            'Generate detailed visualizations',
            'Implement production-ready version',
            'Add data validation and error handling',
            'Create user interface for easy interaction'
        ]
        
        return recommendations
    
    def save_results(self, output_dir: str = "results") -> Dict:
        """
        Save all results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save comprehensive results as JSON
        results_file = f"{output_dir}/comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        saved_files['comprehensive_results'] = results_file
        
        # Save processed data
        if 'data_summary' in self.results:
            data_file = f"{output_dir}/processed_data_summary.json"
            with open(data_file, 'w') as f:
                json.dump(self.results['data_summary'], f, indent=2, default=str)
            saved_files['data_summary'] = data_file
        
        # Save recommendations
        if 'recommendations' in self.results:
            rec_file = f"{output_dir}/recommendations.json"
            with open(rec_file, 'w') as f:
                json.dump(self.results['recommendations'], f, indent=2, default=str)
            saved_files['recommendations'] = rec_file
        
        self.logger.info(f"Results saved to {output_dir}")
        return saved_files
    
    def get_summary_report(self) -> str:
        """
        Generate a summary report of all results.
        
        Returns:
            Summary report as string
        """
        if not self.results:
            return "No results available. Run reconciliation first."
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("FINANCIAL RECONCILIATION SUMMARY REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Data Summary
        if 'data_summary' in self.results:
            data_summary = self.results['data_summary']
            report_lines.append("DATA SUMMARY:")
            report_lines.append(f"- Transactions: {data_summary['transactions_count']}")
            report_lines.append(f"- Targets: {data_summary['targets_count']}")
            report_lines.append(f"- Total Transaction Amount: ${data_summary['transactions_total']:,.2f}")
            report_lines.append(f"- Total Target Amount: ${data_summary['targets_total']:,.2f}")
            report_lines.append("")
        
        # Approach Results
        if 'approaches' in self.results:
            report_lines.append("APPROACH RESULTS:")
            for approach, results in self.results['approaches'].items():
                if 'error' in results:
                    report_lines.append(f"- {approach.replace('_', ' ').title()}: ERROR - {results['error']}")
                else:
                    if approach == 'brute_force':
                        matches = results['reconciliation_metrics']['matched_transactions']
                        rate = results['reconciliation_metrics']['reconciliation_rate_transactions']
                        time_taken = results['total_execution_time']
                    elif approach == 'machine_learning':
                        matches = results['predictions']['high_confidence_matches']
                        rate = (matches / results['feature_engineering']['total_feature_vectors']) * 100
                        time_taken = results['total_execution_time']
                    elif approach == 'advanced':
                        matches = results['reconciliation_metrics']['matched_transactions']
                        rate = results['reconciliation_metrics']['reconciliation_rate_transactions']
                        time_taken = results['total_execution_time']
                    else:
                        continue
                    
                    report_lines.append(f"- {approach.replace('_', ' ').title()}: {matches} matches ({rate:.1f}%) in {time_taken:.2f}s")
            report_lines.append("")
        
        # Recommendations
        if 'recommendations' in self.results:
            recs = self.results['recommendations']
            report_lines.append("RECOMMENDATIONS:")
            if 'best_approach' in recs:
                best = recs['best_approach']
                report_lines.append(f"- Best for Speed: {best.get('for_speed', 'N/A')}")
                report_lines.append(f"- Best for Accuracy: {best.get('for_accuracy', 'N/A')}")
                report_lines.append(f"- Most Efficient: {best.get('for_efficiency', 'N/A')}")
            report_lines.append("")
        
        # Total Execution Time
        if 'total_execution_time' in self.results:
            report_lines.append(f"TOTAL EXECUTION TIME: {self.results['total_execution_time']:.2f} seconds")
        
        return "\n".join(report_lines)
