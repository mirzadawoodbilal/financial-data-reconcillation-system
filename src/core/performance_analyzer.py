"""
Performance Analyzer Module
Handles benchmarking, visualization, and reporting for reconciliation approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

class PerformanceAnalyzer:
    """Analyzes and visualizes performance of different reconciliation approaches."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.results = {}
        
    def benchmark_all_approaches(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame,
                               dataset_sizes: List[int] = [10, 25, 50, 100, 200]) -> Dict:
        """
        Benchmark all reconciliation approaches across different dataset sizes.
        
        Args:
            transactions_df: Full transactions DataFrame
            targets_df: Full targets DataFrame
            dataset_sizes: List of dataset sizes to test
            
        Returns:
            Dictionary with comprehensive benchmarking results
        """
        benchmark_results = {}
        
        for size in dataset_sizes:
            if size > len(transactions_df) or size > len(targets_df):
                continue
                
            self.logger.info(f"Benchmarking with dataset size: {size}")
            
            # Sample data
            sample_txns = transactions_df.sample(n=min(size, len(transactions_df)), random_state=42)
            sample_targets = targets_df.sample(n=min(size, len(targets_df)), random_state=42)
            
            size_results = {}
            
            # Brute Force Approach
            try:
                from .brute_force_reconciler import BruteForceReconciler
                bf_reconciler = BruteForceReconciler()
                
                start_time = time.time()
                bf_summary = bf_reconciler.get_reconciliation_summary(sample_txns, sample_targets)
                bf_time = time.time() - start_time
                
                size_results['brute_force'] = {
                    'execution_time': bf_time,
                    'total_matches': bf_summary['reconciliation_metrics']['matched_transactions'],
                    'reconciliation_rate': bf_summary['reconciliation_metrics']['reconciliation_rate_transactions'],
                    'direct_matches': bf_summary['direct_matching']['total_matches'],
                    'subset_matches': bf_summary['subset_sum_matching']['total_subset_matches']
                }
            except Exception as e:
                self.logger.warning(f"Brute force failed for size {size}: {e}")
                size_results['brute_force'] = {'execution_time': 0, 'total_matches': 0, 'reconciliation_rate': 0}
            
            # Machine Learning Approach
            try:
                from .ml_reconciler import MLReconciler
                ml_reconciler = MLReconciler()
                
                start_time = time.time()
                ml_summary = ml_reconciler.get_ml_reconciliation_summary(sample_txns, sample_targets)
                ml_time = time.time() - start_time
                
                # Calculate reconciliation rate as percentage of transactions matched
                high_confidence_matches = ml_summary['predictions']['high_confidence_matches']
                reconciliation_rate = min((high_confidence_matches / len(sample_txns)) * 100, 100.0)  # Cap at 100%
                
                size_results['machine_learning'] = {
                    'execution_time': ml_time,
                    'total_matches': high_confidence_matches,
                    'reconciliation_rate': reconciliation_rate,
                    'dp_matches': ml_summary['dynamic_programming']['total_subset_matches'],
                    'best_model': ml_summary['predictions']['best_model']
                }
            except Exception as e:
                self.logger.warning(f"Machine learning failed for size {size}: {e}")
                size_results['machine_learning'] = {'execution_time': 0, 'total_matches': 0, 'reconciliation_rate': 0}
            
            # Advanced Approach
            try:
                from .advanced_reconciler import AdvancedReconciler
                adv_reconciler = AdvancedReconciler()
                
                start_time = time.time()
                adv_summary = adv_reconciler.get_advanced_reconciliation_summary(sample_txns, sample_targets)
                adv_time = time.time() - start_time
                
                size_results['advanced'] = {
                    'execution_time': adv_time,
                    'total_matches': adv_summary['reconciliation_metrics']['matched_transactions'],
                    'reconciliation_rate': adv_summary['reconciliation_metrics']['reconciliation_rate_transactions'],
                    'ga_matches': adv_summary['genetic_algorithm']['total_ga_matches'],
                    'fuzzy_matches': adv_summary['fuzzy_matching']['total_fuzzy_matches'],
                    'cluster_matches': adv_summary['clustering']['total_cluster_matches']
                }
            except Exception as e:
                self.logger.warning(f"Advanced approach failed for size {size}: {e}")
                size_results['advanced'] = {'execution_time': 0, 'total_matches': 0, 'reconciliation_rate': 0}
            
            benchmark_results[size] = size_results
        
        self.results['benchmark'] = benchmark_results
        return benchmark_results
    
    def create_performance_visualizations(self, output_dir: str = "reports") -> Dict:
        """
        Create comprehensive performance visualizations.
        
        Args:
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary with file paths of created visualizations
        """
        if 'benchmark' not in self.results:
            self.logger.warning("No benchmark results available for visualization")
            return {}
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        visualization_files = {}
        
        # 1. Execution Time Comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        approaches = ['brute_force', 'machine_learning', 'advanced']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, approach in enumerate(approaches):
            sizes = []
            times = []
            for size, results in self.results['benchmark'].items():
                if results is None:
                    continue
                if approach in results:
                    metrics = results[approach]
                    if isinstance(metrics, dict) and 'execution_time' in metrics:
                        sizes.append(size)
                        times.append(metrics['execution_time'])
            
            if sizes:
                ax.scatter(sizes, times, marker='o', s=100, 
                          label=approach.replace('_', ' ').title(), color=colors[i], alpha=0.7)
                # Add trend line
                if len(sizes) > 1:
                    z = np.polyfit(sizes, times, 1)
                    p = np.poly1d(z)
                    ax.plot(sizes, p(sizes), color=colors[i], alpha=0.5, linestyle='--')
        
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Execution Time Comparison Across Approaches')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        time_plot_path = f"{output_dir}/execution_time_comparison.png"
        plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files['execution_time'] = time_plot_path
        
        # 2. Reconciliation Rate Comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, approach in enumerate(approaches):
            sizes = []
            rates = []
            for size, results in self.results['benchmark'].items():
                if results is None:
                    continue
                if approach in results:
                    metrics = results[approach]
                    if isinstance(metrics, dict) and 'reconciliation_rate' in metrics:
                        sizes.append(size)
                        rates.append(metrics['reconciliation_rate'])
            
            if sizes:
                ax.scatter(sizes, rates, marker='s', s=100,
                          label=approach.replace('_', ' ').title(), color=colors[i], alpha=0.7)
                # Add trend line
                if len(sizes) > 1:
                    z = np.polyfit(sizes, rates, 1)
                    p = np.poly1d(z)
                    ax.plot(sizes, p(sizes), color=colors[i], alpha=0.5, linestyle='--')
        
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Reconciliation Rate (%)')
        ax.set_title('Reconciliation Rate Comparison Across Approaches')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        rate_plot_path = f"{output_dir}/reconciliation_rate_comparison.png"
        plt.savefig(rate_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files['reconciliation_rate'] = rate_plot_path
        
        # 3. Heatmap of Performance Metrics
        self._create_performance_heatmap(output_dir)
        visualization_files['performance_heatmap'] = f"{output_dir}/performance_heatmap.png"
        
        return visualization_files
    

    
    def _create_performance_heatmap(self, output_dir: str):
        """Create performance heatmap."""
        # Prepare data for heatmap
        heatmap_data = []
        
        for size, results in self.results['benchmark'].items():
            if results is None:
                continue
            for approach, metrics in results.items():
                if isinstance(metrics, dict) and 'execution_time' in metrics:
                    heatmap_data.append({
                        'Dataset_Size': size,
                        'Approach': approach.replace('_', ' ').title(),
                        'Execution_Time': metrics['execution_time'],
                        'Reconciliation_Rate': metrics['reconciliation_rate'],
                        'Total_Matches': metrics['total_matches']
                    })
        
        if not heatmap_data:
            self.logger.warning("No valid data for heatmap")
            return
        
        df = pd.DataFrame(heatmap_data)
        
        # Create pivot tables for heatmaps
        time_pivot = df.pivot(index='Dataset_Size', columns='Approach', values='Execution_Time')
        rate_pivot = df.pivot(index='Dataset_Size', columns='Approach', values='Reconciliation_Rate')
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Execution Time Heatmap
        sns.heatmap(time_pivot, annot=True, fmt='.2f', cmap='Reds', ax=ax1, cbar_kws={'label': 'Time (seconds)'})
        ax1.set_title('Execution Time Heatmap (seconds)')
        ax1.set_xlabel('Approach')
        ax1.set_ylabel('Dataset Size')
        
        # Reconciliation Rate Heatmap
        sns.heatmap(rate_pivot, annot=True, fmt='.1f', cmap='Greens', ax=ax2, cbar_kws={'label': 'Rate (%)'})
        ax2.set_title('Reconciliation Rate Heatmap (%)')
        ax2.set_xlabel('Approach')
        ax2.set_ylabel('Dataset Size')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    

    
    def get_performance_summary(self) -> Dict:
        """
        Get summary of performance analysis.
        
        Returns:
            Dictionary with performance summary
        """
        if 'benchmark' not in self.results:
            return {}
        
        summary = {
            'total_benchmark_runs': 0,
            'approaches_tested': [],
            'dataset_sizes_tested': [],
            'best_performing_approach': None,
            'fastest_approach': None,
            'most_accurate_approach': None
        }
        
        all_metrics = {}
        
        for size, results in self.results['benchmark'].items():
            if results is None:
                continue
            summary['dataset_sizes_tested'].append(size)
            
            if isinstance(results, dict):
                for approach, metrics in results.items():
                    if isinstance(metrics, dict) and 'execution_time' in metrics:
                        if approach not in all_metrics:
                            all_metrics[approach] = {'times': [], 'rates': [], 'matches': []}
                        
                        all_metrics[approach]['times'].append(metrics['execution_time'])
                        all_metrics[approach]['rates'].append(metrics['reconciliation_rate'])
                        all_metrics[approach]['matches'].append(metrics['total_matches'])
                        summary['total_benchmark_runs'] += 1
                    elif isinstance(metrics, str):
                        # Handle error cases
                        continue
        
        summary['approaches_tested'] = list(all_metrics.keys())
        
        # Find best performers
        if all_metrics:
            avg_times = {k: np.mean(v['times']) for k, v in all_metrics.items()}
            avg_rates = {k: np.mean(v['rates']) for k, v in all_metrics.items()}
            total_matches = {k: sum(v['matches']) for k, v in all_metrics.items()}
            
            summary['fastest_approach'] = min(avg_times, key=avg_times.get)
            summary['most_accurate_approach'] = max(avg_rates, key=avg_rates.get)
            summary['best_performing_approach'] = max(total_matches, key=total_matches.get)
        
        return summary
