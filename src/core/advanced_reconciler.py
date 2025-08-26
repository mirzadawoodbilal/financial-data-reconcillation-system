"""
Advanced Reconciliation Module
Implements genetic algorithms and fuzzy matching for financial reconciliation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time
import random
from deap import base, creator, tools, algorithms
from fuzzywuzzy import fuzz
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging

class AdvancedReconciler:
    """Implements advanced techniques for financial reconciliation."""
    
    def __init__(self, tolerance: float = 0.01, logger: Optional[logging.Logger] = None):
        self.tolerance = tolerance
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def genetic_algorithm_subset_selection(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame,
                                         population_size: int = 50, generations: int = 100) -> Dict:
        """
        Implement genetic algorithm for subset selection.
        
        Args:
            transactions_df: DataFrame with transaction data
            targets_df: DataFrame with target data
            population_size: Size of genetic algorithm population
            generations: Number of generations to evolve
            
        Returns:
            Dictionary containing genetic algorithm results
        """
        start_time = time.time()
        
        # Get unmatched transactions and targets
        from .brute_force_reconciler import BruteForceReconciler
        bf_reconciler = BruteForceReconciler(self.tolerance, self.logger)
        direct_results = bf_reconciler.direct_matching(transactions_df, targets_df)
        unmatched_txns = direct_results['unmatched_transactions']
        unmatched_targets = direct_results['unmatched_targets']
        
        if not unmatched_txns or not unmatched_targets:
            return {
                'ga_matches': [],
                'execution_time': time.time() - start_time,
                'total_ga_matches': 0
            }
        
        # Create filtered DataFrames
        unmatched_txns_df = transactions_df[transactions_df['transaction_id'].isin(unmatched_txns)]
        unmatched_targets_df = targets_df[targets_df['target_id'].isin(unmatched_targets)]
        
        ga_matches = []
        used_transactions = set()
        
        # Setup genetic algorithm for each target
        for _, target_row in unmatched_targets_df.iterrows():
            target_amount = target_row['target_amount']
            target_id = target_row['target_id']
            
            # Get available transactions
            available_txns = unmatched_txns_df[~unmatched_txns_df['transaction_id'].isin(used_transactions)]
            
            if len(available_txns) == 0:
                continue
            
            # Setup DEAP genetic algorithm
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            
            # Attribute generator
            toolbox.register("attr_bool", random.randint, 0, 1)
            
            # Structure initializers
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 
                           n=len(available_txns))
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evalSubsetSum(individual):
                """Evaluate fitness of individual (subset)."""
                selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
                if not selected_indices:
                    return 0.0,
                
                selected_amounts = available_txns.iloc[selected_indices]['amount'].values
                subset_sum = sum(selected_amounts)
                
                # Fitness based on how close we are to target
                difference = abs(subset_sum - target_amount)
                if difference <= self.tolerance:
                    # Bonus for exact matches
                    fitness = 1000 - difference * 100
                else:
                    # Penalty for being far from target
                    fitness = max(0, 100 - difference)
                
                return fitness,
            
            def cxTwoPoint(ind1, ind2):
                """Two-point crossover."""
                return tools.cxTwoPoint(ind1, ind2)
            
            def mutFlipBit(individual, indpb):
                """Flip bit mutation."""
                return tools.mutFlipBit(individual, indpb)
            
            # Register genetic operators
            toolbox.register("evaluate", evalSubsetSum)
            toolbox.register("mate", cxTwoPoint)
            toolbox.register("mutate", mutFlipBit)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Create initial population
            pop = toolbox.population(n=population_size)
            
            # Track best individual
            best_individual = None
            best_fitness = 0
            
            # Evolution loop
            for gen in range(generations):
                # Select and clone the next generation individuals
                offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))
                offspring = list(offspring)
                
                # Apply crossover and mutation
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.7:  # Crossover probability
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                for mutant in offspring:
                    if random.random() < 0.1:  # Mutation probability
                        toolbox.mutate(mutant, indpb=0.05)
                        del mutant.fitness.values
                
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Replace population
                pop[:] = offspring
                
                # Track best individual
                for ind in pop:
                    if ind.fitness.values[0] > best_fitness:
                        best_fitness = ind.fitness.values[0]
                        best_individual = toolbox.clone(ind)
            
            # Check if we found a good match
            if best_individual and best_fitness > 900:  # High fitness threshold
                selected_indices = [i for i, bit in enumerate(best_individual) if bit == 1]
                if selected_indices:
                    selected_txns = available_txns.iloc[selected_indices]
                    subset_sum = selected_txns['amount'].sum()
                    
                    if abs(subset_sum - target_amount) <= self.tolerance:
                        ga_matches.append({
                            'target_id': target_id,
                            'target_amount': target_amount,
                            'transaction_ids': selected_txns['transaction_id'].tolist(),
                            'transaction_amounts': selected_txns['amount'].tolist(),
                            'combo_sum': subset_sum,
                            'difference': subset_sum - target_amount,
                            'num_transactions': len(selected_txns),
                            'fitness_score': best_fitness
                        })
                        used_transactions.update(selected_txns['transaction_id'].tolist())
            
            # Clean up DEAP creators
            del creator.FitnessMax
            del creator.Individual
        
        execution_time = time.time() - start_time
        
        results = {
            'ga_matches': ga_matches,
            'execution_time': execution_time,
            'total_ga_matches': len(ga_matches),
            'used_transactions': list(used_transactions)
        }
        
        self.logger.info(f"Genetic algorithm completed: {len(ga_matches)} matches found in {execution_time:.4f}s")
        return results
    
    def fuzzy_matching_with_similarity_scores(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame,
                                            similarity_threshold: float = 0.8) -> Dict:
        """
        Implement fuzzy matching with similarity scores for approximate reconciliation.
        
        Args:
            transactions_df: DataFrame with transaction data
            targets_df: DataFrame with target data
            similarity_threshold: Minimum similarity score for matching
            
        Returns:
            Dictionary containing fuzzy matching results
        """
        start_time = time.time()
        
        fuzzy_matches = []
        matched_transactions = set()
        matched_targets = set()
        
        # Create lookup dictionaries
        transactions_dict = transactions_df.set_index('transaction_id').to_dict('index')
        targets_dict = targets_df.set_index('target_id').to_dict('index')
        
        for txn_id, txn_data in transactions_dict.items():
            if txn_id in matched_transactions:
                continue
                
            best_match = None
            best_similarity = 0
            
            for target_id, target_data in targets_dict.items():
                if target_id in matched_targets:
                    continue
                
                # Calculate various similarity metrics
                amount_similarity = 1 - min(abs(txn_data['amount'] - target_data['target_amount']) / 
                                          max(abs(txn_data['amount']), abs(target_data['target_amount'])), 1)
                
                # Text similarity using different algorithms
                desc_similarity = fuzz.ratio(str(txn_data['description']), str(target_data['reference_id'])) / 100
                desc_partial = fuzz.partial_ratio(str(txn_data['description']), str(target_data['reference_id'])) / 100
                desc_token_sort = fuzz.token_sort_ratio(str(txn_data['description']), str(target_data['reference_id'])) / 100
                desc_token_set = fuzz.token_set_ratio(str(txn_data['description']), str(target_data['reference_id'])) / 100
                
                # Weighted similarity score
                text_similarity = (desc_similarity * 0.3 + desc_partial * 0.3 + 
                                 desc_token_sort * 0.2 + desc_token_set * 0.2)
                
                # Combined similarity score
                combined_similarity = (amount_similarity * 0.7 + text_similarity * 0.3)
                
                if combined_similarity > best_similarity:
                    best_similarity = combined_similarity
                    best_match = {
                        'target_id': target_id,
                        'target_data': target_data,
                        'similarity_scores': {
                            'amount_similarity': amount_similarity,
                            'text_similarity': text_similarity,
                            'combined_similarity': combined_similarity,
                            'desc_similarity': desc_similarity,
                            'desc_partial': desc_partial,
                            'desc_token_sort': desc_token_sort,
                            'desc_token_set': desc_token_set
                        }
                    }
            
            # Check if best match meets threshold
            if best_match and best_match['similarity_scores']['combined_similarity'] >= similarity_threshold:
                fuzzy_matches.append({
                    'transaction_id': txn_id,
                    'target_id': best_match['target_id'],
                    'transaction_amount': txn_data['amount'],
                    'target_amount': best_match['target_data']['target_amount'],
                    'transaction_description': txn_data['description'],
                    'target_reference': best_match['target_data']['reference_id'],
                    'difference': txn_data['amount'] - best_match['target_data']['target_amount'],
                    'similarity_scores': best_match['similarity_scores'],
                    'match_type': 'fuzzy'
                })
                matched_transactions.add(txn_id)
                matched_targets.add(best_match['target_id'])
        
        execution_time = time.time() - start_time
        
        results = {
            'fuzzy_matches': fuzzy_matches,
            'matched_transactions': list(matched_transactions),
            'matched_targets': list(matched_targets),
            'unmatched_transactions': [txn_id for txn_id in transactions_dict.keys() 
                                     if txn_id not in matched_transactions],
            'unmatched_targets': [target_id for target_id in targets_dict.keys() 
                                if target_id not in matched_targets],
            'execution_time': execution_time,
            'total_fuzzy_matches': len(fuzzy_matches),
            'average_similarity': np.mean([match['similarity_scores']['combined_similarity'] 
                                         for match in fuzzy_matches]) if fuzzy_matches else 0
        }
        
        self.logger.info(f"Fuzzy matching completed: {len(fuzzy_matches)} matches found in {execution_time:.4f}s")
        self.logger.info(f"Average similarity score: {results['average_similarity']:.3f}")
        return results
    
    def clustering_based_reconciliation(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame,
                                      eps: float = 0.1, min_samples: int = 2) -> Dict:
        """
        Implement clustering-based reconciliation using DBSCAN.
        
        Args:
            transactions_df: DataFrame with transaction data
            targets_df: DataFrame with target data
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
            
        Returns:
            Dictionary containing clustering results
        """
        start_time = time.time()
        
        # Combine transactions and targets for clustering
        combined_data = []
        
        # Add transactions
        for _, row in transactions_df.iterrows():
            combined_data.append({
                'id': row['transaction_id'],
                'amount': row['amount'],
                'description': row['description'],
                'type': 'transaction'
            })
        
        # Add targets
        for _, row in targets_df.iterrows():
            combined_data.append({
                'id': row['target_id'],
                'amount': row['target_amount'],
                'description': row['reference_id'],
                'type': 'target'
            })
        
        combined_df = pd.DataFrame(combined_data)
        
        # Prepare features for clustering
        features = combined_df[['amount']].values
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_scaled)
        combined_df['cluster'] = clustering.labels_
        
        # Analyze clusters
        cluster_matches = []
        used_transactions = set()
        used_targets = set()
        
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
                
            cluster_data = combined_df[combined_df['cluster'] == cluster_id]
            transactions_in_cluster = cluster_data[cluster_data['type'] == 'transaction']
            targets_in_cluster = cluster_data[cluster_data['type'] == 'target']
            
            if len(transactions_in_cluster) > 0 and len(targets_in_cluster) > 0:
                # Find best matches within cluster
                for _, txn in transactions_in_cluster.iterrows():
                    if txn['id'] in used_transactions:
                        continue
                        
                    best_target = None
                    best_difference = float('inf')
                    
                    for _, target in targets_in_cluster.iterrows():
                        if target['id'] in used_targets:
                            continue
                            
                        difference = abs(txn['amount'] - target['amount'])
                        if difference < best_difference and difference <= self.tolerance:
                            best_difference = difference
                            best_target = target
                    
                    if best_target:
                        cluster_matches.append({
                            'transaction_id': txn['id'],
                            'target_id': best_target['id'],
                            'transaction_amount': float(txn['amount']),
                            'target_amount': float(best_target['amount']),
                            'cluster_id': cluster_id,
                            'difference': float(txn['amount']) - float(best_target['amount']),
                            'match_type': 'cluster'
                        })
                        used_transactions.add(txn['id'])
                        used_targets.add(best_target['id'])
        
        execution_time = time.time() - start_time
        
        results = {
            'cluster_matches': cluster_matches,
            'total_clusters': len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0),
            'noise_points': len(combined_df[combined_df['cluster'] == -1]),
            'execution_time': execution_time,
            'total_cluster_matches': len(cluster_matches),
            'used_transactions': list(used_transactions),
            'used_targets': list(used_targets)
        }
        
        self.logger.info(f"Clustering-based reconciliation completed: {len(cluster_matches)} matches found in {execution_time:.4f}s")
        return results
    
    def get_advanced_reconciliation_summary(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict:
        """
        Get comprehensive advanced reconciliation summary.
        
        Args:
            transactions_df: Processed transactions DataFrame
            targets_df: Processed targets DataFrame
            
        Returns:
            Dictionary with comprehensive advanced reconciliation results
        """
        start_time = time.time()
        
        # Genetic algorithm subset selection
        ga_results = self.genetic_algorithm_subset_selection(transactions_df, targets_df)
        
        # Fuzzy matching
        fuzzy_results = self.fuzzy_matching_with_similarity_scores(transactions_df, targets_df)
        
        # Clustering-based reconciliation
        cluster_results = self.clustering_based_reconciliation(transactions_df, targets_df)
        
        total_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        total_transactions = len(transactions_df)
        total_targets = len(targets_df)
        
        # Combine all matched transactions
        all_matched_txns = set()
        all_matched_targets = set()
        
        all_matched_txns.update(ga_results['used_transactions'])
        all_matched_txns.update(fuzzy_results['matched_transactions'])
        all_matched_txns.update(cluster_results['used_transactions'])
        
        all_matched_targets.update([match['target_id'] for match in ga_results['ga_matches']])
        all_matched_targets.update(fuzzy_results['matched_targets'])
        all_matched_targets.update(cluster_results['used_targets'])
        
        reconciliation_rate_txns = (len(all_matched_txns) / total_transactions) * 100 if total_transactions > 0 else 0
        reconciliation_rate_targets = (len(all_matched_targets) / total_targets) * 100 if total_targets > 0 else 0
        
        summary = {
            'total_execution_time': total_time,
            'genetic_algorithm': ga_results,
            'fuzzy_matching': fuzzy_results,
            'clustering': cluster_results,
            'reconciliation_metrics': {
                'total_transactions': total_transactions,
                'total_targets': total_targets,
                'matched_transactions': len(all_matched_txns),
                'matched_targets': len(all_matched_targets),
                'reconciliation_rate_transactions': reconciliation_rate_txns,
                'reconciliation_rate_targets': reconciliation_rate_targets,
                'unmatched_transactions': total_transactions - len(all_matched_txns),
                'unmatched_targets': total_targets - len(all_matched_targets)
            }
        }
        
        self.logger.info(f"Advanced reconciliation completed in {total_time:.4f}s")
        self.logger.info(f"Reconciliation rate: {reconciliation_rate_txns:.2f}% transactions, {reconciliation_rate_targets:.2f}% targets")
        
        return summary
