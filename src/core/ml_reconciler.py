"""
Machine Learning Reconciliation Module
Implements ML approaches for financial reconciliation including feature engineering,
dynamic programming, and predictive models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import logging

class MLReconciler:
    """Implements machine learning approaches for financial reconciliation."""
    
    def __init__(self, tolerance: float = 0.01, logger: Optional[logging.Logger] = None):
        self.tolerance = tolerance
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.models = {}
        
    def feature_engineering(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform reconciliation problem into ML features.
        
        Args:
            transactions_df: DataFrame with transaction data
            targets_df: DataFrame with target data
            
        Returns:
            DataFrame with engineered features
        """
        features_list = []
        
        # Create cartesian product of transactions and targets
        for _, txn in transactions_df.iterrows():
            for _, target in targets_df.iterrows():
                # Basic amount features
                amount_diff = abs(txn['amount'] - target['target_amount'])
                amount_ratio = txn['amount'] / target['target_amount'] if target['target_amount'] != 0 else 0
                amount_sum = txn['amount'] + target['target_amount']
                amount_product = txn['amount'] * target['target_amount']
                
                # Text similarity features
                desc_similarity = fuzz.ratio(str(txn['description']), str(target['reference_id'])) / 100
                desc_partial = fuzz.partial_ratio(str(txn['description']), str(target['reference_id'])) / 100
                desc_token_sort = fuzz.token_sort_ratio(str(txn['description']), str(target['reference_id'])) / 100
                
                # Statistical features
                amount_std_diff = abs(txn['amount'] - targets_df['target_amount'].mean())
                amount_percentile_diff = abs(txn['amount'] - targets_df['target_amount'].quantile(0.5))
                
                # Categorical features
                txn_sign = 1 if txn['amount'] > 0 else -1
                target_sign = 1 if target['target_amount'] > 0 else -1
                sign_match = 1 if txn_sign == target_sign else 0
                
                # Create feature vector
                features = {
                    'transaction_id': txn['transaction_id'],
                    'target_id': target['target_id'],
                    'transaction_amount': txn['amount'],
                    'target_amount': target['target_amount'],
                    'amount_diff': amount_diff,
                    'amount_ratio': amount_ratio,
                    'amount_sum': amount_sum,
                    'amount_product': amount_product,
                    'desc_similarity': desc_similarity,
                    'desc_partial': desc_partial,
                    'desc_token_sort': desc_token_sort,
                    'amount_std_diff': amount_std_diff,
                    'amount_percentile_diff': amount_percentile_diff,
                    'sign_match': sign_match,
                    'txn_sign': txn_sign,
                    'target_sign': target_sign,
                    'is_exact_match': 1 if amount_diff <= self.tolerance else 0
                }
                
                features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Add derived features
        features_df['amount_diff_normalized'] = features_df['amount_diff'] / features_df['target_amount'].abs()
        features_df['amount_ratio_log'] = np.log(features_df['amount_ratio'].abs() + 1)
        features_df['similarity_avg'] = (features_df['desc_similarity'] + features_df['desc_partial'] + features_df['desc_token_sort']) / 3
        
        self.logger.info(f"Engineered {len(features_df)} feature vectors")
        return features_df
    
    def dynamic_programming_subset_sum(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict:
        """
        Implement optimized subset sum solution using dynamic programming.
        
        Args:
            transactions_df: DataFrame with transaction data
            targets_df: DataFrame with target data
            
        Returns:
            Dictionary containing subset sum results
        """
        start_time = time.time()
        
        subset_matches = []
        used_transactions = set()
        
        # Get unmatched transactions and targets
        from .brute_force_reconciler import BruteForceReconciler
        bf_reconciler = BruteForceReconciler(self.tolerance, self.logger)
        direct_results = bf_reconciler.direct_matching(transactions_df, targets_df)
        unmatched_txns = direct_results['unmatched_transactions']
        unmatched_targets = direct_results['unmatched_targets']
        
        if not unmatched_txns or not unmatched_targets:
            return {
                'subset_matches': [],
                'execution_time': time.time() - start_time,
                'total_subset_matches': 0
            }
        
        # Create filtered DataFrames
        unmatched_txns_df = transactions_df[transactions_df['transaction_id'].isin(unmatched_txns)]
        unmatched_targets_df = targets_df[targets_df['target_id'].isin(unmatched_targets)]
        
        def dp_subset_sum(amounts, target):
            """Dynamic programming subset sum implementation."""
            n = len(amounts)
            dp = [[False] * (target + 1) for _ in range(n + 1)]
            
            # Base case: empty subset sums to 0
            for i in range(n + 1):
                dp[i][0] = True
            
            # Fill the dp table
            for i in range(1, n + 1):
                for j in range(1, target + 1):
                    if amounts[i-1] <= j:
                        dp[i][j] = dp[i-1][j] or dp[i-1][j - amounts[i-1]]
                    else:
                        dp[i][j] = dp[i-1][j]
            
            # Backtrack to find the subset
            if dp[n][target]:
                subset = []
                i, j = n, target
                while i > 0 and j > 0:
                    if dp[i-1][j]:
                        i -= 1
                    else:
                        subset.append(i-1)
                        j -= amounts[i-1]
                        i -= 1
                return subset[::-1]
            return None
        
        for _, target_row in unmatched_targets_df.iterrows():
            target_amount = int(target_row['target_amount'] * 100)  # Convert to cents for integer DP
            target_id = target_row['target_id']
            
            # Get available transactions
            available_txns = unmatched_txns_df[~unmatched_txns_df['transaction_id'].isin(used_transactions)]
            
            if len(available_txns) == 0:
                continue
            
            # Convert amounts to integers (cents)
            amounts = [int(abs(amt) * 100) for amt in available_txns['amount'].values]
            
            # Find subset that sums to target
            subset_indices = dp_subset_sum(amounts, target_amount)
            
            if subset_indices:
                subset_txns = available_txns.iloc[subset_indices]
                subset_sum = subset_txns['amount'].sum()
                
                subset_matches.append({
                    'target_id': target_id,
                    'target_amount': target_row['target_amount'],
                    'transaction_ids': subset_txns['transaction_id'].tolist(),
                    'transaction_amounts': subset_txns['amount'].tolist(),
                    'combo_sum': subset_sum,
                    'difference': subset_sum - target_row['target_amount'],
                    'num_transactions': len(subset_txns)
                })
                used_transactions.update(subset_txns['transaction_id'].tolist())
        
        execution_time = time.time() - start_time
        
        results = {
            'subset_matches': subset_matches,
            'execution_time': execution_time,
            'total_subset_matches': len(subset_matches),
            'used_transactions': list(used_transactions)
        }
        
        self.logger.info(f"Dynamic programming subset sum completed: {len(subset_matches)} matches found in {execution_time:.4f}s")
        return results
    
    def train_matching_model(self, features_df: pd.DataFrame, model_type: str = 'random_forest') -> Dict:
        """
        Train machine learning model to predict matching likelihood.
        
        Args:
            features_df: DataFrame with engineered features
            model_type: Type of model to train ('random_forest', 'logistic', 'linear')
            
        Returns:
            Dictionary with model performance metrics
        """
        # Prepare features and target
        feature_columns = ['amount_diff', 'amount_ratio', 'amount_sum', 'amount_product',
                          'desc_similarity', 'desc_partial', 'desc_token_sort',
                          'amount_std_diff', 'amount_percentile_diff', 'sign_match',
                          'amount_diff_normalized', 'amount_ratio_log', 'similarity_avg']
        
        X = features_df[feature_columns].fillna(0)
        y = features_df['is_exact_match']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model - only use classification models
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced for speed
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=500)  # Reduced for speed
        elif model_type == 'linear':
            # Skip linear regression for classification
            self.logger.warning("Linear regression not suitable for classification, using logistic instead")
            model = LogisticRegression(random_state=42, max_iter=500)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics for classification only
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Cross-validation with error handling
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')  # Reduced CV folds
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            cv_scores = np.array([metrics['accuracy']])  # Use accuracy as fallback
        
        # Store model
        self.models[model_type] = {
            'model': model,
            'scaler': self.scaler,
            'feature_columns': feature_columns
        }
        
        results = {
            'model_type': model_type,
            'metrics': metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': None
        }
        
        # Add feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = dict(zip(feature_columns, model.feature_importances_))
        
        self.logger.info(f"Trained {model_type} model with accuracy: {metrics.get('accuracy', metrics.get('rmse', 'N/A'))}")
        return results
    
    def predict_matches(self, features_df: pd.DataFrame, model_type: str = 'random_forest', 
                       threshold: float = 0.5) -> pd.DataFrame:
        """
        Predict matches using trained model.
        
        Args:
            features_df: DataFrame with engineered features
            model_type: Type of model to use
            threshold: Probability threshold for classification
            
        Returns:
            DataFrame with predictions
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained. Call train_matching_model first.")
        
        model_info = self.models[model_type]
        model = model_info['model']
        scaler = model_info['scaler']
        feature_columns = model_info['feature_columns']
        
        # Prepare features
        X = features_df[feature_columns].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
            if probabilities.shape[1] > 1:
                predictions = probabilities[:, 1]  # Probability of positive class
            else:
                predictions = probabilities[:, 0]  # Only one class
        else:
            predictions = model.predict(X_scaled)
        
        # Add predictions to features DataFrame
        result_df = features_df.copy()
        result_df['predicted_probability'] = predictions
        result_df['predicted_match'] = (predictions > threshold).astype(int)
        
        return result_df
    
    def get_ml_reconciliation_summary(self, transactions_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict:
        """
        Get comprehensive ML reconciliation summary.
        
        Args:
            transactions_df: Processed transactions DataFrame
            targets_df: Processed targets DataFrame
            
        Returns:
            Dictionary with comprehensive ML reconciliation results
        """
        start_time = time.time()
        
        # Feature engineering
        features_df = self.feature_engineering(transactions_df, targets_df)
        
        # Train models
        model_results = {}
        for model_type in ['random_forest', 'logistic', 'linear']:
            try:
                model_results[model_type] = self.train_matching_model(features_df, model_type)
            except Exception as e:
                self.logger.warning(f"Failed to train {model_type} model: {e}")
                continue
        
        # Dynamic programming subset sum
        dp_results = self.dynamic_programming_subset_sum(transactions_df, targets_df)
        
        # Predict matches using best model
        if model_results:
            best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics'].get('accuracy', 0))
            predictions_df = self.predict_matches(features_df, best_model)
            
            # Calculate ML-based reconciliation metrics - FIXED!
            # Only count actual successful matches, not all predictions
            high_confidence_matches = predictions_df[
                (predictions_df['predicted_probability'] > 0.8) & 
                (predictions_df['is_exact_match'] == 1)  # Only count actual matches
            ]
        else:
            best_model = 'none'
            predictions_df = pd.DataFrame()
            high_confidence_matches = pd.DataFrame()
        
        total_time = time.time() - start_time
        
        summary = {
            'total_execution_time': total_time,
            'feature_engineering': {
                'total_feature_vectors': len(features_df),
                'positive_samples': len(features_df[features_df['is_exact_match'] == 1]),
                'negative_samples': len(features_df[features_df['is_exact_match'] == 0])
            },
            'model_performance': model_results,
            'dynamic_programming': dp_results,
            'predictions': {
                'total_predictions': len(predictions_df) if not predictions_df.empty else 0,
                'high_confidence_matches': len(high_confidence_matches) if not high_confidence_matches.empty else 0,
                'best_model': best_model,
                'prediction_threshold': 0.8
            }
        }
        
        self.logger.info(f"ML reconciliation completed in {total_time:.4f}s")
        self.logger.info(f"Best model: {best_model}")
        
        return summary
