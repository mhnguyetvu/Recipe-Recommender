"""
Efficient evaluation utilities for recommendation systems with large datasets
"""
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Optional
import time


def evaluate_lightfm_subset(
    model,
    test_matrix: csr_matrix,
    train_matrix: csr_matrix,
    user_indices: np.ndarray,
    k: int = 10,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Efficiently evaluate precision@k and recall@k for a subset of users
    
    Args:
        model: Trained LightFM model
        test_matrix: Test interaction matrix (CSR format)
        train_matrix: Train interaction matrix (CSR format) 
        user_indices: Array of user indices to evaluate
        k: Number of top recommendations to consider
        verbose: Whether to print progress
        
    Returns:
        Tuple of (mean_precision, mean_recall)
    """
    precisions = []
    recalls = []
    
    n_items = test_matrix.shape[1]
    start_time = time.time()
    
    for idx, user_idx in enumerate(user_indices):
        # Progress tracking
        if verbose and idx % 100 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (len(user_indices) - idx) / rate
            print(f"Processed {idx}/{len(user_indices)} users ({idx/len(user_indices)*100:.1f}%) - "
                  f"ETA: {remaining:.1f}s", end='\r')
        
        # Get test items for this user
        test_items = test_matrix[user_idx].indices
        if len(test_items) == 0:
            continue
            
        # Get train items (to exclude from recommendations)
        train_items = set(train_matrix[user_idx].indices)
        
        # Predict scores for all items
        # LightFM predict expects arrays of user_ids and item_ids
        item_ids = np.arange(n_items)
        user_ids = np.full(n_items, user_idx, dtype=np.int32)
        scores = model.predict(user_ids, item_ids)
        
        # Exclude training items
        for item in train_items:
            scores[item] = -np.inf
            
        # Get top-k recommendations
        top_k_items = np.argsort(-scores)[:k]
        
        # Calculate precision and recall
        hits = len(set(top_k_items) & set(test_items))
        precision = hits / k
        recall = hits / len(test_items) if len(test_items) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    if verbose:
        print()  # New line after progress
        
    return np.mean(precisions), np.mean(recalls)


def evaluate_lightfm_fast(
    model,
    test_matrix: csr_matrix,
    train_matrix: csr_matrix,
    n_users: int = 1000,
    k: int = 10,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """
    Fast evaluation on a random subset of users
    
    Args:
        model: Trained LightFM model
        test_matrix: Test interaction matrix (CSR format)
        train_matrix: Train interaction matrix (CSR format)
        n_users: Number of users to sample for evaluation
        k: Number of top recommendations to consider
        random_state: Random seed for reproducibility
        verbose: Whether to print results
        
    Returns:
        Dictionary with precision and recall metrics
    """
    # Convert to CSR if needed
    if not isinstance(test_matrix, csr_matrix):
        test_matrix = test_matrix.tocsr()
    if not isinstance(train_matrix, csr_matrix):
        train_matrix = train_matrix.tocsr()
    
    # Sample users
    rng = np.random.RandomState(random_state)
    n_total_users = test_matrix.shape[0]
    n_eval_users = min(n_users, n_total_users)
    
    # Only sample users who have interactions in test set
    users_with_test_interactions = np.where(np.array(test_matrix.sum(axis=1)).flatten() > 0)[0]
    
    if len(users_with_test_interactions) == 0:
        print("Warning: No users with test interactions found!")
        return {'precision': 0.0, 'recall': 0.0, 'n_users': 0}
    
    n_eval_users = min(n_eval_users, len(users_with_test_interactions))
    eval_users = rng.choice(users_with_test_interactions, size=n_eval_users, replace=False)
    
    if verbose:
        print(f"Evaluating on {n_eval_users} users (out of {n_total_users} total)...")
    
    # Evaluate
    precision, recall = evaluate_lightfm_subset(
        model, test_matrix, train_matrix, eval_users, k=k, verbose=verbose
    )
    
    results = {
        'precision': precision,
        'recall': recall,
        'n_users': n_eval_users,
        'k': k
    }
    
    if verbose:
        print(f"\nResults (k={k}):")
        print(f"  Precision@{k}: {precision:.4f}")
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  Evaluated on: {n_eval_users} users")
    
    return results


def evaluate_ranking_metrics(
    model,
    train_matrix: csr_matrix,
    val_matrix: csr_matrix,
    test_matrix: Optional[csr_matrix] = None,
    n_users: int = 1000,
    k_values: list = [5, 10, 20],
    random_state: int = 42
) -> dict:
    """
    Comprehensive evaluation across train/val/test sets
    
    Args:
        model: Trained LightFM model
        train_matrix: Training interaction matrix
        val_matrix: Validation interaction matrix
        test_matrix: Optional test interaction matrix
        n_users: Number of users to sample
        k_values: List of k values to evaluate
        random_state: Random seed
        
    Returns:
        Dictionary with all evaluation results
    """
    results = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    # Evaluate on train set
    print("=" * 60)
    print("EVALUATING ON TRAIN SET")
    print("=" * 60)
    for k in k_values:
        train_results = evaluate_lightfm_fast(
            model, train_matrix, train_matrix, 
            n_users=n_users, k=k, random_state=random_state
        )
        results['train'][f'k={k}'] = train_results
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("EVALUATING ON VALIDATION SET")
    print("=" * 60)
    for k in k_values:
        val_results = evaluate_lightfm_fast(
            model, val_matrix, train_matrix,
            n_users=n_users, k=k, random_state=random_state
        )
        results['val'][f'k={k}'] = val_results
    
    # Evaluate on test set if provided
    if test_matrix is not None:
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)
        for k in k_values:
            test_results = evaluate_lightfm_fast(
                model, test_matrix, train_matrix,
                n_users=n_users, k=k, random_state=random_state
            )
            results['test'][f'k={k}'] = test_results
    
    return results


def print_evaluation_summary(results: dict):
    """
    Pretty print evaluation results
    
    Args:
        results: Dictionary from evaluate_ranking_metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for dataset in ['train', 'val', 'test']:
        if dataset in results and results[dataset]:
            print(f"\n{dataset.upper()} SET:")
            print("-" * 40)
            for k_label, metrics in results[dataset].items():
                k = metrics['k']
                print(f"  {k_label}:")
                print(f"    Precision@{k}: {metrics['precision']:.4f}")
                print(f"    Recall@{k}:    {metrics['recall']:.4f}")
