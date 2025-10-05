"""Performance metrics, confusion matrices, Hosmer-Lemeshow test, and cross-validation."""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve, auc
)
from typing import Dict, Any, Tuple


def add_predictions(df: pd.DataFrame, models: Dict[str, Any], 
                   threshold: float = 0.5) -> pd.DataFrame:
    """Add predicted probabilities and classes to DataFrame.
    
    Args:
        df: Input DataFrame
        models: Dictionary of model name -> fitted model
        threshold: Classification threshold
        
    Returns:
        DataFrame with prediction columns added
    """
    df = df.copy()
    
    for model_name, model in models.items():
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        prob_col = f'pred_prob_{safe_name}'
        class_col = f'pred_class_{safe_name}'
        
        df[prob_col] = model.predict()
        df[class_col] = (df[prob_col] > threshold).astype(int)
    
    return df


def calculate_performance_metrics(df: pd.DataFrame, models: Dict[str, Any],
                                  threshold: float = 0.5) -> pd.DataFrame:
    """Calculate classification performance metrics for all models.
    
    Args:
        df: DataFrame with predictions
        models: Dictionary of model name -> fitted model
        threshold: Classification threshold
        
    Returns:
        DataFrame with performance metrics
    """
    performance_metrics = []
    
    for model_name, model in models.items():
        pred_prob = model.predict()
        pred_class = (pred_prob > threshold).astype(int)
        
        accuracy = accuracy_score(df['Incident'], pred_class)
        precision = precision_score(df['Incident'], pred_class, zero_division=0)
        recall = recall_score(df['Incident'], pred_class, zero_division=0)
        f1 = f1_score(df['Incident'], pred_class, zero_division=0)
        roc_auc = roc_auc_score(df['Incident'], pred_prob)
        
        performance_metrics.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
    
    return pd.DataFrame(performance_metrics)


def calculate_confusion_matrices(df: pd.DataFrame, models: Dict[str, Any],
                                 threshold: float = 0.5) -> Dict[str, Dict]:
    """Calculate confusion matrices and related metrics for all models.
    
    Args:
        df: DataFrame with target variable
        models: Dictionary of model name -> fitted model
        threshold: Classification threshold
        
    Returns:
        Dictionary with confusion matrix statistics per model
    """
    results = {}
    
    for model_name, model in models.items():
        pred_prob = model.predict()
        pred_class = (pred_prob > threshold).astype(int)
        
        cm = confusion_matrix(df['Incident'], pred_class)
        tn, fp, fn, tp = cm.ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results[model_name] = {
            'confusion_matrix': cm,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'specificity': specificity,
            'sensitivity': sensitivity
        }
    
    return results


def hosmer_lemeshow_test(y_true: np.ndarray, y_pred: np.ndarray, 
                         g: int = 10) -> Tuple[float, int, float]:
    """Perform Hosmer-Lemeshow goodness-of-fit test.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        g: Number of groups (default 10)
        
    Returns:
        Tuple of (chi-square statistic, degrees of freedom, p-value)
    """
    df_hl = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df_hl['decile'] = pd.qcut(df_hl['y_pred'], g, duplicates='drop')
    
    hl_table = df_hl.groupby('decile').agg(
        observed=('y_true', 'sum'),
        n=('y_true', 'count'),
        expected=('y_pred', 'sum')
    )
    
    observed = hl_table['observed'].values
    expected = hl_table['expected'].values
    n = hl_table['n'].values
    
    eps = 1e-10
    chi_sq = np.sum((observed - expected)**2 / 
                    (expected * (1 - expected/np.maximum(n, eps)) + eps))
    df_test = g - 2
    p_value = 1 - stats.chi2.cdf(chi_sq, df_test)
    
    return chi_sq, df_test, p_value


def perform_cross_validation(df: pd.DataFrame, feature_names: list,
                             config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform stratified k-fold cross-validation.
    
    Args:
        df: DataFrame with features and target
        feature_names: List of feature column names
        config: Configuration dictionary
        
    Returns:
        Dictionary with CV results
    """
    cv_config = config['cross_validation']
    
    X = df[feature_names]
    y = df['Incident']
    
    cv = StratifiedKFold(
        n_splits=cv_config['n_splits'],
        shuffle=cv_config['shuffle'],
        random_state=cv_config['random_state']
    )
    
    clf = LogisticRegression(max_iter=1000, 
                            random_state=cv_config['random_state'])
    cv_scores = cross_val_score(clf, X, y, cv=cv, 
                               scoring=cv_config['scoring'])
    
    return {
        'scores': cv_scores,
        'mean': cv_scores.mean(),
        'std': cv_scores.std()
    }


def get_roc_data(df: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Dict]:
    """Calculate ROC curve data for all models.
    
    Args:
        df: DataFrame with target variable
        models: Dictionary of model name -> fitted model
        
    Returns:
        Dictionary with ROC data per model
    """
    roc_data = {}
    
    for model_name, model in models.items():
        pred_prob = model.predict()
        fpr, tpr, thresholds = roc_curve(df['Incident'], pred_prob)
        roc_auc = auc(fpr, tpr)
        
        roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    return roc_data