"""Feature selection methods for identifying important proteins."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Union, Tuple, Optional, Literal


def anova_select(
    data: Union[np.ndarray, pd.DataFrame],
    labels: np.ndarray,
    k: Union[int, str] = "all",
    return_scores: bool = False
) -> Union[Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray], 
           Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray, np.ndarray]]:
    """Select features using ANOVA F-test.
    
    Uses univariate linear regression tests to identify features
    that are most associated with the target labels.
    
    Args:
        data: Input data matrix (samples x features)
        labels: Target labels for samples
        k: Number of top features to select, or "all" for all features with scores
        return_scores: If True, also return F-scores and p-values
        
    Returns:
        Tuple of (selected_data, mask), optionally with (f_scores, p_values)
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> labels = np.random.randint(0, 3, 100)
        >>> selected, mask = anova_select(data, labels, k=10)
        >>> print(selected.shape)  # (100, 10)
    """
    # Convert DataFrame to numpy if needed
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        feature_names = data.columns
        data_array = data.values
    else:
        data_array = data
        feature_names = None
    
    # Calculate F-scores and p-values
    f_scores, p_values = f_classif(data_array, labels)
    
    # Select features
    if k == "all":
        mask = np.ones(data_array.shape[1], dtype=bool)
        selected_data = data_array
    else:
        selector = SelectKBest(f_classif, k=k)
        selected_data = selector.fit_transform(data_array, labels)
        mask = selector.get_support()
    
    # Convert back to DataFrame if needed
    if is_dataframe:
        if k == "all":
            selected_data = pd.DataFrame(selected_data, index=data.index, columns=feature_names)
        else:
            selected_data = pd.DataFrame(
                selected_data,
                index=data.index,
                columns=feature_names[mask]
            )
    
    if return_scores:
        return selected_data, mask, (f_scores, p_values)
    return selected_data, mask


# should not be used for your feature-selection task. 
# Do NOT use LASSO to select protein features for classification.
# It produces unstable masks.
def lasso_select(
    data: Union[np.ndarray, pd.DataFrame],
    target: np.ndarray,
    alpha: Optional[float] = None,
    cv: int = 5,
    task: Literal["classification", "regression"] = "classification",
    random_state: Optional[int] = 42,
    return_model: bool = False
) -> Union[Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray],
           Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray, Union[Lasso, LassoCV]]]:
    """Select features using LASSO regularization.
    
    LASSO (Least Absolute Shrinkage and Selection Operator) performs
    feature selection by shrinking some coefficients to zero.
    
    Args:
        data: Input data matrix (samples x features)
        target: Target variable (continuous for regression, discrete for classification)
        alpha: Regularization strength. If None, uses cross-validation
        cv: Number of cross-validation folds (used when alpha is None)
        task: Type of task ('classification' or 'regression')
        random_state: Random seed for reproducibility
        return_model: If True, also return the fitted model
        
    Returns:
        Tuple of (selected_data, mask), optionally with fitted model
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> target = np.random.randn(100)
        >>> selected, mask = lasso_select(data, target, task='regression')
    """
    # Convert DataFrame to numpy if needed
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        feature_names = data.columns
        data_array = data.values
    else:
        data_array = data
        feature_names = None
    
    # Fit LASSO model
    if alpha is None:
        model = LassoCV(cv=cv, random_state=random_state, max_iter=5000)
    else:
        model = Lasso(alpha=alpha, random_state=random_state, max_iter=5000)
    
    model.fit(data_array, target)
    
    # Get non-zero coefficients
    mask = model.coef_ != 0
    selected_data = data_array[:, mask]
    
    # Convert back to DataFrame if needed
    if is_dataframe:
        selected_data = pd.DataFrame(
            selected_data,
            index=data.index,
            columns=feature_names[mask]
        )
    
    if return_model:
        return selected_data, mask, model
    return selected_data, mask


def random_forest_importance(
    data: Union[np.ndarray, pd.DataFrame],
    target: np.ndarray,
    task: Literal["classification", "regression"] = "classification",
    n_estimators: int = 100,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    random_state: Optional[int] = 42,
    return_model: bool = False,
    return_importances: bool = True
) -> Union[Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray, np.ndarray],
           Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray, np.ndarray, 
                 Union[RandomForestClassifier, RandomForestRegressor]]]:
    """Select features using Random Forest feature importance.
    
    Random Forest calculates feature importance based on how much each
    feature decreases the impurity (Gini or entropy) in the trees.
    
    Args:
        data: Input data matrix (samples x features)
        target: Target variable
        task: Type of task ('classification' or 'regression')
        n_estimators: Number of trees in the forest
        top_k: Select top k most important features (overrides threshold)
        threshold: Minimum importance threshold for feature selection
        random_state: Random seed for reproducibility
        return_model: If True, also return the fitted model
        return_importances: If True, return feature importances
        
    Returns:
        Tuple of (selected_data, mask, importances), optionally with model
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> target = np.random.randint(0, 3, 100)
        >>> selected, mask, importances = random_forest_importance(data, target, top_k=10)
        >>> print(selected.shape)  # (100, 10)
    """
    # Convert DataFrame to numpy if needed
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        feature_names = data.columns
        data_array = data.values
    else:
        data_array = data
        feature_names = None
    
    # Fit Random Forest model
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    
    model.fit(data_array, target)
    importances = model.feature_importances_
    
    # Select features
    if top_k is not None:
        # Select top k features
        indices = np.argsort(importances)[::-1][:top_k]
        mask = np.zeros(data_array.shape[1], dtype=bool)
        mask[indices] = True
    elif threshold is not None:
        # Select features above threshold
        mask = importances >= threshold
    else:
        # Select all features with importance > 0
        mask = importances > 0
    
    selected_data = data_array[:, mask]
    
    # Convert back to DataFrame if needed
    if is_dataframe:
        selected_data = pd.DataFrame(
            selected_data,
            index=data.index,
            columns=feature_names[mask]
        )
    
    result = [selected_data, mask]
    if return_importances:
        result.append(importances)
    if return_model:
        result.append(model)
    
    return tuple(result)
