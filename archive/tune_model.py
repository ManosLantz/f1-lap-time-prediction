import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def tune_xgb(X_train, y_train, n_iter=50, cv_folds=5, scoring='accuracy'):
    """
    Robust tuning for XGBoost using RandomizedSearchCV.
    """
    print(f"--- Starting XGBoost Tuning (n_iter={n_iter}, cv={cv_folds}) ---")
    
    # 1. Define the Parameter Space
    # Using a mix of uniform and log-uniform distributions usually works best
    param_dist = {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.5, 1.0],         # Regularization
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],   # Row sampling
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], # Column sampling
        'scale_pos_weight': [1, (len(y_train) - sum(y_train)) / sum(y_train)] # Handling imbalance automatically
    }

    # 2. Initialize the Model
    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,  # Use all cores
        random_state=42
    )

    # 3. Setup Cross-Validation Strategy
    # StratifiedKFold preserves class percentage in splits (crucial for classification)
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # 4. Run Randomized Search
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_strategy,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    # 5. Report Results
    print(f"\nBest XGBoost Params: {random_search.best_params_}")
    print(f"Best CV {scoring}: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_