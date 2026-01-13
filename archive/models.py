"""
Models module for F1 Lap Time Prediction.

Contains the definitions for training the core physics models (XGBoost) and the
Adaptive Ensemble logic that mixes model predictions with a moving average baseline.
"""

import numpy as np
import xgboost as xgb

def train_physics_model(X_train, y_train):
    """
    Trains the XGBoost Regressor on the provided training features and targets.
    
    The model is configured to predict the 'Delta' (Change in Lap Time),
    rather than absolute lap time, to better generalize across tracks.
    
    Args:
        X_train: Feature matrix.
        y_train: Target vector (DeltaNextLapSec).
        
    Returns:
        xgb.XGBRegressor: Trained model instance.
    """
    model = xgb.XGBRegressor(
        # === HYPERPARAMETERS ===
        # These are calibrated defaults. For optimal results, use the tuned
        # parameters from 'best_params_season_2025_fast.pkl'.
        learning_rate=0.005,
        n_estimators=500,       
        max_depth=6,            
        subsample=0.8,          
        colsample_bytree=0.8,   
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.1,          
        # =======================
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def run_adaptive_ensemble(test_df, preds_model, preds_base, y_true, alpha=0.5, start_w=0.5):
    """
    Applies the Adaptive Ensemble Strategy (Online Learning).
    
    Dynamically blends predictions from the Physics Model and the Baseline (Previous Lap)
    based on their recent performance. This allows the system to fallback to the baseline
    when the model starts failing (e.g., during changing weather or tyre cliffs).
    
    Formula:
        Weight_Model = Error_Base / (Error_Model + Error_Base)
        Prediction = (Weight * Model_Pred) + ((1 - Weight) * Base_Pred)
        
    Args:
        test_df: DataFrame containing metadata (Driver, RaceName).
        preds_model: Array of predictions from the Physics Model.
        preds_base: Array of predictions from the Baseline (Previous Lap).
        y_true: Array of actual ground truth values.
        alpha: Smoothing factor for error tracking (0.0 to 1.0). 
               Higher alpha = Faster adaptation to recent errors.
        start_w: Initial weight confidence in the Physics Model (0.0 to 1.0).
        
    Returns:
        np.array: Final adaptive predictions.
    """
    adaptive_preds = np.zeros(len(test_df))
    drivers = test_df['Driver'].unique()
    current_idx = 0
    
    for driver in drivers:
        n_laps = len(test_df[test_df['Driver'] == driver])
        
        # Slice data for current driver
        d_preds_model = preds_model[current_idx : current_idx + n_laps]
        d_preds_base = preds_base[current_idx : current_idx + n_laps]
        d_y_true = y_true[current_idx : current_idx + n_laps]
        
        d_adaptive_preds = []
        
        # Initialize Weights & Error Trackers
        w_model = start_w
        err_model_smooth = 0.1 
        err_base_smooth = 0.1
        
        for i in range(n_laps):
            # 1. Predict (Weighted Average)
            pred = (w_model * d_preds_model[i]) + ((1 - w_model) * d_preds_base[i])
            d_adaptive_preds.append(pred)
            
            # 2. Observe Actual Error
            true_val = d_y_true[i]
            raw_err_model = abs(true_val - d_preds_model[i])
            raw_err_base = abs(true_val - d_preds_base[i])
            
            # 3. Update Error Estimates (Exponential Moving Average)
            err_model_smooth = (alpha * raw_err_model) + ((1 - alpha) * err_model_smooth)
            err_base_smooth = (alpha * raw_err_base) + ((1 - alpha) * err_base_smooth)
            
            # 4. Update Weights for Next Lap
            total_error = err_model_smooth + err_base_smooth
            if total_error > 0:
                w_model = err_base_smooth / total_error
            else:
                w_model = 0.5
        
        adaptive_preds[current_idx : current_idx + n_laps] = d_adaptive_preds
        current_idx += n_laps
        
    return adaptive_preds
