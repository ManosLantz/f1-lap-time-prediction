import numpy as np
import xgboost as xgb

def train_physics_model(X_train, y_train):
    """
    Trains the XGBoost model
    """
    model = xgb.XGBRegressor(
        # === NEW CLEAN CONFIGURATION ===
        learning_rate=0.005,
        n_estimators=500,       # Back to 500 (Crucial!)
        max_depth=6,            # Back to 6 (Capture complexity)
        subsample=0.8,          # Back to 0.8
        colsample_bytree=0.8,   # Back to 0.8
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.1,          # Back to 0.1 (Less aggressive)
        # =========================
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def run_adaptive_ensemble(test_df, preds_model, preds_base, y_true, alpha=0.5, start_w=0.5):
    """
    Applies the adaptive weighting logic with Tunable Parameters.
    alpha: Smoothing factor (Higher = React faster to recent errors).
    start_w: Initial weight for the Physics Model (0.5 = Neutral, 0.9 = Trust Model).
    """
    adaptive_preds = np.zeros(len(test_df))
    drivers = test_df['Driver'].unique()
    current_idx = 0
    
    for driver in drivers:
        n_laps = len(test_df[test_df['Driver'] == driver])
        
        # Slice data
        d_preds_model = preds_model[current_idx : current_idx + n_laps]
        d_preds_base = preds_base[current_idx : current_idx + n_laps]
        d_y_true = y_true[current_idx : current_idx + n_laps]
        
        d_adaptive_preds = []
        
        # === TUNABLE PARAMETERS ===
        w_model = start_w  # Allow starting with higher trust in AI
        
        # Initialize error tracking
        # We assume initial error is small to avoid wild swings lap 1
        err_model_smooth = 0.1 
        err_base_smooth = 0.1
        
        for i in range(n_laps):
            # 1. Predict
            pred = (w_model * d_preds_model[i]) + ((1 - w_model) * d_preds_base[i])
            d_adaptive_preds.append(pred)
            
            # 2. Update Weights
            true_val = d_y_true[i]
            raw_err_model = abs(true_val - d_preds_model[i])
            raw_err_base = abs(true_val - d_preds_base[i])
            
            # 3. Smooth Errors (Uses the Tunable Alpha)
            err_model_smooth = (alpha * raw_err_model) + ((1 - alpha) * err_model_smooth)
            err_base_smooth = (alpha * raw_err_base) + ((1 - alpha) * err_base_smooth)
            
            total_error = err_model_smooth + err_base_smooth
            if total_error > 0:
                w_model = err_base_smooth / total_error
            else:
                w_model = 0.5
        
        adaptive_preds[current_idx : current_idx + n_laps] = d_adaptive_preds
        current_idx += n_laps
        
    return adaptive_preds