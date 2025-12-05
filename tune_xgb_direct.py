# tune_xgb_direct.py

import pandas as pd
from math import sqrt
from train_model import prepare_data_for_training  # we will create this helper
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def try_params(params, X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_jobs=-1,
        tree_method="hist",
        random_state=42,
        **params
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = sqrt(mean_squared_error(y_test, preds))
    return mae, rmse


def main():
    X_train, y_train, X_test, y_test = prepare_data_for_training()

    search_space = [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n}
        for d in [6, 7, 8, 9]
        for lr in [0.03, 0.05, 0.07]
        for n in [300, 400, 500]
    ]

    fixed_params = {
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    }

    best = (999, None)

    for combo in search_space:
        params = {**combo, **fixed_params}
        mae, rmse = try_params(params, X_train, y_train, X_test, y_test)
        print(params, "-> MAE:", mae)

        if mae < best[0]:
            best = (mae, params)

    print("\nBest MAE:", best[0])
    print("Best Params:", best[1])


if __name__ == "__main__":
    main()
