import pandas as pd
import numpy as np
import xgboost as xgb

import shap

def get_shap_values(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)

    return shap_values, explainer

def get_shap_values_ensemble(preprocessed_data, params, random_seed_initialization):
    shap_values_ensemble = []

    X_train = preprocessed_data["X_train"]
    y_train_reg = preprocessed_data["y_train_reg"]
    X_test = preprocessed_data["X_test"]

    for idx_seed in range(10):
        params['seed'] = 1000 * random_seed_initialization + idx_seed
        # Fit model
        model = xgb.XGBRegressor(**params)
        fitted_model = model.fit(X_train, y_train_reg)

        shap_values, _ = get_shap_values(fitted_model, X_test)
        shap_values_ensemble.append(shap_values)

    # Compute averaged shap values across ensemble models
    shap_values_ensemble_avg = np.mean(np.array(shap_values_ensemble), axis=0)

    # Compute mean and std of absolute SHAP values across samples
    shap_values_ensemble_abs = np.abs(shap_values_ensemble_avg)
    shap_values_ensemble_abs_mean = np.mean(shap_values_ensemble_abs, axis=0)
    shap_values_ensemble_abs_std = np.std(shap_values_ensemble_abs, axis=0)
    # Concatenate np arrays
    shap_values_ensemble_abs_mean_std = np.concatenate((np.array(X_test.columns).reshape(-1,1), shap_values_ensemble_abs_mean.reshape(-1,1), shap_values_ensemble_abs_std.reshape(-1,1)), axis=1)
    # Set in a df
    shap_values_ensemble_df = pd.DataFrame(shap_values_ensemble_abs_mean_std, columns=["Feature", "Mean abs SHAP", "Std abs SHAP"])

    return shap_values_ensemble_df

def get_feature_importance(models, features):
    feature_importances = np.ndarray([len(features), len(models)])

    for idx, model in enumerate(models):
        feature_importances[:, idx] = model.feature_importances_

    mean_feature_importances = np.mean(feature_importances, axis = 1)
    std_feature_importances = np.std(feature_importances, axis = 1)

    feature_importances_df = pd.DataFrame(np.concatenate((np.array(features).reshape(-1,1), mean_feature_importances.reshape(-1,1), std_feature_importances.reshape(-1,1)), axis=1), columns=["Feature", "Mean feature importance", "Std feature importance"])

    return feature_importances_df