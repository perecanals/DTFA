import os
import math

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.metrics import mean_squared_error, roc_curve, auc, precision_recall_curve, f1_score

def get_model_name(params):
    model_name = "xgb_reg-"
    for key, value in params.items():
        # If value is float, round to 1 decimal
        if isinstance(value, float):
            model_name += key + "_" + f"{value:.1f}" + "-"
        else:
            model_name += key + "_" + str(value) + "-"
    return model_name[:-1]

def fit_reg_model(X_train, y_train, X_test, y_test, params):
    # Initialize results_summary dictionary
    results_summary = {}
    results_summary['model_name'] = get_model_name(params)
    results_summary['params'] = params
    results_summary['y_train_reg'] = y_train
    results_summary['y_test_reg'] = y_test

    # Fit model
    model = xgb.XGBRFRegressor(**params)
    # model = xgb.XGBRegressor(**params)
    fitted_model = model.fit(X_train, y_train)
    results_summary['model'] = fitted_model

    # Store prediction for test and train results_summary
    y_train_pred = fitted_model.predict(X_train)
    y_test_pred = fitted_model.predict(X_test)
    results_summary['y_train_reg_pred'] = y_train_pred
    results_summary['y_test_reg_pred'] = y_test_pred
    results_summary['y_train_reg_pred_std'] = None
    results_summary['y_test_reg_pred_std'] = None

    rmse_train = math.sqrt(mean_squared_error(y_train_pred, y_train))
    rmse_test = math.sqrt(mean_squared_error(y_test_pred, y_test))
    results_summary['rmse_train'] = rmse_train
    results_summary['rmse_test'] = rmse_test

    return results_summary

def fit_reg_ensemble(X_train, y_train, X_test, y_test, params, random_seed, n_ensemble=10):
    # Initialize results_summary dictionary
    results_summary_ensemble = {}
    results_summary_ensemble['model_name'] = get_model_name(params) + f"-seed_{random_seed}" + "-ensemble"
    results_summary_ensemble['params'] = params
    results_summary_ensemble['models_ensemble'] = {}
    results_summary_ensemble['model'] = []
    results_summary_ensemble['y_train_reg'] = y_train
    results_summary_ensemble['y_test_reg'] = y_test

    # Fit models for n_ensemble random seeds
    for idx_seed in range(n_ensemble):
        params['seed'] = 1000 * random_seed + idx_seed
        results_summary_ensemble['models_ensemble'][idx_seed] = fit_reg_model(X_train, y_train, X_test, y_test, params)
        results_summary_ensemble['model'].append(results_summary_ensemble['models_ensemble'][idx_seed]['model'])

    # Compute mean and std of predicted values
    y_train_reg_pred = np.ndarray(shape=(n_ensemble, len(y_train)))
    y_test_reg_pred = np.ndarray(shape=(n_ensemble, len(y_test)))
    for idx_seed in range(n_ensemble):
        y_train_reg_pred[idx_seed] = results_summary_ensemble['models_ensemble'][idx_seed]['y_train_reg_pred']
        y_test_reg_pred[idx_seed] = results_summary_ensemble['models_ensemble'][idx_seed]['y_test_reg_pred']
    results_summary_ensemble['y_train_reg_pred'] = np.mean(y_train_reg_pred, axis=0)
    results_summary_ensemble['y_train_reg_pred_std'] = np.std(y_train_reg_pred, axis=0)
    results_summary_ensemble['y_test_reg_pred'] = np.mean(y_test_reg_pred, axis=0)
    results_summary_ensemble['y_test_reg_pred_std'] = np.std(y_test_reg_pred, axis=0)

    # Compute RMSE resulting from the ensemble
    rmse_train = math.sqrt(mean_squared_error(results_summary_ensemble['y_train_reg_pred'], y_train))
    rmse_test = math.sqrt(mean_squared_error(results_summary_ensemble['y_test_reg_pred'], y_test))
    results_summary_ensemble['rmse_train'] = rmse_train
    results_summary_ensemble['rmse_test'] = rmse_test

    return results_summary_ensemble

def evaluate_model(model_name, results_summary, y_train_class, y_test_class, plot=True, output_dir=None):
    # Read regression prediction from summary
    y_train_reg_pred = results_summary['y_train_reg_pred']
    y_test_reg_pred = results_summary['y_test_reg_pred']

    # Compute tpr, fpr, roc_auc from train predictions
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_class, y_train_reg_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    results_summary['fpr_train'] = fpr_train
    results_summary['tpr_train'] = tpr_train
    results_summary['roc_auc_train'] = roc_auc_train

    # Compute non-weighted f1 score to determine optimal threshold
    precision_train, recall_train, thresholds_train = precision_recall_curve(y_train_class, y_train_reg_pred)
    f1_scores_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)
    # Substitute nans for 0
    f1_scores_train = np.nan_to_num(f1_scores_train)
    optimal_threshold_train = thresholds_train[np.argmax(f1_scores_train)]
    results_summary['optimal_threshold_train'] = optimal_threshold_train
    # Compute class predictions and weighted f1 score with optimal threshold
    y_train_class_pred = np.where(y_train_reg_pred >= optimal_threshold_train, 1, 0)
    results_summary['y_train_class_pred'] = y_train_class_pred
    f1_score_train =  f1_score(y_train_class, y_train_class_pred, average="weighted")
    results_summary['f1_score_train'] = f1_score_train

    # Compute tpr, fpr, roc_auc from test predictions
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test_class, y_test_reg_pred)
    roc_auc_test = auc(fpr_test, tpr_test)
    results_summary['fpr_test'] = fpr_test
    results_summary['tpr_test'] = tpr_test
    results_summary['roc_auc_test'] = roc_auc_test

    # Compute non-weighted f1 score to determine optimal threshold
    precision_test, recall_test, thresholds_test = precision_recall_curve(y_test_class, y_test_reg_pred)
    f1_scores_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
    # Substitute nans for 0
    f1_scores_test = np.nan_to_num(f1_scores_test)
    optimal_threshold_test = thresholds_test[np.argmax(f1_scores_test)]
    results_summary['optimal_threshold_test'] = optimal_threshold_test
    # Compute class predictions and weighted f1 score with optimal threshold (test)
    y_test_class_pred = np.where(y_test_reg_pred >= optimal_threshold_test, 1, 0)
    results_summary['y_test_class_pred'] = y_test_class_pred
    f1_score_test =  f1_score(y_test_class, y_test_class_pred, average="weighted")
    results_summary['f1_score_test'] = f1_score_test

    # Compute class predictions and weighted f1 score with optimal threshold (train)
    y_test_train_class_pred = np.where(y_test_reg_pred >= optimal_threshold_train, 1, 0)
    results_summary['y_test_train_class_pred'] = y_test_train_class_pred
    f1_score_test_train =  f1_score(y_test_class, y_test_train_class_pred, average="weighted")
    results_summary['f1_score_test_train'] = f1_score_test_train

    print("results_summary for {}".format(model_name))
    print("Train: f1_score = {}, roc_auc = {}".format(round(f1_score_train, 2), round(roc_auc_train, 2)))
    print("Test: f1_score = {}, roc_auc = {}".format(round(f1_score_test_train, 2), round(roc_auc_test, 2)))

    if plot:
        make_regression_plots(results_summary['model_name'], results_summary['y_train_reg'], y_train_reg_pred, results_summary['y_test_reg'], y_test_reg_pred, optimal_threshold_train, optimal_threshold_test, output_dir)
        plot_roc(results_summary['model_name'], fpr_train, tpr_train, roc_auc_train, fpr_test, tpr_test, roc_auc_test, output_dir)

    return results_summary

def make_regression_plots(model_name, y_train_reg, y_train_reg_pred, y_test_reg, y_test_reg_pred, optimal_threshold, optimal_threshold_test, output_dir = None):
    # Compute R2 score
    # r2_train = r2_score(y_train_reg, y_train_reg_pred)
    # r2_test = r2_score(y_test_reg, y_test_reg_pred)
    # Make a subplot to plot the data side by side
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(y_train_reg, y_train_reg_pred, alpha=0.5)
    # Make horizontal line at optimal threshold
    ax[0].axhline(y=optimal_threshold, color='r', linestyle='--', label='Optimal threshold = {:.2f}'.format(optimal_threshold))
    ax[0].axvline(x=30, color='g', linestyle='--', label='30 min')
    # ax[0].plot([min(y_train_reg), max(y_train_reg)], [min(y_train_reg), max(y_train_reg)], color='navy', lw=2, linestyle='--', label='R2 = {:.2f}'.format(r2_train))
    ax[0].set_xlabel("Observed")
    ax[0].set_ylabel("Predicted")
    ax[0].set_title("Train")
    ax[0].legend(loc="upper left")
    
    ax[1].scatter(y_test_reg, y_test_reg_pred, alpha=0.5)
    ax[1].axhline(y=optimal_threshold_test, color='r', linestyle='--', label='Optimal threshold = {:.2f}'.format(optimal_threshold_test))
    # ax[1].plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], color='navy', lw=2, linestyle='--', label='R2 = {:.2f}'.format(r2_test))
    ax[1].axvline(x=30, color='g', linestyle='--', label='30 min')
    ax[1].set_xlabel("Observed")
    ax[1].set_ylabel("Predicted")
    ax[1].set_title("Test")
    ax[1].legend(loc="upper left")

    # Set title for the whole plot
    plt.suptitle("Regression results for {}".format(model_name))

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "regression.png"))
        plt.close()

def plot_roc(model_name, fpr_train, tpr_train, roc_auc_train, fpr_test, tpr_test, roc_auc_test, output_dir = None):
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(fpr_train, tpr_train, color='red', lw=2, label='ROC curve (area = {})'.format(round(roc_auc_train, 2)))
    ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curve (train)")
    ax[0].legend(loc="lower right")

    ax[1].plot(fpr_test, tpr_test, color='red', lw=2, label='ROC curve (area = {})'.format(round(roc_auc_test, 2)))
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_xlabel("False positive rate")
    ax[1].set_ylabel("True positive rate")
    ax[1].set_title("ROC curve (test)")
    ax[1].legend(loc="lower right")

    # Set title for the whole plot
    plt.suptitle("ROC for {}".format(model_name))

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "roc.png"))
        plt.close()