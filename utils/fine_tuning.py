import os
os.getcwd()
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

import itertools

from utils.regression_utils import *

from arterial.io.load_and_save_operations import save_pickle

def run_fine_tuning(preprocessed_data, random_seed_splitting, random_seed_initialization, params_with_bounds, n_ensemble = 10, important_features = None, flag = None):

    X_train = preprocessed_data["X_train"]
    y_train_reg = preprocessed_data["y_train_reg"]
    y_train_class = preprocessed_data["y_train_class"]
    X_test = preprocessed_data["X_test"]
    y_test_reg = preprocessed_data["y_test_reg"]
    y_test_class = preprocessed_data["y_test_class"]

    if important_features is not None:
        X_train = X_train[important_features]
        X_test = X_test[important_features]

    results_df = pd.DataFrame(columns=["Model name", "Train RMSE", "Test RMSE", "Train ROC", "Test ROC", "Train F1-score", "Test F1-score", "Test F1-score (train)"])

    param_array_dict = {}

    for param, bounds in params_with_bounds.items():
        param_array_dict[param] = np.arange(bounds[0], bounds[1], bounds[2])

    # Generate all combinations of parameters
    all_param_combinations = list(itertools.product(*param_array_dict.values()))
    
    for param_combination in all_param_combinations:
        # Set parameters
        params = {param: param_combination[idx] for idx, param in enumerate(param_array_dict.keys())}

        model_name = get_model_name(params) + f"-seed_{random_seed_initialization}" + "-ensemble"
        # We can set an output directory for the model
        output_dir = "data/splits/{}/results/fine_tuning/{}".format(random_seed_splitting, model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Train the model and get predictions
        summary = fit_reg_ensemble(X_train, y_train_reg, X_test, y_test_reg, params, random_seed_initialization, n_ensemble=n_ensemble)
        # Evaluate train and validaiton performance of the model
        summary = evaluate_model(model_name, summary, y_train_class, y_test_class, plot=False, output_dir=output_dir)
        # Save summary
        save_pickle(summary, f"{output_dir}/summary.pkl")

        results_line = pd.DataFrame({"Model name": model_name,
                        "Train RMSE": summary["rmse_train"],
                        "Test RMSE": summary["rmse_test"],
                        "Train ROC": summary["roc_auc_train"],
                        "Test ROC": summary["roc_auc_test"],
                        "Train F1-score": summary["f1_score_train"],
                        "Test F1-score": summary["f1_score_test_train"],
                        "Test F1-score (train)": summary["f1_score_test"]}, index =[0])
        
        # Concatenate results
        results_df = pd.concat([results_df, results_line], ignore_index=True)
        if flag is None:
            results_df.to_excel("data/splits/{}/results/results_fine_tuning.xlsx".format(random_seed_splitting), index=False)
        else:
            results_df.to_excel("data/splits/{}/results/results_fine_tuning_{}.xlsx".format(random_seed_splitting, flag), index=False)

    return results_df

def choose_best_model(random_seed_range, flag=None):
    # Select the best model following fine_tuning

    if flag is None:
        filename = "results_fine_tuning"
    else:
        filename = "results_fine_tuning_{}".format(flag)
    results_df_list = []
    for random_seed in random_seed_range:
        if os.path.exists("data/splits/{}/results/{}.xlsx".format(random_seed, filename)):
            if len(results_df_list) == 0:
                results_df_list.append(pd.read_excel("data/splits/{}/results/{}.xlsx".format(random_seed, filename)))
            else:
                results_df_next = pd.read_excel("data/splits/{}/results/{}.xlsx".format(random_seed, filename))
                results_df_list.append(results_df_next)

    print("A total of {} random seeds is taken into consideration for fine tuning".format(len(results_df_list)))

    # We will construct a new df with the mean and std of the Train RMSE, Test RMSE, Train ROC, Test ROC for each model, across folds
    averaged_results_df = pd.DataFrame(columns=['Model name', 
                                                'Mean Train RMSE', 'Mean Test RMSE', 
                                                'Mean Train ROC', 'Mean Test ROC',
                                                'Mean Train F1-score', 'Mean Test F1-score (train)',
                                                'Std Train RMSE', 'Std Test RMSE', 
                                                'Std Train ROC', 'Std Test ROC',
                                                'Std Train F1-score', 'Std Test F1-score (train)'])
    for idx in results_df_list[0].index:
        model_name = results_df_list[0].loc[idx, 'Model name']
        train_rmse = []
        test_rmse = []
        train_roc = []
        test_roc = []
        f1_train = []
        f1_test = []
        for seed_idx in range(len(results_df_list)):
            train_rmse.append(results_df_list[seed_idx].loc[idx, 'Train RMSE'])
            test_rmse.append(results_df_list[seed_idx].loc[idx, 'Test RMSE'])
            train_roc.append(results_df_list[seed_idx].loc[idx, 'Train ROC'])
            test_roc.append(results_df_list[seed_idx].loc[idx, 'Test ROC'])
            f1_train.append(results_df_list[seed_idx].loc[idx, 'Train F1-score'])
            f1_test.append(results_df_list[seed_idx].loc[idx, 'Test F1-score (train)'])
        averaged_results_df = pd.concat([averaged_results_df, pd.DataFrame({'Model name': model_name,
                                                        'Mean Train RMSE': np.mean(train_rmse),
                                                        'Mean Test RMSE': np.mean(test_rmse),
                                                        'Mean Train ROC': np.mean(train_roc),
                                                        'Mean Test ROC': np.mean(test_roc),
                                                        'Mean Train F1-score': np.mean(f1_train),
                                                        'Mean Test F1-score (train)': np.mean(f1_test),
                                                        'Std Train RMSE': np.std(train_rmse),
                                                        'Std Test RMSE': np.std(test_rmse),
                                                        'Std Train ROC': np.std(train_roc),
                                                        'Std Test ROC': np.std(test_roc),
                                                        'Std Train F1-score': np.std(f1_train),
                                                        'Std Test F1-score (train)': np.std(f1_test)}, index = [0])], ignore_index=True)
        

    # Sort by Mean Test ROC
    averaged_results_df.sort_values(by='Mean Test ROC', ascending=False, inplace=True)
    best_model = averaged_results_df.iloc[0, 0]
    print("Best 10 models according to Mean Test ROC")
    display(averaged_results_df.head(10))

    print("Best model was: {}".format(best_model))

    descriptors = best_model.split("-")
    params = {descriptor[:-len(descriptor.split("_")[-1]) - 1]: descriptor.split("_")[-1] for descriptor in descriptors if descriptor[:-len(descriptor.split("_")[-1]) - 1] in ["max_depth", "n_estimators", "subsample"]}

    return best_model, params