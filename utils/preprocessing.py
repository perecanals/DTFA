import os
os.getcwd()

import pandas as pd
import copy

from imblearn.over_sampling import RandomOverSampler

from arterial.io.load_and_save_operations import save_pickle

def preprocess_data(train_df, test_df, oversampling = False, feature_selection = False):
    reg_label = "timediff_first_series"
    class_label = "classification"
    identifier_label = "nhc"

    info_columns = [identifier_label, class_label] # The regression label is kept in case oversampling is needed

    def min_max_norm(data):
        return (data - data.min()) / (data.max() - data.min())
    def min_max_norm_test(data, minimum, maximum):
        return (data - minimum) / (maximum - minimum)

    # Set features to ignore
    global_features = ["AA type_1", "AA type_2", "AA type_3", "Bovine arch", 
                        "AA-BT max angle difference", "AA-CCA max angle difference", "BT-CCA max angle difference"
                        "AA-BT max azimuthal difference", "AA-CCA max azimuthal difference", "BT-CCA max azimuthal difference"
                        "AA-BT max polar difference", "AA-CCA max polar difference", "BT-CCA max polar difference"]
    vessel_types = ["Total", "AA", "CCA"]
    segment_features = ["length", "mean diameter", "std diameter", "tortuosity_index", "min polar angle", "tortuosity index 5 cm", "bending length"]

    # Features to ignore
    global_features_to_ignore = ["AA type_1", "AA type_2", "AA type_3", "Bovine arch", 
                                 "AA-BT max angle difference", "AA-BT max azimuthal difference", "AA-BT max polar difference"]
    segment_features_to_ignore = ["min polar angle", "tortuosity index 5 cm", "bending length"]
    vessels_to_ignore = []

    if feature_selection:
        specific_features_to_ignore = ['AA std diameter', 'Total length', 'BT-CCA max angle difference',
        'BT-CCA max polar difference', 'CCA std diameter',
        'AA-CCA max polar difference', 'AA mean diameter',
        'AA tortuosity_index', 'Total mean diameter', 'CCA mean diameter',
        'CCA length', 'BT-CCA max azimuthal difference', 'side']
    else:
        specific_features_to_ignore = []

    ignore = info_columns

    for feature in global_features_to_ignore:
        ignore.append(feature)
    for feature in segment_features_to_ignore:
        for vessel_type in vessel_types:
            if "{} {}".format(vessel_type, feature) not in ignore:
                ignore.append("{} {}".format(vessel_type, feature))
    for vessel_type in vessels_to_ignore:
        for feature in segment_features:
                ignore.append("{} {}".format(vessel_type, feature))
    for feature in specific_features_to_ignore:
        if feature not in ignore:
            ignore.append(feature)

    # Preprocess train data
    preprocessed_train_df = copy.deepcopy(train_df)
    # For train set, pass classification to binary (pass 2 to 1). This will be handy for oversampling
    preprocessed_train_df[class_label] = preprocessed_train_df[class_label].apply(lambda x: 1 if x == 2 else x)

    # Get min max values to normalize test data using train values
    min_max_values = {}
    for feature in preprocessed_train_df.columns:
        if feature in ignore or feature == reg_label:
            continue
        min_max_values[feature] = {"min": preprocessed_train_df[feature].min(), "max": preprocessed_train_df[feature].max()}
        preprocessed_train_df[feature] = min_max_norm(preprocessed_train_df[feature])

    # Finally, we need to impute missing values. We will use the median value from the training set to impute missing values in both the training and val set
    imputation_values = {}
    for feature in preprocessed_train_df.columns:
        if feature in ignore or feature == reg_label:
            continue
        elif preprocessed_train_df[feature].isnull().sum() > 0:
            preprocessed_train_df[feature].fillna(preprocessed_train_df[feature].median(), inplace=True)
            imputation_values[feature] = preprocessed_train_df[feature].median()
        else:
            imputation_values[feature] = preprocessed_train_df[feature].median()
    
    # Preprocess test data
    preprocessed_test_df = copy.deepcopy(test_df)

    # Apply min-max normalization using the same values as the training cohort
    for feature in preprocessed_test_df.columns:
        if feature in ignore or feature == reg_label:
            continue
        preprocessed_test_df[feature] = min_max_norm_test(preprocessed_test_df[feature], 
                                                        min_max_values[feature]["min"], 
                                                        min_max_values[feature]["max"])

    # Impute missing values using the same values as the training cohort
    for feature in preprocessed_test_df.columns:
        if feature in ignore or feature == reg_label:
            continue
        elif preprocessed_test_df[feature].isnull().sum() > 0:
            preprocessed_test_df[feature].fillna(imputation_values[feature], inplace=True)
            
    # Drop columns that are not used in the model
    X_train = copy.deepcopy(preprocessed_train_df).drop(columns=ignore)
    if oversampling:
        y_train_class_ros = preprocessed_train_df[class_label]
        # Apply random oversampling to the training set
        # The regression label is kept for the oversampling
        ros = RandomOverSampler(random_state=0)
        X_train, _ = ros.fit_resample(X_train, y_train_class_ros)
        # Get the regression label from the oversampled data
        y_train_reg = X_train[reg_label]
    else:
        y_train_reg = preprocessed_train_df[reg_label]

    # Drop the regression label from the training set
    X_train = X_train.drop(columns=[reg_label])
    # Get the classification label from the regression data in the oversampled train set
    y_train_class = y_train_reg.apply(lambda x: 1 if x >= 30 else 0)

    # Test is treated as normal
    X_test = copy.deepcopy(preprocessed_test_df).drop(columns=ignore)
    X_test = X_test.drop(columns=[reg_label])
    y_test_reg = preprocessed_test_df[reg_label]
    y_test_class = y_test_reg.apply(lambda x: 1 if x >= 30 else 0)

    pickle_data = {"X_train": X_train, "y_train_reg": y_train_reg, "y_train_class": y_train_class,
                   "X_test": X_test, "y_test_reg": y_test_reg, "y_test_class": y_test_class}

    return pickle_data