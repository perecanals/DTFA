import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def split_data(raw_df, ratio, random_seed, plot_distributions = False, verbose = False):
    np.random.seed(random_seed)

    class_label = "classification"

    if verbose:
        print("Total number of samples: ", len(raw_df))
        print("Total number of samples with normal access: {} ({:.2f}%)".format(len(raw_df[raw_df[class_label] == 0]), 100 * len(raw_df[raw_df[class_label] == 0]) / len(raw_df)))
        print("Total number of samples with difficult access: {} ({:.2f}%)".format(len(raw_df[raw_df[class_label] == 1]), 100 * len(raw_df[raw_df[class_label] == 1]) / len(raw_df)))
        print("Total number of samples with impossible access: {} ({:.2f}%)".format(len(raw_df[raw_df[class_label] == 2]), 100 * len(raw_df[raw_df[class_label] == 2]) / len(raw_df)))

        print("\nNumber of features: ", len(raw_df.columns))

    # Split into train and test (stratified)
    train_df, test_df = train_test_split(raw_df, test_size=ratio, random_state=random_seed, stratify=raw_df[class_label])

    if verbose:
        print("Train:")
        print("Total number of samples: ", len(train_df))
        print("Total number of samples with normal access: {} ({:.2f}%)".format(len(train_df[train_df[class_label] == 0]), 100 * len(train_df[train_df[class_label] == 0]) / len(train_df)))
        print("Total number of samples with difficult access: {} ({:.2f}%)".format(len(train_df[train_df[class_label] == 1]), 100 * len(train_df[train_df[class_label] == 1]) / len(train_df)))
        print("Total number of samples with impossible access: {} ({:.2f}%)".format(len(train_df[train_df[class_label] == 2]), 100 * len(train_df[train_df[class_label] == 2]) / len(train_df)))

        print("\nTest:")
        print("Total number of samples: ", len(test_df))
        print("Total number of samples with normal access: {} ({:.2f}%)".format(len(test_df[test_df[class_label] == 0]), 100 * len(test_df[test_df[class_label] == 0]) / len(test_df)))
        print("Total number of samples with difficult access: {} ({:.2f}%)".format(len(test_df[test_df[class_label] == 1]), 100 * len(test_df[test_df[class_label] == 1]) / len(test_df)))
        print("Total number of samples with impossible access: {} ({:.2f}%)".format(len(test_df[test_df[class_label] == 2]), 100 * len(test_df[test_df[class_label] == 2]) / len(test_df)))

    if plot_distributions:
        times = train_df["timediff_first_series"].values
        times_test = test_df["timediff_first_series"].values

        # Plot a histogram
        plt.rcParams.update({'font.size': 14})
        _, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].hist(times, bins=100)
        ax[0].set_xlabel("T1A (min)")
        ax[0].set_ylabel("Number of patients")
        ax[0].set_title("Distribution of T1A (train)")
        ax[0].set_xlim(-10, 150)
        ax[0].set_ylim(0, 80)

        ax[1].hist(times_test, bins=100)
        ax[1].set_xlabel("T1A (min)")
        ax[1].set_ylabel("Number of patients")
        ax[1].set_title("Distribution of T1A (test)")
        ax[1].set_xlim(-10, 150)
        ax[1].set_ylim(0, 80)

    return train_df, test_df