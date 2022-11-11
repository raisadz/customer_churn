"""
Main functions for customer churn task

Date: November 2022
Author: Raisa
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import (CATEGORY_LIST, KEEP_COLS, PATH_DATA, PATH_EDA,
                       PATH_MODEL, PATH_RESULTS)

sns.set()


os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df_read: pandas dataframe
    """
    df_read = pd.read_csv(pth)
    return df_read


def add_target(df_in):
    """
    Add target 'Churn' to the dataframe

    Inputs:
    -------
            df_in: pandas dataframe

    Outputs:
    --------
            df_out: pandas dataframe
    """
    df_out = df_in.copy()
    df_out["Churn"] = df_in["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return df_out


def perform_eda(df_in, path_eda=PATH_EDA):
    """
    perform eda on df and save figures to images folder
    input:
            df_in: pandas dataframe

    output:
            None
    """
    os.makedirs(path_eda, exist_ok=True)
    # plot and save churn distribution
    fig = plt.figure(figsize=(20, 10))
    df_in["Churn"].hist()
    plt.title("Churn distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    fig.savefig(f"{path_eda}/churn_distribution.png")

    # plot and save customer age distribution
    fig = plt.figure(figsize=(20, 10))
    df_in["Customer_Age"].hist()
    plt.title("Customer Age Distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    fig.savefig(f"{path_eda}/customer_age_distribution.png")

    # plot and save marital status
    fig = plt.figure(figsize=(20, 10))
    df_in.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.title("Marital Status Distribution")
    plt.xlabel("Value")
    plt.ylabel("Percent")
    fig.savefig(f"{path_eda}/marital_status_distribution.png")

    # plot and save transation distribution
    fig = plt.figure(figsize=(20, 10))
    sns.histplot(df_in["Total_Trans_Ct"], stat="density", kde=True)
    plt.title("Distribution of the number of transactions")
    fig.savefig(f"{path_eda}/total_transaction_distribution.png")

    # plot and save heatmap of correlations
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(df_in.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.title("Correlation matrix")
    plt.show()
    fig.savefig(f"{path_eda}/heatmap.png")


def encoder_helper(df_in):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df_int: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df_out: pandas dataframe with new columns for
    """
    df_out = df_in.copy()
    for cat in CATEGORY_LIST:
        cat_lst = []
        cat_groups = df_in.groupby(cat).mean()["Churn"]

        for val in df_in[cat]:
            cat_lst.append(cat_groups.loc[val])

        df_out[f"{cat}_Churn"] = cat_lst

    return df_out


def perform_train_test_split(df_in):
    """
    input:
              df_in: pandas dataframe
    output:
              features_train: X training data
              features_test: X testing data
              target_train: y training data
              target_test: y testing data
    """

    target = df_in["Churn"]
    features_matrix = pd.DataFrame()

    features_matrix[KEEP_COLS] = df_in[KEEP_COLS]
    features_train, features_test, target_train, target_test = train_test_split(
        features_matrix, target, test_size=0.3, random_state=42)
    return features_train, features_test, target_train, target_test


def train_models(
    features_train,
    features_test,
    target_train,
    target_test,
    path_results=PATH_RESULTS,
    path_model=PATH_MODEL,
):
    """
    train, store model results: images + scores, and store models
    input:
              features_train: training data
              features_test: testing data
              target_train: training target
              target_test: testing target
    output:
              None
    """

    os.makedirs(path_results, exist_ok=True)
    os.makedirs(path_model, exist_ok=True)

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(features_train, target_train)
    lrc.fit(features_train, target_train)

    # plot roc auc curve
    lrc_plot = plot_roc_curve(lrc, features_test, target_test)
    fig = plt.figure(figsize=(15, 8))
    axes = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_, features_test, target_test, ax=axes, alpha=0.8
    )
    lrc_plot.plot(ax=axes, alpha=0.8)
    plt.title("ROC Curve")
    plt.show()
    fig.savefig(f"{path_results}/roc_curve_result.png")

    # save best model
    joblib.dump(cv_rfc.best_estimator_, f"{path_model}/rfc_model.pkl")
    joblib.dump(lrc, f"{path_model}/logistic_model.pkl")


def calculate_predictions(features_matrix, model):
    """
    Calculate predictions of the model located in the model_path using features in features_matrix

    Inputs:
    --------
    features_matrix: (pandas dataframe) a matrix of features with rows being observations
        and columns being features which are used in model calculations
    model: (pickle object) model to calculate predictions

    Outputs:
    ---------
    target_predict: (numpy array) a vector of the model's predictions
        where the length is the number of observations
    """

    target_predict = model.predict(features_matrix)
    return target_predict


def classification_report_image(
    target_train,
    target_test,
    target_train_preds_lr,
    target_train_preds_rf,
    target_test_preds_lr,
    target_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            target_train: training response values
            target_test:  test response values
            target_train_preds_lr: training predictions from logistic regression
            target_train_preds_rf: training predictions from random forest
            target_test_preds_lr: test predictions from logistic regression
            target_test_preds_rf: test predictions from random forest

    output:
             None
    """
    fig = plt.figure()
    plt.rc("figure", figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(
        0.01,
        1.25,
        str("Random Forest Train"),
        {"fontsize": 6},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(target_test, target_test_preds_rf)),
        {"fontsize": 6},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Random Forest Test"),
        {"fontsize": 6},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(target_train, target_train_preds_rf)),
        {"fontsize": 6},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    fig.savefig(f"{PATH_RESULTS}/rf_results.png")

    fig = plt.figure()
    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str("Logistic Regression Train"),
        {"fontsize": 6},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(target_train, target_train_preds_lr)),
        {"fontsize": 6},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Logistic Regression Test"),
        {"fontsize": 6},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(target_test, target_test_preds_lr)),
        {"fontsize": 6},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    fig.savefig(f"{PATH_RESULTS}/logistic_results.png")


def feature_importance_plot(model, features_data, output_pth=PATH_RESULTS):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            features_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(features_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(features_data.shape[1]), names, rotation=90)
    fig.savefig(
        f"{output_pth}/feature_importances.png",
        dpi=100,
        bbox_inches="tight")


if __name__ == "__main__":
    df_import = import_data(PATH_DATA)
    df_main = add_target(df_import)
    perform_eda(df_main)
    df_main = encoder_helper(df_main)
    feat_train, feat_test, tar_train, tar_test = perform_train_test_split(
        df_main)
    train_models(feat_train, feat_test, tar_train, tar_test)
    model_rf = joblib.load(f"{PATH_MODEL}/rfc_model.pkl")
    model_lr = joblib.load(f"{PATH_MODEL}/logistic_model.pkl")
    tar_train_preds_lr = calculate_predictions(feat_train, model_lr)
    tar_train_preds_rf = calculate_predictions(feat_train, model_rf)
    tar_test_preds_lr = calculate_predictions(feat_test, model_lr)
    tar_test_preds_rf = calculate_predictions(feat_test, model_rf)
    classification_report_image(
        tar_train,
        tar_test,
        tar_train_preds_lr,
        tar_train_preds_rf,
        tar_test_preds_lr,
        tar_test_preds_rf,
    )
    feature_importance_plot(model_rf, feat_test)
