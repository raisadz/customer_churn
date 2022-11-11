"""
Testing main functions and logging errors
"""

import logging
import os
import shutil

import numpy as np
import pandas as pd

from churn_library import (add_target, encoder_helper, import_data,
                           perform_eda, perform_train_test_split, train_models)
from constants import CATEGORY_LIST, KEEP_COLS, PATH_DATA

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import(testing_function=import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df_read = testing_function(PATH_DATA)
        df_read = pd.DataFrame(df_read)
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    logging.info("SUCCESS: Testing import_data")

    try:
        assert df_read.shape[0] > 0
        assert df_read.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_add_target(testing_function=add_target):
    """
    test add_target function
    """
    df_in = pd.DataFrame(
        {
            "Gender": ["M", "F"],
            "Education_Level": ["High School", "Graduate"],
            "Attrition_Flag": ["Existing Customer", "Attrited Customer"],
        }
    )
    df_out = testing_function(df_in)
    try:
        assert "Churn" in df_out.columns
    except AssertionError:
        logging.error("ERROR in add_target: target columns wasn't created")
        raise

    logging.info("SUCCESS: Testing add_target")


def test_eda(testing_function=perform_eda):
    """
    test perform eda function
    """
    path_eda = "temp"
    df_testing = pd.DataFrame(
        {
            "Churn": [1, 0],
            "Customer_Age": [20, 50],
            "Marital_Status": ["Married", "Single"],
            "Total_Trans_Ct": [10, 20],
        }
    )
    testing_function(df_testing, path_eda=path_eda)
    file_age = f"{path_eda}/customer_age_distribution.png"
    file_marital = f"{path_eda}/marital_status_distribution.png"
    file_transactions = f"{path_eda}/total_transaction_distribution.png"
    file_heatmap = f"{path_eda}/heatmap.png"
    try:
        assert os.path.isfile(file_age)
        assert os.path.isfile(file_marital)
        assert os.path.isfile(file_transactions)
        assert os.path.isfile(file_heatmap)
    except AssertionError:
        logging.error("ERROR in perform_eda: Some images were not created")
        raise

    finally:
        shutil.rmtree(path_eda)
    logging.info("SUCCESS: Testing perform_eda")


def test_encoder_helper(testing_function=encoder_helper):
    """
    testing function encoder_helper
    """

    df_in = pd.DataFrame(
        {
            "Churn": [1, 0],
            "Gender": ["M", "F"],
            "Education_Level": ["High School", "Graduate"],
            "Marital_Status": ["Married", "Single"],
            "Income_Category": ["Less than $40K", "$60K - $80K"],
            "Card_Category": ["Blue", "Silver"],
        }
    )
    df_out = testing_function(df_in)
    cols_enc = [x + "_Churn" for x in CATEGORY_LIST]

    try:
        assert set(cols_enc).issubset(set(df_out.columns))
    except AssertionError:
        logging.error(
            "ERROR in encoder_helper: function didn't encode all the required columns"
        )
        raise
    try:
        assert df_in.shape[0] == df_out.shape[0]
    except AssertionError:
        logging.error(
            """ERROR in encoder_helper: the number of observations
                in the input and output dataframes are different"""
        )
        raise
    logging.info("SUCCESS: Testing encoder_helper")


def test_train_test_split(testing_function=perform_train_test_split):
    """
    testing perform_train_test_split
    """
    cols_plus_churn = KEEP_COLS.copy()
    cols_plus_churn.append("Churn")
    df_in = pd.DataFrame(np.zeros((10, len(KEEP_COLS) + 1)),
                         columns=cols_plus_churn)
    data_train, data_test, target_train, target_test = testing_function(df_in)
    try:
        assert data_train.shape[0] + data_test.shape[0] == df_in.shape[0]
        assert target_train.shape[0] + target_test.shape[0] == df_in.shape[0]
        assert data_train.shape[0] == target_train.shape[0]
        assert data_test.shape[0] == target_test.shape[0]
        assert data_train.shape[1] == len(KEEP_COLS)
        assert data_test.shape[1] == len(KEEP_COLS)
    except AssertionError:
        logging.error(
            "ERROR in perform_train_test_split: train_test_split incorrect split"
        )
        raise
    logging.info("SUCCESS: testing perform_train_test_split")


def test_train_models(testing_function=train_models):
    """
    test train_models
    """
    data_train = pd.DataFrame(
        np.reshape(np.asarray(np.random.normal(0, 1, 100)), (50, 2)),
        columns=["feature_1", "feature_2"],
    )
    target_train = pd.Series(np.random.choice([0, 1], 50), name="Churn")
    data_test = pd.DataFrame(
        np.reshape(np.asarray(np.random.normal(0, 1, 50)), (25, 2)),
        columns=["feature_1", "feature_2"],
    )
    target_test = pd.Series(np.random.choice([0, 1], 25), name="Churn")

    path_results = "temp/results"
    path_model = "temp/model"

    testing_function(
        data_train,
        data_test,
        target_train,
        target_test,
        path_results=path_results,
        path_model=path_model,
    )
    roc_curve_file = f"{path_results}/roc_curve_result.png"
    rf_model_file = f"{path_model}/rfc_model.pkl"
    lr_model_file = f"{path_model}/logistic_model.pkl"

    try:
        assert os.path.isfile(roc_curve_file)
    except AssertionError:
        logging.error(
            "ERROR in train_models: train_models roc_curve_results was not created"
        )
        raise

    try:
        assert os.path.isfile(rf_model_file)
        assert os.path.isfile(lr_model_file)
    except AssertionError:
        logging.error(
            "ERROR in train_models: train_models trained models were not saved"
        )
        raise
    finally:
        shutil.rmtree("temp")

    logging.info("SUCCESS: testing train_models")


if __name__ == "__main__":
    test_import(import_data)
    test_add_target(add_target)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_train_test_split(perform_train_test_split)
    test_train_models(train_models)
