from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory



def load_data(url: str="https://www.dropbox.com/s/l9ihi4w0u7mu9h2/train.csv?dl=1") -> pd.DataFrame: 
    """load data used for training model from a publicly available url, hosted on Dropbox

    Parameters
    ----------
    url : str, optional
        URL of the file, by default "https://www.dropbox.com/s/l9ihi4w0u7mu9h2/train.csv?dl=1"

    Returns
    -------
    pd.DataFrame
    """

    df = TabularDatasetFactory.from_delimited_files(url).to_pandas_dataframe()
    return df

def preprocess(df: pd.DataFrame):
    """Placeholder for all preprocessing steps needed on the raw dataframe.
    Currently, it has a minimal step to extract features and the target variable
    from the provided dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe to be preprocessed

    Returns
    -------
    X, y : pd.DataFrame of features, pd.Series of target variable
    """
    X = df.drop("RUL", axis=1)
    y = df["RUL"]
    return X, y

df = load_data()
X, y = preprocess(df)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1234
)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_estimators", 
        type=int, 
        default=100, 
        help="The number of trees in the forest."
    )

    parser.add_argument(
        "--max_depth", 
        type=int, 
        help="The maximum depth of the tree"
    )

    parser.add_argument('--model_dir', type=str, help="Location to save the model")
    parser.add_argument('--model_name', type=str, help="name of the persisted model")
    
    args = parser.parse_args()

    run.log("Number of estimators:", np.float(args.n_estimators))
    run.log("Maximum depth of tree:", np.int(args.max_depth))

    model = RandomForestRegressor(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth
    ).fit(x_train, y_train)

    y_hat = model.predict(x_test)
    rmse = sqrt(mean_squared_error(y_test, y_hat))
    print("rmse = {}".format(rmse))
    run.log("rmse", np.float(rmse))

    if args.model_name and args.model_dir:
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(
            args.model_dir, 
            args.model_name + ".pkl"
        )
        joblib.dump(model, model_path)
        print("Saving model to as {}".format(model_path))


if __name__ == '__main__':
    main()