import os
import pandas as pd
import numpy as np
import h5py
import json
from typing import List
from capstone.config import DATA_DIR, PROCESSED_DATA_DIR
from pathlib import Path
from loguru import logger

DEFAULT_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR.joinpath("resolution=5min")

train_file_path = DEFAULT_PROCESSED_DATA_DIR.joinpath("train.csv")
test_file_path = DEFAULT_PROCESSED_DATA_DIR.joinpath("test.csv")
var_names_file_path = DEFAULT_PROCESSED_DATA_DIR.joinpath("variable_names.json")


def preprocess(
    df: pd.DataFrame, select_columns: List[str], resample_to: str = "15min"
) -> pd.DataFrame:
    """[summary]

    Parameters
    ----------
    df : pd.DataFrame
        [Input pandas dataframe to be processed]
    resample_to : str, optional
        [resolution to resample the dataframe to], by default "15min"

    Returns
    -------
    pd.DataFrame
        [Preprocessed pandas dataframe]
    """
    logger.info("preprocessing df ...")
    df.index = pd.date_range(
        start="2000-01-01 00:00:00", periods=df.shape[0], freq="1s"
    )
    return (
        df.groupby(["unit", "cycle"])
        .resample(resample_to)
        .agg("mean")
        .reset_index(drop=True)[select_columns]
    )


def get_data_from_file(file_name=str) -> pd.DataFrame:
    """Get raw data from file

    Parameters
    ----------
    file_name : [str], optional
        Name of the file to be read, by default str

    Returns
    -------
    pd.DataFrame
        Raw pandas dataframe
    """

    file_path = DATA_DIR.joinpath(file_name)
    logger.info("reading from {}".format(file_path))
    with h5py.File(file_path, "r") as hdf:
        # Development set
        W_dev = np.array(hdf.get("W_dev"))  # W
        X_s_dev = np.array(hdf.get("X_s_dev"))  # X_s
        X_v_dev = np.array(hdf.get("X_v_dev"))  # X_v
        T_dev = np.array(hdf.get("T_dev"))  # T
        Y_dev = np.array(hdf.get("Y_dev"))  # RUL
        A_dev = np.array(hdf.get("A_dev"))  # Auxiliary

        # Test set
        W_test = np.array(hdf.get("W_test"))  # W
        X_s_test = np.array(hdf.get("X_s_test"))  # X_s
        X_v_test = np.array(hdf.get("X_v_test"))  # X_v
        T_test = np.array(hdf.get("T_test"))  # T
        Y_test = np.array(hdf.get("Y_test"))  # RUL
        A_test = np.array(hdf.get("A_test"))  # Auxiliary

        # Varnams
        W_var = np.array(hdf.get("W_var"))
        X_s_var = np.array(hdf.get("X_s_var"))
        X_v_var = np.array(hdf.get("X_v_var"))
        T_var = np.array(hdf.get("T_var"))
        A_var = np.array(hdf.get("A_var"))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype="U20"))
        X_s_var = list(np.array(X_s_var, dtype="U20"))
        X_v_var = list(np.array(X_v_var, dtype="U20"))
        T_var = list(np.array(T_var, dtype="U20"))
        A_var = list(np.array(A_var, dtype="U20"))

    var_names = {
        "operating conditions": W_var,
        "monitoring sensors": X_s_var,
        "virtual sensors": X_v_var,
        "health parameters": T_var,
        "RUL": ["RUL"],
        "aux": A_var,
    }

    data_train = np.concatenate((W_dev, X_s_dev, X_v_dev, T_dev, Y_dev, A_dev), axis=1)
    data_test = np.concatenate(
        (W_test, X_s_test, X_v_test, T_test, Y_test, A_test), axis=1
    )

    columns = W_var + X_s_var + X_v_var + T_var + ["RUL"] + A_var
    select_columns = W_var + X_s_var + X_v_var + ["RUL"]

    df_train_raw = pd.DataFrame(data=data_train, columns=columns)
    df_test_raw = pd.DataFrame(data=data_test, columns=columns)
    return df_train_raw, df_test_raw, var_names, select_columns


def output_payload(
    df: pd.DataFrame, n_samples: int = 2, file_path: str = "payload.json"
) -> bool:
    """Save example data as json

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe from which to extract example data
    n_samples : int, optional
        Number of samples to be extracted, by default 2
    file_path : str, optional
        Full path to the output file, by default "payload.json"

    Returns
    -------
    bool
        True if sucessful.
    """
    data = {"data": list(df.loc[0 : n_samples - 1, :].T.to_dict().values())}

    logger.info("saving {}".format(file_path))

    with open(file_path, "w") as f:
        json.dump(data, f)

    return True


def get_file_path(target_resolution: str, file_name: str) -> Path:
    """Construct full path to a file that contains processed data

    Parameters
    ----------
    target_resolution : str
        Resolution of the processed data, used in folder name
    file_name : str
        Name of the file

    Returns
    -------
    Path
        Full path to the specified file
    """
    out_dir = PROCESSED_DATA_DIR.joinpath("resolution={}".format(target_resolution))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir.joinpath(file_name)


if __name__ == "__main__":
    target_resolution = "5min"
    df_train_raw, df_test_raw, var_names, select_columns = get_data_from_file(
        file_name="N-CMAPSS_DS02-006.h5"
    )
    df_train_resampled = preprocess(
        df=df_train_raw, select_columns=select_columns, resample_to=target_resolution
    )
    df_test_resampled = preprocess(
        df=df_test_raw, select_columns=select_columns, resample_to=target_resolution
    )

    output_payload(
        df=df_test_resampled.drop("RUL", axis=1),
        n_samples=2,
        file_path=get_file_path(target_resolution, "payload.json"),
    )

    with open(
        get_file_path(
            target_resolution=target_resolution, file_name="variable_names.json"
        ),
        "w",
    ) as f:
        json.dump(var_names, f)

    df_train_resampled.to_csv(
        get_file_path(target_resolution=target_resolution, file_name="train.csv"),
        index=False,
    )

    df_test_resampled.to_csv(
        get_file_path(target_resolution=target_resolution, file_name="test.csv"),
        index=False,
    )
