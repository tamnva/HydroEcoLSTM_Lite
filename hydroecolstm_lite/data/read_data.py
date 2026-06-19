import pandas as pd


"""Data loading utilities for HydroEcoLSTM-Lite.

This module contains helpers to read, filter and prepare time series and
static attribute CSV files according to the package configuration dict.
Each function returns pandas DataFrame objects or dictionaries of DataFrames
that are ready to be passed into downstream scaling and modelling utilities.
"""


#-----------------------------------------------------------------------------#
#                         Read train test valid data                          #
#-----------------------------------------------------------------------------#
def read_train_valid_test_data(config: dict = None) -> dict:
    """Read and split timeseries and static data into train/valid/test.

    The function expects `config` to contain keys such as
    `timeseries_data_file`, `input_timeseries_features`, `target_features`,
    `id_train`, `id_valid`, `id_test` and corresponding time period ranges.

    Parameters
    ----------
    config : dict
        Configuration dictionary describing file locations and splits.

    Returns
    -------
    dict
        Dictionary with keys `timeseries_data_train`, `timeseries_data_valid`,
        `timeseries_data_test` and `static_data`.
    """

    # The column names must contains the following names
    require_columns = [
        "id",
        "time",
        *config["input_timeseries_features"],
        *config["target_features"],
    ]

    timeseries_data = pd.read_csv(
        config["timeseries_data_file"][0],
        usecols=require_columns,
        parse_dates=["time"],
        dtype={"id": str},
    )

    float_cols = timeseries_data.select_dtypes(include="float").columns
    timeseries_data[float_cols] = timeseries_data[float_cols].astype("float32")

    # To save memory later
    timeseries_data["id"] = timeseries_data["id"].astype("category")

    data_train = timeseries_data[
        timeseries_data["id"].isin(config["id_train"]) & timeseries_data["time"].between(
            config["train_period"][0], config["train_period"][1]
        )
    ]

    data_valid = timeseries_data[
        timeseries_data["id"].isin(config["id_valid"]) & timeseries_data["time"].between(
            config["valid_period"][0], config["valid_period"][1]
        )
    ]

    data_test = timeseries_data[
        timeseries_data["id"].isin(config["id_test"]) & timeseries_data["time"].between(
            config["test_period"][0], config["test_period"][1]
        )
    ]

    del timeseries_data

    # Read static input data file
    if "input_static_features" in config:

        require_columns = ["id", *config["input_static_features"]]

        static_data = pd.read_csv(
            config["static_data_file"][0], usecols=require_columns, dtype={"id": str}
        )

        float_cols = static_data.select_dtypes(include="float").columns
        static_data[float_cols] = static_data[float_cols].astype("float32")

        static_data["id"] = static_data["id"].astype("category")
        static_data = static_data.set_index("id")
    else:
        static_data = None

    return {
        "timeseries_data_train": data_train,
        "timeseries_data_valid": data_valid,
        "timeseries_data_test": data_test,
        "static_data": static_data,
    }


#-----------------------------------------------------------------------------#
#                         Read train test valid data                          #
#-----------------------------------------------------------------------------#
def read_inference_data(config: dict = None, keep_target_features: bool = True) -> dict:
    """Read data intended for inference (prediction) windows.

    Parameters
    ----------
    config : dict
        Configuration describing inference file paths and id/time filters.
    keep_target_features : bool, optional
        If True include target columns in the returned DataFrame, by default True.

    Returns
    -------
    dict
        Dictionary with keys `inference_timeseries_data` and `inference_static_data`.
    """

    # The column names must contains the following names
    require_columns = ["id", "time", *config["input_timeseries_features"]]

    if keep_target_features:
        require_columns += config["target_features"]

    timeseries_data = pd.read_csv(
        config["timeseries_data_file_inference"][0],
        usecols=require_columns,
        parse_dates=["time"],
        dtype={"id": str},
    )

    float_cols = timeseries_data.select_dtypes(include="float").columns
    timeseries_data[float_cols] = timeseries_data[float_cols].astype("float32")

    # To save memory later
    timeseries_data["id"] = timeseries_data["id"].astype("category")

    mask = (
        timeseries_data["id"].isin(config["id_inference"]) & (timeseries_data["time"] >= config["inference_period"][0]) & (timeseries_data["time"] <= config["inference_period"][1])
    )

    inference_data = timeseries_data.loc[mask, require_columns]

    del timeseries_data

    # Read static input data file
    if "input_static_features" in config:

        require_columns = ["id", *config["input_static_features"]]

        static_data = pd.read_csv(
            config["static_data_file"][0], usecols=require_columns, dtype={"id": str}
        )

        float_cols = static_data.select_dtypes(include="float").columns
        static_data[float_cols] = static_data[float_cols].astype("float32")

        static_data["id"] = static_data["id"].astype("category")
        static_data = static_data.set_index("id")

    else:
        static_data = None

    return {"inference_timeseries_data": inference_data, "inference_static_data": static_data}


#-----------------------------------------------------------------------------#
#                         Read scale inference data                           #
#-----------------------------------------------------------------------------#
def read_scale_inference_data(config, scaler, keep_target_features: bool = True):
    """Read inference data and apply provided `scaler` transforms.

    Parameters
    ----------
    config : dict
        Configuration for inference.
    scaler : dict
        Dictionary containing `timeseries_data` and `static_data` Scaler instances.
    keep_target_features : bool, optional
        Passed to `read_inference_data`, by default True.

    Returns
    -------
    dict
        Same structure as `read_inference_data` but with scaled DataFrames.
    """

    inference_data = read_inference_data(config)

    inference_data["inference_timeseries_data"] = scaler["timeseries_data"].transform(
        inference_data["inference_timeseries_data"]
    )

    inference_data["inference_static_data"] = scaler["static_data"].transform(
        inference_data["inference_static_data"]
    )

    return inference_data


#-----------------------------------------------------------------------------#
#                         Read scale inference data                           #
#-----------------------------------------------------------------------------#
def get_scaler_name(config, timeseries=True):
    """Construct a mapping of column names to scaler types.

    The returned dict maps feature column names to scaler identifiers
    (e.g. "min_max", "z_score", "none") based on the configuration.
    """

    if timeseries:
        col_scaler = dict(
            zip(
                config["input_timeseries_features"] + config["target_features"],

                len(config["input_timeseries_features"]) * config["scaler_input_timeseries_features"]
                + len(config["target_features"]) * config["scaler_target_features"],
            )
        )
    else:
        col_scaler = dict(
            zip(
                config["input_static_features"],
                len(config["input_static_features"]) * config["scaler_input_static_features"],
            )
        )

    return col_scaler


#-----------------------------------------------------------------------------#
#          Combine static and dynmiac method for the model                    #
#-----------------------------------------------------------------------------#
def combine_timeseries_static(timeseries_data: pd.DataFrame, static_data: pd.DataFrame, model, keep_target_features=True):
    """Combine timeseries rows with static attributes for model input.

    Parameters
    ----------
    timeseries_data : pandas.DataFrame
        Timeseries data containing columns `id`, `time` and feature columns.
    static_data : pandas.DataFrame
        Static attributes indexed by `id`.
    model : object
        Model-like object exposing `input_timeseries_features`, `target_features`
        and `input_static_features` attributes used to select columns.
    keep_target_features : bool, optional
        Whether to keep target columns in the combined output, by default True.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame ready for model consumption.
    """

    if keep_target_features:
        col_names = (["id", "time"] + model.input_timeseries_features + model.target_features)
    else:
        col_names = ["id", "time"] + model.input_timeseries_features

    # Select and resort column order
    combined_data = timeseries_data[col_names].copy()

    # Now join time series and static data together
    for name in model.input_static_features:
        combined_data[name] = combined_data["id"].map(static_data[name]).astype("float32")

    return combined_data
    
    




















