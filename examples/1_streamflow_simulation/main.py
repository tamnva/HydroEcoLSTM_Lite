
from hydroecolstm_lite.model_run import run_config
from hydroecolstm_lite.data.read_config import read_config
from hydroecolstm_lite.utility.evaluation_function import nse

#-----------------------------------------------------------------------------#
#                        Set up, train, test model                            #
#-----------------------------------------------------------------------------#

# Read configuration file, please modify the path to the config.yml file
config = read_config("C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM_Lite/examples/1_streamflow_simulation/config.yml")

data, model, loss_epoch, scaler = run_config(config)

y_test_simulated = model(data['test_data_scaled'])

data['test_data_scaled']["simulated"] = y_test_simulated.flatten().detach().numpy()  

# Get NSE test period for each basins
(data["test_data_scaled"].groupby("id", observed=True).apply(
    lambda g: nse(g["simulated"], g["discharge_vol_m3_s"], 60),
    include_groups=False).rename("nse").reset_index())

# Plot for sepcfic pbasin
data['test_data_scaled'][data['test_data_scaled']["id"] == 2033].plot()


    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    NONE = "none"

    VALID_SCALERS = {MIN_MAX, Z_SCORE, NONE}

    def __init__(self, exclude_columns = ("id", "time")) -> None:
        exclude_columns = list(exclude_columns)
        columns = None
        offset = None
        scale = None
        scaler_names = None

    def fit(self, data: pd.DataFrame, scaler_names: list[str]) -> "Scaler":
        """Calculate scaling parameters for each selected column."""
        _validate_dataframe(data)

        columns = data.columns.drop(exclude_columns, errors="ignore")

        if len(scaler_names) != len(columns):
            raise ValueError(
                f"Expected {len(columns)} scaler names, "
                f"but got {len(scaler_names)}."
            )

        invalid_scalers = set(scaler_names) - VALID_SCALERS
        if invalid_scalers:
            raise ValueError(f"Invalid scaler name(s): {sorted(invalid_scalers)}")

        stats = data[columns].agg(["min", "max", "mean", "std"])

        offset = pd.Series(0.0, index=columns)
        scale = pd.Series(1.0, index=columns)
