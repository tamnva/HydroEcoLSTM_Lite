import torch
import numpy as np
from pathlib import Path
from hydroecolstm_lite.data.read_config import read_config
from hydroecolstm_lite.data.read_data import combine_timeseries_static
from hydroecolstm_lite.model.create_model import create_model
from hydroecolstm_lite.utility.get_device import get_device
from hydroecolstm_lite.utility.logger import get_logger
from hydroecolstm_lite.utility.evaluation_function import nse
from hydroecolstm_lite.data.read_data import read_train_valid_test_data
from hydroecolstm_lite.data.read_data import get_scaler_name
from hydroecolstm_lite.data.scaler import Scaler


# Read configuration file, please modify the path to the config.yml file
# lstm_data_dir = "/gpfs1/schlecker/home/nguyenta/gpu_test/kit_de"
lstm_data_dir = "C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM_Lite/examples/camels_de"

# Read and update config file
config = read_config(Path(lstm_data_dir, "config.yml"))
config["timeseries_data_file"][0] = Path(lstm_data_dir, "time_series.csv")
config["static_data_file"][0] = Path(lstm_data_dir, "static_attributes.csv")
config["timeseries_data_file_inference"] = config["timeseries_data_file"]
config["static_data_file_inference"] = config["static_data_file"]
config["output_directory"][0] = Path(lstm_data_dir)

# configure logger early so examples write logs to output directory
logger = get_logger(config)

# Note: see link in readme file to download processed camles-de data
#-----------------------------------------------------------------------------#
#             The code within this section is used for training               #
#-----------------------------------------------------------------------------#
data = read_train_valid_test_data(config)

# Transform timeseries and static attributes
col_scaler_timeseries = get_scaler_name(config, True)
col_scaler_static = get_scaler_name(config, False)

scaler = {}

scaler["timeseries_data"] = Scaler()
scaler["timeseries_data"].fit(data["timeseries_data_train"], 
                              col_scaler_timeseries)

scaler["static_data"] = Scaler()
scaler["static_data"].fit(data["static_data"], col_scaler_static)

data_scaled = {}

for key in data.keys():
    if "timeseries_data" in key:
        data_scaled[key] = scaler["timeseries_data"].transform(data[key])
    else:
        data_scaled[key] = scaler["static_data"].transform(data[key])    

model = create_model(config, Path(lstm_data_dir, "state_dict.pt"))

# Save data and model
# torch.save(scaler, Path(lstm_data_dir, "scaler.pt"))
# torch.save(data_scaled, Path(lstm_data_dir, "data_scaled.pt"))

scaler = torch.load(Path(lstm_data_dir, "scaler.pt"), weights_only=False)
data_scaled = torch.load(Path(lstm_data_dir, "data_scaled.pt"), weights_only=False)

# Combine time series and statics
test_data = combine_timeseries_static(
    data_scaled['timeseries_data_test'], 
    data_scaled['static_data'], model
    )

basin_orig = test_data[test_data["id"] == "DE110000"].copy()
basin_mod = test_data[test_data["id"] == "DE110000"].copy()

# Transform basin attributes to normal => modify => transform again
basin_mod = scaler["static_data"].inverse(basin_mod)
basin_mod["slope_deg"] = basin_mod["slope_deg"] 
basin_mod["area"] = basin_mod["area"]*10
basin_mod = scaler["static_data"].transform(basin_mod)

basin_orig_tensor = torch.tensor(basin_orig[model.input_features].values, dtype=torch.float32)
basin_mod_tensor = torch.tensor(basin_mod[model.input_features].values, dtype=torch.float32)


simulated_orig = basin_orig[["id", "time"]].copy()
simulated_orig["discharge_spec_obs"] = np.nan

simulated_mod = basin_mod[["id", "time"]].copy()
simulated_mod["discharge_spec_obs"] = np.nan


model.eval()
with torch.inference_mode():
    # ensure model and inputs are on the same device to avoid device mismatch
    simulated_orig["discharge_spec_obs"] = model(basin_orig_tensor).squeeze().detach().numpy()
    simulated_mod["discharge_spec_obs"] = model(basin_mod_tensor).squeeze().detach().numpy()


simulated_orig = scaler["timeseries_data"].inverse(simulated_orig)
simulated_mod = scaler["timeseries_data"].inverse(simulated_mod)

simulated_orig["new"] = simulated_mod["discharge_spec_obs"]
simulated_orig.plot(x="time",y=["discharge_spec_obs", "new"])
