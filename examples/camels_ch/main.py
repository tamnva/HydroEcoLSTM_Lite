import torch
from pathlib import Path
from hydroecolstm_lite.model_run import run_config
from hydroecolstm_lite.data.read_config import read_config
from hydroecolstm_lite.data.read_data import combine_timeseries_static
from hydroecolstm_lite.utility.evaluation_function import nse

#-----------------------------------------------------------------------------#
#                        Set up, train, test model                            #
#-----------------------------------------------------------------------------#

# Read configuration file, please modify the path to the config.yml file
lstm_data_dir = "C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM_Lite/examples/camels_ch"

# Read and update config file
config = read_config(Path(lstm_data_dir, "config.yml"))
config["timeseries_data_file"][0] = Path(lstm_data_dir, "time_series.csv")
config["static_data_file"][0] = Path(lstm_data_dir, "static_attributes.csv")
config["timeseries_data_file_inference"] = config["timeseries_data_file"]
config["static_data_file_inference"] = config["static_data_file"]
config["output_directory"][0] = Path(lstm_data_dir)

# Note: see link in readme file to download processed camles-de data
#-----------------------------------------------------------------------------#
#             The code within this section is used for training               #
#-----------------------------------------------------------------------------#
# Just train for scratch, don't use initial state dict
del config["init_model_state_dict"]

data_scaled, scaler, model, trainer = run_config(config)

# Combine time series and statics
test_data_scaled = combine_timeseries_static(
    data_scaled['timeseries_data_test'], data_scaled['static_data'], model)

# Run inference with test data
model.eval()
simulated = torch.empty(0, len(config["target_features"]))

with torch.inference_mode():
    for ids in test_data_scaled["id"].unique(): 
        input_data = test_data_scaled[test_data_scaled["id"] == ids]
        input_data = torch.tensor(input_data[model.input_features].values)
        simulated = torch.cat((simulated, model(input_data)), dim = 0)
    
# Add results to data frame
data_scaled['timeseries_data_test']["simulated"] = simulated.flatten().numpy()

# Get NSE test period for each basins
nse_val = (data_scaled['timeseries_data_test'].
           groupby("id", observed=True).
           apply(lambda g: nse(g["simulated"], 
                               g["discharge_vol_m3_s"], 
                               config["warmup_length"]),
                 include_groups=False))
nse_val.mean()


    
    
