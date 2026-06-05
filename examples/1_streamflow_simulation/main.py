import torch
from hydroecolstm_lite.model_run import run_config
from hydroecolstm_lite.data.read_config import read_config
from hydroecolstm_lite.data.read_data import combine_timeseries_static
from hydroecolstm_lite.utility.evaluation_function import nse

#-----------------------------------------------------------------------------#
#                        Set up, train, test model                            #
#-----------------------------------------------------------------------------#

# Read configuration file, please modify the path to the config.yml file
config = read_config("C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM_Lite/examples/1_streamflow_simulation/config.yml")

data_scaled, scaler, model, loss_epoch = run_config(config)

# Combine time series and statics
test_data_scaled = combine_timeseries_static(
    data_scaled['timeseries_data_train'], data_scaled['static_data'], model)

# Convert to model input
test_data_scaled = torch.tensor(test_data_scaled[model.input_features].values,
                                dtype=torch.float32)

# Run inference with test data
model.eval()
with torch.inference_mode(): 
    simulated = model(test_data_scaled)
    
# Add results to data frame
data_scaled['timeseries_data_train']["simulated"] = simulated.flatten().numpy()

# Get NSE test period for each basins
(data_scaled['timeseries_data_train'].groupby("id", observed=True).apply(
    lambda g: nse(g["simulated"], g["discharge_vol_m3_s"], 60),
    include_groups=False).rename("nse").reset_index())