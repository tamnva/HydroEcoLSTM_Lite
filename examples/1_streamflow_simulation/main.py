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

data_scaled, scaler, model, trainer = run_config(config)

# Combine time series and statics
test_data_scaled = combine_timeseries_static(
    data_scaled['timeseries_data_test'], data_scaled['static_data'], model)


# Run inference with test data
model.eval()

with torch.inference_mode():
    for ids in test_data_scaled["id"].unique(): 
        # Loop over basin 
        input_data = test_data_scaled[test_data_scaled["id"] == ids]
        input_data = torch.tensor(input_data[model.input_features].values)
        
        if ids == test_data_scaled["id"].unique()[0]:
            simulated = model(input_data)
        else:
            simulated = torch.cat((simulated, model(input_data)), dim = 0)
    
# Add results to data frame
data_scaled['timeseries_data_test']["simulated"] = simulated.flatten().numpy()

# Get NSE test period for each basins
nse_val = (data_scaled['timeseries_data_test'].
           groupby("id", observed=True).
           apply(lambda g: nse(g["simulated"], 
                               g["discharge_vol_m3_s"], 
                               60),
                 include_groups=False))
nse_val.mean()




static_data = data_scaled["static_data"].copy()
timeseries_data = data_scaled["timeseries_data_train"].copy()
warmup_length = config["warmup_length"]
