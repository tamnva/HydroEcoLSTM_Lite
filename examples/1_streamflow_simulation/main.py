import torch
from hydroecolstm_lite.model_run import run_config
from hydroecolstm_lite.data.read_config import read_config
from hydroecolstm_lite.utility.evaluation_function import nse

#-----------------------------------------------------------------------------#
#                        Set up, train, test model                            #
#-----------------------------------------------------------------------------#

# Read configuration file, please modify the path to the config.yml file
config = read_config("C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM_Lite/examples/1_streamflow_simulation/config.yml")

data, model, loss_epoch, scaler = run_config(config)

# Convert test data to torch tensor to feed into the model
test_data_scaled = torch.tensor(
    data['test_data_scaled'][model.input_features].values, 
    dtype=torch.float32)

# Run in inference mode
model.eval()
with torch.inference_mode(): 
    test_simulated_scaled = model(test_data_scaled)

# Add results to data frame
data["test_data_scaled"]["simulated"] = test_simulated_scaled.flatten().detach().numpy()

# Get NSE test period for each basins
(data["test_data_scaled"].groupby("id", observed=True).apply(
    lambda g: nse(g["simulated"], g["discharge_vol_m3_s"], 60),
    include_groups=False).rename("nse").reset_index())

