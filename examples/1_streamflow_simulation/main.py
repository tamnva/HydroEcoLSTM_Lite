
from hydroecolstm_lite.model_run import run_config
from hydroecolstm_lite.data.read_config import read_config
from hydroecolstm_lite.utility.evaluation_function import nse

#-----------------------------------------------------------------------------#
#                        Set up, train, test model                            #
#-----------------------------------------------------------------------------#

# Read configuration file, please modify the path to the config.yml file
config = read_config("C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM_Lite/examples/1_streamflow_simulation/config.yml")

data, model, loss_epoch = run_config(config)

y_test_simulated = model(data['test_data_scaled'])

data['test_data_scaled']["simulated"] = y_test_simulated.flatten().detach().numpy()  

# Get NSE test period for each basins
(data["test_data_scaled"].groupby("id", observed=True).apply(
    lambda g: nse(g["simulated"], g["discharge_vol_m3_s"], 60),
    include_groups=False).rename("nse").reset_index())

# Plot for sepcfic pbasin
data['test_data_scaled'][data['test_data_scaled']["id"] == 2033].plot()