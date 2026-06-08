#!/usr/bin/env python

from hydroecolstm_lite.data.read_data import read_train_valid_test_data
from hydroecolstm_lite.data.read_data import get_scaler_name
from hydroecolstm_lite.data.scaler import Scaler
from hydroecolstm_lite.model.create_model import create_model
from hydroecolstm_lite.train.trainer import Trainer

# Function to train and test the model 
def run_config(config):

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
        if 'timeseries_data' in key:
            data_scaled[key] = scaler["timeseries_data"].transform(data[key])
        else:
            data_scaled[key] = scaler["static_data"].transform(data[key])
        
    del data[key]
    
    model = create_model(config)
        
    trainer = Trainer(config, model)
    
    model = trainer.train(
        data_scaled['timeseries_data_train'],
        data_scaled['timeseries_data_valid'],
        data_scaled['static_data']
        )
        
    return data_scaled, scaler, model, trainer

