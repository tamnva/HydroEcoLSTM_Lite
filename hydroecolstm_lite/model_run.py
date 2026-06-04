#!/usr/bin/env python

from hydroecolstm_lite.data.read_data import get_scaler_name, read_train_valid_test_data
from hydroecolstm_lite.data.scaler import Scaler
from hydroecolstm_lite.model.create_model import create_model
from hydroecolstm_lite.train.trainer import Trainer

# Function to train and test the model 
def run_config(config):

    data = read_train_valid_test_data(config)
    
    # Scale/transformer name for static, timeseries, and target features
    scaler_names = get_scaler_name(config)
    
    # Scaler/transformer
    scaler = Scaler()
    scaler.fit(data=data['train_data'], scaler_names=scaler_names)
    
    data['train_data_scaled'] = scaler.transform(data['train_data'])
    del data['train_data'] 
    
    data['valid_data_scaled'] = scaler.transform(data['valid_data'])
    del data['valid_data'] 
    
    data['test_data_scaled'] = scaler.transform(data['test_data'])
    del data['test_data']  
    
    model = create_model(config)
        
    trainer = Trainer(config, model)
    
    model = trainer.train(data['train_data_scaled'], data['valid_data_scaled'])
        
    loss_epoch = trainer.loss_epoch
        
    return data, model, loss_epoch, scaler