import pandas as pd
    
#-----------------------------------------------------------------------------#
#                         Read train test valid data                          #
#-----------------------------------------------------------------------------#
def read_train_valid_test_data(config:dict=None) -> dict:
    
    # The column names must contains the following names
    require_columns = [
        'id', 
        'time', 
        *config['input_timeseries_features'],         
        *config['target_features']
        ]
    
    timeseries_data = pd.read_csv(
        config["timeseries_data_file"][0],
        usecols=require_columns,
        parse_dates=["time"],
        date_format="%Y-%m-%d %H:%M"
        )
    
    float_cols = timeseries_data.select_dtypes(include="float").columns 
    timeseries_data[float_cols] = timeseries_data[float_cols].astype("float32")
    
    # To save memory later
    timeseries_data["id"] = timeseries_data["id"].astype("category")
    
    train_data = timeseries_data[
        timeseries_data["id"].isin(config["id_train"]) & 
        timeseries_data["time"].between(config["train_period"][0], 
                                     config["train_period"][1])]
    
    valid_data = timeseries_data[
        timeseries_data["id"].isin(config["id_train"]) &
        timeseries_data["time"].between(config["valid_period"][0], 
                                     config["valid_period"][1])]
    
    test_data = timeseries_data[
        timeseries_data["id"].isin(config["id_test"]) & 
        timeseries_data["time"].between(config["test_period"][0], 
                                     config["test_period"][1])]
    
    del timeseries_data
    
    # Read static input data file    
    if 'input_static_features' in config:
        
        require_columns = ["id", *config["input_static_features"]]
        
        static_data = pd.read_csv(
            config["static_data_file"][0],
            usecols=require_columns,
        )

        float_cols = static_data.select_dtypes(include="float").columns
        static_data[float_cols] = static_data[float_cols].astype("float32")
        
        static_data["id"] = static_data["id"].astype("category")
        static_data = static_data.set_index("id")
        
        # map is better than join in term of memory
        for name in config["input_static_features"]:
            train_data[name] = train_data['id'].map(
                static_data[name]).astype("float32")
            
            valid_data[name] = valid_data['id'].map(
                static_data[name]).astype("float32")
            
            test_data[name] = test_data['id'].map(
                static_data[name]).astype("float32")

        
    return {'train_data':train_data,
            'valid_data':valid_data, 
            'test_data':test_data}


#-----------------------------------------------------------------------------#
#                         Read train test valid data                          #
#-----------------------------------------------------------------------------#
def read_inference_data(config:dict=None) -> dict:
    
    # The column names must contains the following names
    require_columns = [
        'id',
        'time',
        *config['input_timeseries_features']
        ]
    
    timeseries_data = pd.read_csv(
        config['timeseries_data_file_inference'][0],
        usecols=require_columns,
        parse_dates=["time"],
        date_format="%Y-%m-%d %H:%M"
        )
    
    float_cols = timeseries_data.select_dtypes(include="float").columns 
    timeseries_data[float_cols] = timeseries_data[float_cols].astype("float32")
    
    # To save memory later
    timeseries_data["id"] = timeseries_data["id"].astype("category")
    
    inference_data = timeseries_data[require_columns][
        timeseries_data["id"].isin(config["id_train"]) & 
        (timeseries_data["time"] >= config["test_period"][0]) & 
        (timeseries_data["time"] <= config["test_period"][1])
        ]
    
    del timeseries_data
    
    # Read static input data file    
    if 'input_static_features' in config:
        
        require_columns = ["id", *config["input_static_features"]]
        
        static_data = pd.read_csv(
            config['static_data_file'][0],
            usecols=require_columns,
        )

        float_cols = static_data.select_dtypes(include="float").columns
        static_data[float_cols] = static_data[float_cols].astype("float32")
        
        static_data["id"] = static_data["id"].astype("category")
        static_data = static_data.set_index("id")
        
        # map is better than join in term of memory
        for name in config["input_static_features"]:
            inference_data[name] = inference_data['id'].map(
                static_data[name]).astype("float32")
        
    return {'inference_data':inference_data}


#-----------------------------------------------------------------------------#
#                         Read scale inference data                           #
#-----------------------------------------------------------------------------#
def read_scale_inference_data(config, scaler):
    
    inference_data = read_inference_data(config)
    
    return {'inference_data_scaled':scaler.transform(x=inference_data)}


def get_scaler_name(config):
    
    scaler_name = config["scaler_input_timeseries_features"]*len(
        config['input_timeseries_features']) + \
    config["scaler_target_features"]* len(config["target_features"])
    
    if "input_static_features" in config.keys():
        scaler_name +=  config["scaler_input_static_features"]*len(
            config["input_static_features"])
    
    return scaler_name