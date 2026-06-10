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
        date_format="%Y-%m-%d %H:%M",
        dtype={"id": str}
        )
    
    float_cols = timeseries_data.select_dtypes(include="float").columns 
    timeseries_data[float_cols] = timeseries_data[float_cols].astype("float32")
    
    # To save memory later
    timeseries_data["id"] = timeseries_data["id"].astype("category")
    
    data_train = timeseries_data[
        timeseries_data["id"].isin(config["id_train"]) & 
        timeseries_data["time"].between(config["train_period"][0], 
                                     config["train_period"][1])]
    
    data_valid = timeseries_data[
        timeseries_data["id"].isin(config["id_train"]) &
        timeseries_data["time"].between(config["valid_period"][0], 
                                     config["valid_period"][1])]
    
    data_test = timeseries_data[
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
            dtype={"id": str}
        )

        float_cols = static_data.select_dtypes(include="float").columns
        static_data[float_cols] = static_data[float_cols].astype("float32")
        
        static_data["id"] = static_data["id"].astype("category")
        static_data = static_data.set_index("id")
    else:
        static_data = None

    return {'timeseries_data_train':data_train,
            'timeseries_data_valid':data_valid, 
            'timeseries_data_test':data_test,
            'static_data':static_data}


#-----------------------------------------------------------------------------#
#                         Read train test valid data                          #
#-----------------------------------------------------------------------------#
def read_inference_data(config:dict=None, keep_target_features=True) -> dict:
    
    # The column names must contains the following names
    require_columns = [
        'id',
        'time',
        *config['input_timeseries_features']
        ]
    
    if keep_target_features:
        require_columns += config['target_features']
    
    timeseries_data = pd.read_csv(
        config['timeseries_data_file_inference'][0],
        usecols=require_columns,
        parse_dates=["time"],
        date_format="%Y-%m-%d %H:%M",
        dtype={"id": str}
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
            dtype={"id": str}
        )

        float_cols = static_data.select_dtypes(include="float").columns
        static_data[float_cols] = static_data[float_cols].astype("float32")
        
        static_data["id"] = static_data["id"].astype("category")
        static_data = static_data.set_index("id")
        
    else:
        static_data = None
        
    return {'inference_timeseries_data':inference_data,
            'inference_static_data':static_data}


#-----------------------------------------------------------------------------#
#                         Read scale inference data                           #
#-----------------------------------------------------------------------------#
def read_scale_inference_data(config, scaler):
    
    inference_data = read_inference_data(config)
    
    return {'inference_data_scaled':scaler.transform(inference_data)}


#-----------------------------------------------------------------------------#
#                         Read scale inference data                           #
#-----------------------------------------------------------------------------#
def get_scaler_name(config, timeseries = True):
    
    if timeseries:
        col_scaler = dict(
            zip(
                config["input_timeseries_features"] + 
                config["target_features"],

                len(config["input_timeseries_features"]) *
                config["scaler_input_timeseries_features"] + 
                
                len(config["target_features"])*
                config["scaler_target_features"]
                
                )
            )
    else:
        col_scaler = dict(
            zip(
                config["input_static_features"],
                len(config["input_static_features"]) *
                config["scaler_input_static_features"]
                )
            )
    
    return col_scaler
    
#-----------------------------------------------------------------------------#
#          Combine static and dynmiac method for the model                    #
#-----------------------------------------------------------------------------#
def combine_timeseries_static(timeseries_data:pd.DataFrame, 
                              static_data:pd.DataFrame,
                              model,
                              keep_target_features=True):
    
    if keep_target_features: 
        col_names = (['id', 'time'] + model.input_timeseries_features + 
                     model.target_features)
    else:
        col_names = ['id', 'time'] + model.input_timeseries_features
        
    # Select and resort column order
    combined_data = timeseries_data[col_names].copy()
    
    # Now join time series and static data together
    for name in model.input_static_features:
        
        combined_data[name] = combined_data['id'].map(
            static_data[name]).astype("float32")
    
    return combined_data
    

    
    
    
    
    
    
    
    
       
    









    
    