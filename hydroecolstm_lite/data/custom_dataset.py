import torch
import pandas as pd
from torch.utils.data import Dataset
                 
#timeseries_data = data_train.copy()
        
class CustomDataset(Dataset):
    def __init__(self,
                 timeseries_data:pd.DataFrame, 
                 static_data:pd.DataFrame, 
                 model, 
                 warmup_length:int,
                 sequence_length:int):
        
        # Check order of id in timeseries
        id_order = timeseries_data['id'].unique().tolist()
        
        static_data_order = static_data.loc[id_order].copy()
        
        if len(static_data_order.index) != len(id_order):
            raise ValueError("mismatch id(s) in static file")
            
        # Don't use observed values in warm up period
        idx = timeseries_data.reset_index().index[
            timeseries_data.groupby('id', observed=True).cumcount() < warmup_length
            ]
        
        self.Y = torch.tensor(timeseries_data[model.target_features].values,
                              dtype=torch.float32)
        self.Y[idx,:] = torch.nan
        
        repeats = torch.tensor(
            timeseries_data['id'].astype(str).value_counts(sort=False).values)
        
        static_data_order = torch.tensor(
            static_data_order[model.input_static_features].values,
            dtype=torch.float32)
        
        static_data_order = torch.repeat_interleave(static_data_order, 
                                                    repeats, dim=0)

        timeseries_data = torch.tensor(
            timeseries_data[model.input_timeseries_features].values,
            dtype=torch.float32
            )
        
        self.X = torch.cat([timeseries_data, static_data_order], dim=1)
        self.warmup_length = warmup_length
        self.sequence_length = sequence_length
        
    def __getitem__(self, index):
        
        istart = index*(self.sequence_length - self.warmup_length)
        iend = istart + self.sequence_length

        # Mini batch data
        if iend <= len(self.X):
            x_batch = self.X[istart:iend,:]
            y_batch = self.Y[istart:iend,:].clone()
        else:
            x_batch = self.X[-self.sequence_length:,:]
            y_batch = self.Y[-self.sequence_length:,:].clone()
            
        # Don't use observed values in warm up period
        y_batch[:self.warmup_length] = torch.nan
        
        # Return mini batch consist of input, output
        return x_batch, y_batch
    
    def __len__(self):
        num_batches = int(round((len(self.X) - self.warmup_length)/
                          (self.sequence_length - self.warmup_length) + 0.49))                   
        return num_batches 
