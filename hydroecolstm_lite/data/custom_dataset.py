import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, 
                 data:pd.DataFrame, 
                 input_features:str, 
                 target_features:str,
                 warmup_length:int,
                 sequence_length:int):
        
        self.X = torch.tensor(data[input_features].values, 
                              dtype=torch.float32)
        
        self.Y = torch.tensor(data[target_features].values,
                              dtype=torch.float32)
        
        # Don't use observed values in warm up period
        idx = data.reset_index().index[
            data.groupby('id', observed=True).cumcount() < warmup_length
            ]
        
        self.Y[idx,:] = torch.nan
        
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
