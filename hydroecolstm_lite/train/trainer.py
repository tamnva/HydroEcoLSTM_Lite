import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from hydroecolstm_lite.train.custom_loss import CustomLoss
from hydroecolstm_lite.data.custom_dataset import CustomDataset

# LSTM + Linears
class Trainer():
    def __init__(self, config, model):

        # Training parameters
        self.lr = config["learning_rate"]
        self.loss_function = CustomLoss(config["loss_function"])
        self.n_epochs = config["n_epochs"]
        self.batch_size = config["batch_size"]
        self.model = model
        self.patience = config["patience"]
        self.out_dir = config["output_directory"][0]
        self.loss_epoch = None
        self.best_state_dict = None
        self.best_train_loss = None
        self.warmup_length = config['warmup_length']
        self.sequence_length = config['sequence_length']
        self.target_features = model.target_features
        self.input_features = model.input_features
        
    # Train function
    def train(self, data_train_scaled:pd.DataFrame, 
              data_valid_scaled:pd.DataFrame):
        
        # Make sure column names order as in the model
        col_names = ['id', 'time'] + self.input_features + self.target_features
        data_train_scaled = data_train_scaled[col_names]
        data_valid_scaled = data_valid_scaled[col_names]
        
        # Optimization function
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                
        # Create custom dataset
        xy_train = CustomDataset(data_train_scaled, self.input_features, 
                                 self.target_features, self.warmup_length,
                                 self.sequence_length)
        
        xy_valid = CustomDataset(data_valid_scaled, self.input_features, 
                                 self.target_features, self.warmup_length,
                                 self.sequence_length)
        
        print("Number of iteration per epoch = ", 
              int(xy_train.__len__()/self.batch_size))
        
        # Train and valid loss per epoch
        train_loss_epoch = []
        valid_loss_epoch = []
        
        # initialize early stoping
        early_stopping = EarlyStopping(patience=self.patience, verbose=False,
                                       path=self.out_dir)
        
        # Train the model
        for epoch in range(self.n_epochs):
            
            # Create batch data for each epoch
            xy_train_batch = DataLoader(xy_train, self.batch_size, shuffle=True)
            xy_valid_batch = DataLoader(xy_valid, self.batch_size, shuffle=True)
            
            # Create list to store train and valid loss per batch
            train_loss_batch = []
            valid_loss_batch = []
            
            # Set model to train mode
            self.model.train()
            
            # Loop over batches
            for x_batch, y_batch in xy_train_batch:
                  
                # Get model output
                y_predict = self.model(x_batch)

                # Reset the gradients to zero
                optim.zero_grad()
                
                # Loss value    
                loss = self.loss_function(y_batch, y_predict)
                 
                # Backward prop
                loss.backward()
                    
                # Update weights and biases
                optim.step()
                
                # Save traning loss 
                train_loss_batch.append(loss.item())
                
            # Set model to eval mode (in this mode, dropout = 0, no normlization)
            self.model.eval()

            # Loop over batches
            with torch.inference_mode():
                for x_batch, y_batch in xy_valid_batch:
                    
                    # Forward pass:
                    y_predict = self.model(x_batch)
                    
                    # Get Loss
                    loss = self.loss_function(y_batch, y_predict)
                    
                    # Save traning loss 
                    valid_loss_batch.append(loss.item())
            

            # Store average loss per epoch for training and validation
            train_loss_epoch.append(np.average(train_loss_batch))
            valid_loss_epoch.append(np.average(valid_loss_batch))
            
            print(f"Epoch [{epoch+1}/{self.n_epochs}]:", 
                  f"train_loss = {train_loss_epoch[-1]:.8f},",
                  f"valid_loss = {valid_loss_epoch[-1]:.8f}")
                
            # Early stopping based on validation loss and make checkpoint            
            if early_stopping.early_stop:
                print("Early stopping.")
                break
            
        # If the model does not stops until the last epoch
        # It means that the best model is from last epoch
        if (epoch + 1) == self.n_epochs:
            print("Validation loss continue decreasing. Take last model")
            
        else:
            # Load the last checkpoint with the best model
            self.model.load_state_dict(self.best_state_dict)
            
            # Set model to eval mode
            self.model.eval()
            pass
            
        self.loss_epoch = pd.DataFrame({
            'train_loss': train_loss_epoch,
            'valid_loss': valid_loss_epoch})

        return self.model
    
# ----------------------------------------------------------------------------#
# The EarlyStopping code was copied from                                     #
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
# MIT License                                                                 #
# Copyright (c) 2018 Bjarte Mehus Sunde                                       #
#                                                                             #                                       #
# ----------------------------------------------------------------------------#
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path:str=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss
        flag = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            flag = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            flag = True
        return flag

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss