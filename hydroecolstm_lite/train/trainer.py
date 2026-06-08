import torch
import copy
import numpy as np
import pandas as pd
from pathlib import Path
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
        self.input_timeseries_features = model.input_timeseries_features
        self.input_static_features = model.input_static_features
        self.input_features = model.input_features
        
    # Train function
    def train(self, 
              timeseries_data_train:pd.DataFrame,
              timeseries_data_valid:pd.DataFrame,
              static_data:pd.DataFrame):
        
        # Make sure column names order as in the model
        col_names = (['id', 'time'] + 
                     self.input_timeseries_features + 
                     self.target_features)

        # Select and resort column order
        timeseries_data_train = timeseries_data_train[col_names]
        timeseries_data_valid = timeseries_data_valid[col_names]
        
        # Create custom dataset
        xy_train = CustomDataset(timeseries_data_train, static_data, 
                                 self.model, self.warmup_length,
                                 self.sequence_length)
        
        xy_valid = CustomDataset(timeseries_data_valid, static_data, 
                                 self.model, self.warmup_length,
                                 self.sequence_length)
        
        print("Number of iteration per epoch = ", 
              int(xy_train.__len__()/self.batch_size))
        
        # Train and valid loss per epoch
        train_loss_epoch = []
        valid_loss_epoch = []
        
        patience = 0
        
        # Train the model
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr[0])
        
        for epoch in range(self.n_epochs):
            
            for param_group in optim.param_groups:
                param_group["lr"] = (self.lr[0] + (self.lr[1] - self.lr[0])*
                                     epoch/(self.n_epochs -1))
            
            patience += 1
            
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
                
                if not torch.isnan(loss):
                    
                    # Backward prop
                    loss.backward()
                    
                    # Update weights and biases
                    optim.step()
                    
                else:
                    print("Loss is nan, skip this batch")
                
                # Save traning loss 
                train_loss_batch.append(loss.item())
                
            # Set model to eval mode (in this mode, dropout = 0, no normlization)
            self.model.eval()

            # Save model state dict
            torch.save(self.model.state_dict(), 
                       Path(self.out_dir, "epoch_" + 
                            str(epoch) + "_state_dict.pt"))
            
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
            
            if epoch == 0:
                best_loss = np.average(valid_loss_batch)
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), 
                           Path(self.out_dir, "best_model_state_dict.pt"))
                print(f"Saved best model state dict at epoch {epoch+1}")

            else:
                if np.average(valid_loss_batch) < best_loss:
                    patience = 0
                    best_loss = np.average(valid_loss_batch)
                    self.best_state_dict = copy.deepcopy(
                        self.model.state_dict()
                        )
                    torch.save(self.model.state_dict(), 
                               Path(self.out_dir, "best_model_state_dict.pt"))
                    print(f"Saved best model state dict at epoch {epoch+1}")

            if patience > self.patience:
                print("Early stopping")
                break
        
        
        self.model.load_state_dict(self.best_state_dict)
        print(f"Model with the lowest validation loss was selected: {best_loss:.8f}")
            
        self.loss_epoch = pd.DataFrame({
            'train_loss': train_loss_epoch,
            'valid_loss': valid_loss_epoch})

        return self.model