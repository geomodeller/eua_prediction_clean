import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.optim as optim
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import mean_squared_error

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        
        # Define layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),  # Layer normalization instead of batch norm for the last layer
            
            nn.Linear(512, output_dim),
            # nn.LeakyReLU(0.1)  # LeakyReLU for output regression flexibility
        )

# class NeuralNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(NeuralNetwork, self).__init__()
        
#         # Define layers
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Dropout(0.1),
            
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.1),
            
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.LayerNorm(512),  # Layer normalization instead of batch norm for the last layer
            
#             nn.Linear(512, output_dim),
#             # nn.LeakyReLU(0.1)  # LeakyReLU for output regression flexibility
#         )

    def forward(self, x):
        return self.model(x)
    """
    # Example usage:
    # Define input and output dimensions
    input_dim = X_train.shape[1]  # Replace with the actual input shape
    output_dim = len(predictors_lst)  # Replace with actual output dimensions

    # Initialize the model
    model = NeuralNetwork(input_dim=input_dim, output_dim=output_dim)

    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    """

def generate_nn_multi_step(X_train, y_train):
    input_dim, output_dim = X_train.shape[1], y_train.shape[1]
    model = NeuralNetwork(input_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the GPU
    return model

def train_NN(model, checkpoint_path, 
             X_train_tensor, y_train_tensor, 
             X_test_tensor, y_test_tensor, 
             epochs=1000, 
             batch_size=512, 
             patience=50,
             verbose: int | bool = 0):
    
    # Move model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_scheduler_func(epoch))
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Load existing model weights and training history if they exist
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        with open(checkpoint_path.replace('.pt', '_history.pkl'), 'rb') as f:
            history = pickle.load(f)
    except FileNotFoundError:
        history = {'train_loss': [], 'val_loss': []}
    
    # Convert data to PyTorch tensors and create data loaders    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Calculate average training loss
        train_loss = sum(train_losses) / len(train_losses)
        
        # Validation loss
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        
        val_loss = sum(val_losses) / len(val_losses)
        
        # Print epoch summary
        if verbose and type(verbose) == bool:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        elif verbose and type(verbose) == int and (epoch + 1)%verbose == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            
        # Save loss history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduler step
        lr_scheduler.step()
        
        # Early stopping and checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            # print(f"Model checkpoint saved at epoch {epoch+1}")
        else:
            patience_counter += 1
        
        if (epoch+1)%50 == 0:
            torch.save(model.state_dict(), checkpoint_path.replace('.pth', f'_epoch_{epoch+1}.pth'))
            # print(f"Model checkpoint saved at epoch {epoch+1}")
        
        # Check for early stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    # Save training history
    with open(checkpoint_path.replace('.pth', '_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    return history

# Custom learning rate scheduler function (equivalent to lr_scheduler)
def lr_scheduler_func(epoch, lr=0.001):
    return lr
