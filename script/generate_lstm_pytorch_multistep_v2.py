import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class LSTMModel_less_complex(nn.Module):
    def __init__(self, n_features, n_outputs):
        super(LSTMModel_less_complex, self).__init__()
        # Reduced hidden size and increased dropout rate
        hidden_size = 256
        dropout_rate = 0.25
        
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.repeat = n_outputs
        
        self.lstm3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.lstm4 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.layer_norm4 = nn.LayerNorm(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer_norm_dense = nn.LayerNorm(hidden_size // 2)
        self.dropout_dense = nn.Dropout(dropout_rate)  # Added dropout in dense layer
        self.dense2 = nn.Linear(hidden_size // 2, n_features)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.layer_norm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.layer_norm2(x)
        x = self.dropout2(x)
        
        x = x[:, -1, :].unsqueeze(1).repeat(1, self.repeat, 1)
        
        x, _ = self.lstm3(x)
        x = self.layer_norm3(x)
        x = self.dropout3(x)
        
        x, _ = self.lstm4(x)
        x = self.layer_norm4(x)
        x = self.dropout4(x)
        
        x = torch.tanh(self.dense1(x))
        x = self.layer_norm_dense(x)
        x = self.dropout_dense(x)  # Dropout in dense layer
        x = self.dense2(x)
        
        return x


class LSTMModel_much_less_complex(nn.Module):
    def __init__(self, n_features, n_outputs):
        super(LSTMModel_much_less_complex, self).__init__()
        # Reduced hidden size and increased dropout rate
        hidden_size = 256
        dropout_rate = 0.3
        
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=128, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.layer_norm1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.layer_norm2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.repeat = n_outputs
        
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.layer_norm3 = nn.LayerNorm(32)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.dense1 = nn.Linear(32, 16)
        self.layer_norm_dense = nn.LayerNorm(16)
        self.dropout_dense = nn.Dropout(dropout_rate)  # Added dropout in dense layer
        self.dense2 = nn.Linear(16, n_features)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.layer_norm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.layer_norm2(x)
        x = self.dropout2(x)
        
        x = x[:, -1, :].unsqueeze(1).repeat(1, self.repeat, 1)
        
        x, _ = self.lstm3(x)
        x = self.layer_norm3(x)
        x = self.dropout3(x)
        
        x = torch.tanh(self.dense1(x))
        x = self.layer_norm_dense(x)
        x = self.dropout_dense(x)  # Dropout in dense layer
        x = self.dense2(x)
        
        return x



class LSTMModel(nn.Module):
    def __init__(self, n_features, n_outputs):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=512, num_layers=1, batch_first=True, dropout=0.2)
        self.layer_norm1 = nn.LayerNorm(512)  # Layer normalization after lstm1
        self.dropout1 = nn.Dropout(0.2)
        
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.2)
        self.layer_norm2 = nn.LayerNorm(512)  # Layer normalization after lstm2
        self.dropout2 = nn.Dropout(0.2)
        
        self.repeat = n_outputs
        
        self.lstm3 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.2)
        self.layer_norm3 = nn.LayerNorm(512)  # Layer normalization after lstm3
        self.dropout3 = nn.Dropout(0.2)
        
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.2)
        self.layer_norm4 = nn.LayerNorm(512)  # Layer normalization after lstm4
        self.dropout4 = nn.Dropout(0.2)
        
        self.dense1 = nn.Linear(512, 256)
        self.layer_norm_dense = nn.LayerNorm(256)  # Layer normalization after dense1
        self.dense2 = nn.Linear(256, n_features)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.layer_norm1(x)  # Apply layer norm
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.layer_norm2(x)  # Apply layer norm
        x = self.dropout2(x)
        
        x = x[:, -1, :].unsqueeze(1).repeat(1, self.repeat, 1)
        
        x, _ = self.lstm3(x)
        x = self.layer_norm3(x)  # Apply layer norm
        x = self.dropout3(x)
        
        x, _ = self.lstm4(x)
        x = self.layer_norm4(x)  # Apply layer norm
        x = self.dropout4(x)
        
        x = torch.tanh(self.dense1(x))
        x = self.layer_norm_dense(x)  # Apply layer norm
        x = self.dense2(x)
        
        return x

# class LSTMModel(nn.Module):
#     def __init__(self, n_features, n_outputs):
#         super(LSTMModel, self).__init__()
#         self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=512, num_layers=1, batch_first=True, dropout=0.2)
#         self.dropout1 = nn.Dropout(0.2)  # Add a dropout layer after LSTM
#         self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.2)
#         self.dropout2 = nn.Dropout(0.2)  # Add a dropout layer after LSTM
#         self.repeat = n_outputs
#         self.lstm3 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.2)
#         self.dropout3 = nn.Dropout(0.2)  # Add a dropout layer after LSTM
#         self.lstm4 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.2)
#         self.dropout4 = nn.Dropout(0.2)  # Add a dropout layer after LSTM
#         self.dense1 = nn.Linear(512, 256)
#         # self.dropout5 = nn.Dropout(0.3)  # Add a dropout layer after LSTM
#         self.dense2 = nn.Linear(256, n_features)

#     def forward(self, x):
#         x, _ = self.lstm1(x)
#         x = self.dropout1(x)
#         x, _ = self.lstm2(x)
#         x = self.dropout2(x)
#         x = x[:, -1, :].unsqueeze(1).repeat(1, self.repeat, 1)
#         x, _ = self.lstm3(x)
#         x = self.dropout3(x)
#         x, _ = self.lstm4(x)
#         x = self.dropout4(x)
#         x = torch.tanh(self.dense1(x))
#         # x = self.dropout5(x)
#         x = self.dense2(x)
#         return x

def generate_lstm_multi_step(X_train, y_train, device = 'cpu'):
    _, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = LSTMModel(n_features, n_outputs)
    model.to(device)  # Move the model to the GPU
    return model
def generate_lstm_multi_step_less_complex(X_train, y_train, device='cpu'):
    _, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = LSTMModel_less_complex(n_features, n_outputs)
    model.to(device)  # Move the model to the GPU
    return model
def train_lstm_multi_step(model, checkpoint_path, X_train, y_train, epochs=100, batch_size=100, validation_split=0.05, patience=10, verbose=0):
    # Creating training and validation datasets
    train_size = int((1 - validation_split) * X_train.size(0))
    train_dataset = TensorDataset(X_train[:train_size], y_train[:train_size])
    val_dataset = TensorDataset(X_train[train_size:], y_train[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1)  # Adjust lambda function as needed

    best_loss = float('inf')
    patience_counter = 0

    history = {'train_loss':[],'val_loss':[]}
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()        
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss}")
            
        if (val_loss < best_loss) and (epoch > 75):
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
        if (epoch+1)%50 ==0:
            torch.save(model.state_dict(), os.path.splitext(checkpoint_path)[0]+f'_epoch_{epoch:05}.pth')
        scheduler.step()
    return history
# Example usage:
# X_train, y_train are assumed to be preprocessed and loaded as torch tensors
# checkpoint_path = "./checkpoint.pth"
# model = generate_lstm_multi_step(X_train, y_train)
# history = train_lstm_multi_step(model, checkpoint_path, X_train, y_train, batch_size=128, epochs=100, verbose=1)
