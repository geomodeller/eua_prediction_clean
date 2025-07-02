import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUModel(nn.Module):
    def __init__(self, n_features, n_outputs):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size=n_features, hidden_size=512, num_layers=1, batch_first=True, dropout=0.5)
        self.dropout1 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.5)
        self.dropout2 = nn.Dropout(0.2)
        self.repeat = n_outputs
        self.gru3 = nn.GRU(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.5)
        self.dropout3 = nn.Dropout(0.2)
        self.gru4 = nn.GRU(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.5)
        self.dropout4 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(256, n_features)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x = x[:, -1, :].unsqueeze(1).repeat(1, self.repeat, 1)
        x, _ = self.gru3(x)
        x = self.dropout3(x)
        x, _ = self.gru4(x)
        x = self.dropout4(x)
        x = torch.tanh(self.dense1(x))
        x = self.dropout5(x)
        x = self.dense2(x)
        return x

def generate_gru_multi_step(X_train, y_train):
    _, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = GRUModel(n_features, n_outputs)
    model.to(device)  # Move the model to the GPU
    return model

def train_gru_multi_step(model, checkpoint_path, X_train, y_train, epochs=100, batch_size=100, validation_split=0.05, patience=10, verbose=0):
    # Creating training and validation datasets
    train_size = int((1 - validation_split) * X_train.size(0))
    train_dataset = TensorDataset(X_train[:train_size], y_train[:train_size])
    val_dataset = TensorDataset(X_train[train_size:], y_train[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.995**epoch)  # Adjust lambda function as needed

    best_loss = float('inf')
    patience_counter = 0

    flag = 0
    history = {'train_loss':[], 'val_loss':[]}
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch.to(device))
            loss = criterion(output, y_batch.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch.to(device))
                val_loss += criterion(output, y_batch.to(device)).item()        
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss}")
            
        if (val_loss < best_loss) and (epoch > 0):
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f'flag == {flag}')
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss}")
            flag += 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
        scheduler.step()

    return history
