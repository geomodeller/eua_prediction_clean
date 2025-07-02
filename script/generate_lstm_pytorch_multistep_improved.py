import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# --- 1. Device Configuration ---
# Set the device to GPU if available, otherwise CPU. This is crucial for performance.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Improved LSTM Model Architecture ---
# This revised model uses a more standard and robust Encoder-Decoder architecture.
class LSTMModel(nn.Module):
    def __init__(self, n_features, n_outputs, hidden_size=256, dropout_rate=0.3):
        """
        Initializes the model.
        Args:
            n_features (int): The number of input features.
            n_outputs (int): The number of time steps to predict in the future.
            hidden_size (int): The number of features in the hidden state.
            dropout_rate (float): The dropout rate to use for regularization.
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_outputs = n_outputs

        # ENCODER: Processes the input sequence.
        # A single bidirectional LSTM layer is often a strong baseline.
        # `bidirectional=True` means it processes the sequence forwards and backwards.
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # DECODER: Generates the output sequence.
        # The input to the decoder is the final hidden state of the encoder.
        # The hidden size is doubled because the encoder is bidirectional.
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Fully connected layers to map the LSTM output to the desired prediction shape.
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU() # ReLU is a common and effective activation function.
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_size // 2, n_features)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        # --- Encoder ---
        # The encoder processes the entire input sequence.
        # We only need the final hidden and cell states.
        # encoder_outputs shape: (batch_size, seq_len, hidden_size * 2)
        # hidden shape: (num_layers * 2, batch_size, hidden_size)
        _, (hidden, cell) = self.encoder_lstm(x)

        # Concatenate the hidden states from the forward and backward passes.
        # This creates the context vector that summarizes the input sequence.
        # New hidden shape: (num_layers, batch_size, hidden_size * 2)
        hidden = torch.cat((hidden[0:1], hidden[1:2]), dim=2)
        cell = torch.cat((cell[0:1], cell[1:2]), dim=2)

        # --- Decoder ---
        # The decoder's input is the context vector from the encoder, repeated for each
        # future time step we want to predict.
        # decoder_input shape: (batch_size, n_outputs, hidden_size * 2)
        decoder_input = hidden.permute(1, 0, 2).repeat(1, self.n_outputs, 1)

        # The decoder uses the encoder's final state as its initial state.
        decoder_outputs, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        
        # --- Output Layers ---
        x = self.dropout(self.activation(self.dense1(decoder_outputs)))
        x = self.dense2(x)
        return x

# --- 3. Refined Training Function ---
def train_model(model, checkpoint_path, X_train, y_train, lr=0.001, epochs=1000, batch_size=128, 
                validation_split=0.1, patience=15, verbose=1, device = 'cpu'):
    """
    Trains the LSTM model with best practices like early stopping and a learning rate scheduler.
    """
    # Create training and validation datasets
    val_size = int(validation_split * len(X_train))
    train_size = len(X_train) - val_size
    
    dataset = TensorDataset(X_train, y_train)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer, loss function, and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # ReduceLROnPlateau is more adaptive than a fixed decay schedule.
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            # CRITICAL FIX: Move data to the same device as the model
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            
            # Gradient Clipping: Prevents exploding gradients, crucial for RNNs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # CRITICAL FIX: Move data to the same device as the model
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                total_val_loss += criterion(output, y_batch).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Early stopping and checkpointing logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation loss improved. Saving model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

    # Load the best model state before returning
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history

# --- 4. Example Usage ---
if __name__ == '__main__':
    # Generate some dummy data for demonstration
    # In a real scenario, you would load your preprocessed data here.
    N_SAMPLES = 1000
    INPUT_SEQ_LEN = 50  # Length of input sequence
    OUTPUT_SEQ_LEN = 10 # Length of output sequence (n_outputs)
    N_FEATURES = 13     # Number of features in the data

    # Create dummy tensors
    X = torch.randn(N_SAMPLES, INPUT_SEQ_LEN, N_FEATURES)
    y = torch.randn(N_SAMPLES, OUTPUT_SEQ_LEN, N_FEATURES)

    # Define model and checkpoint path
    checkpoint_path = "./best_lstm_model.pth"
    
    # Instantiate the model and move it to the configured device
    model = LSTMModel(n_features=N_FEATURES, n_outputs=OUTPUT_SEQ_LEN).to(device)
    
    print("\n--- Starting Model Training ---")
    
    # Train the model
    trained_model, history = train_model(
        model,
        checkpoint_path,
        X,
        y,
        epochs=100, # Increased epochs as early stopping will handle it
        batch_size=64, # Common batch size
        patience=15
    )

    print("\n--- Model Training Finished ---")

    # You can now use `trained_model` for predictions or further analysis.
    # For example, to make a prediction:
    trained_model.eval()
    with torch.no_grad():
        sample_input = X[0:1].to(device) # Get one sample and move to device
        prediction = trained_model(sample_input)
        print(f"\nShape of a sample prediction: {prediction.shape}")

