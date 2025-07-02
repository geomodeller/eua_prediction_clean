
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, LayerNormalization
from tensorflow.keras.layers import LeakyReLU, Input, RepeatVector,TimeDistributed
import pickle
import os

def generate_lstm_multi_step(X_train, y_train):
    
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    model = Sequential()
    # First LSTM layer with LayerNormalization and recurrent dropout
    model.add(LSTM(256, input_shape=(n_timesteps, n_features),
                return_sequences=True,
                activation='tanh', recurrent_activation='sigmoid',
                recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LSTM(256,
                activation='tanh', recurrent_activation='sigmoid',
                recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                return_sequences=True, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                return_sequences=True, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))

    model.add(TimeDistributed(Dense(64, activation='tanh')))
    model.add(TimeDistributed(Dense(n_features)))
    
    # Compile the model using AdamW optimizer and a learning rate scheduler
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)  # AdamW improves generalization
    model.compile(optimizer=optimizer, loss='mse')

    return model
def train_lstm_multi_step(model, checkpoint_path, X_train, y_train,
                          epochs=100, batch_size = 100,
                          validation_split=0.05,
                          patience = 100,
                          verbose = 0):
    try:
        model.load_weights(checkpoint_path) 
        # with open(os.path.join(checkpoint_path.split('/')[1],checkpoint_path.split('/')[-1].split('.')[0]), 'rb') as f:
        #     history = pickle.load(f)
    except:
        def lr_scheduler(epoch, lr):
        # if epoch < 10:
            return lr
        # Learning rate scheduler callback
        lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

        checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                    monitor='val_loss', 
                                    save_best_only=True,
                                    mode='min',  
                                    verbose=verbose)

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_split=validation_split,
                            verbose=verbose, callbacks=[checkpoint, lr_scheduler_callback, early_stopping])# ,early_stopping]) 
        
        with open(os.path.join(checkpoint_path.split('/')[1],checkpoint_path.split('/')[-1].split('.')[0]), 'wb') as f:
            pickle.dump(history, f)
    return history
