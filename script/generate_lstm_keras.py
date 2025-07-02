
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, LayerNormalization, LeakyReLU, Input
import pickle
import os


def generate_lstm(X_train, predictors_lst):
    # Learning rate scheduler function

        # else:
        #     return lr * np.exp(-0.1)

    model = Sequential()

    # First LSTM layer with LayerNormalization and recurrent dropout
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]),
                activation='tanh', recurrent_activation='sigmoid',
                return_sequences=True, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LayerNormalization())
    # model.add(Dropout(0.1))

    # Second LSTM layer with residual connection, LayerNormalization, and recurrent dropout
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                return_sequences=True, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LayerNormalization())
    # model.add(Dropout(0.1))

    # Third LSTM layer (final) without returning sequences, adding residual connection
    model.add(LSTM(256, activation='tanh', recurrent_activation='sigmoid',
                return_sequences=False, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LayerNormalization())

    # Dense layer with LeakyReLU activation for flexibility in output
    model.add(Dense(len(predictors_lst)))
    # model.add(LeakyReLU(alpha=0.1))  # LeakyReLU is more flexible for output regression

    # Compile the model using AdamW optimizer and a learning rate scheduler
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)  # AdamW improves generalization
    model.compile(optimizer=optimizer, loss='mse')

    return model


def generate_lstm_per_jose(X_train, predictors_lst):
    # Learning rate scheduler function

        # else:
        #     return lr * np.exp(-0.1)

    model = Sequential()

    # First LSTM layer with LayerNormalization and recurrent dropout
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]),
                activation='tanh', recurrent_activation='sigmoid',
                return_sequences=True, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LayerNormalization())
    # model.add(Dropout(0.1))

    # Second LSTM layer with residual connection, LayerNormalization, and recurrent dropout
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                return_sequences=True, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LayerNormalization())
    # model.add(Dropout(0.1))

    # Third LSTM layer (final) without returning sequences, adding residual connection
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid',
                return_sequences=False, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LayerNormalization())

    # Dense layer with LeakyReLU activation for flexibility in output
    model.add(Dense(len(predictors_lst)))
    # model.add(LeakyReLU(alpha=0.1))  # LeakyReLU is more flexible for output regression

    # Compile the model using AdamW optimizer and a learning rate scheduler
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)  # AdamW improves generalization
    model.compile(optimizer=optimizer, loss='mse')

    return model

def train_lstm(model, checkpoint_path, X_train, y_train):
    try:
        model.load_weights(checkpoint_path) 
        with open(os.path.join(checkpoint_path.split('/')[1],checkpoint_path.split('/')[-1].split('.')[0]), 'rb') as f:
            history = pickle.load(f)
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
                                    verbose=1)

        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)

        history = model.fit(X_train, y_train, epochs=1000, batch_size=512, 
                            validation_split=0.05,
                            verbose=1, callbacks=[checkpoint, lr_scheduler_callback, early_stopping])# ,early_stopping]) 
        
        with open(os.path.join(checkpoint_path.split('/')[1],checkpoint_path.split('/')[-1].split('.')[0]), 'wb') as f:
            pickle.dump(history, f)
    return history
