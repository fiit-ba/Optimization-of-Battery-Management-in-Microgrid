import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import tensorflow as tf
from attention import Attention
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model, Model

def build_model_BasicLSTM(n_timesteps, n_features, n_outputs):
    """
    https://www.kaggle.com/nicapotato/keras-timeseries-multi-step-multi-output
    """
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(n_timesteps,
              n_features)))

    model.add(LSTM(16, activation='tanh'))

    model.add(Dense(n_outputs))

    model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae')

    print(model.summary())
    return model


def build_model_BasicLSTM_OneLayer(n_timesteps, n_features, n_outputs):
    """
    https://www.kaggle.com/nicapotato/keras-timeseries-multi-step-multi-output
    """
    model = Sequential()
    model.add(LSTM(32, return_sequences=False,
              input_shape=(n_timesteps, n_features)))

    model.add(Dense(n_outputs))

    model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae')

    print(model.summary())
    return model


def build_model_BasicLSTM_TwoDense(n_timesteps, n_features, n_outputs):
    """
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
    """
    model = Sequential()
    model.add(LSTM(200, input_shape=(n_timesteps, n_features)))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')

    print(model.summary())
    return model


def build_model_EncoderDecoderLSTM(n_timesteps, n_features, n_outputs):
    """
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
    """
    model = Sequential()
    model.add(LSTM(200, input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')

    print(model.summary())
    return model


def build_model_EncoderDecoder_LSTMCNN(n_timesteps, n_features,n_outputs):
    """
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
              input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')

    print(model.summary())
    return model

def build_model_BidirectionalLSTM(n_timesteps, n_features,
        n_outputs,mode):
    """
    https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/
    modes - sum, mul, ave, concat
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(20, return_sequences=True), 
                            input_shape=(n_timesteps,  n_features),merge_mode=mode))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def build_model_parallel_lstm(n_timesteps, n_features,n_outputs,train_data):
    """
    TODO pozriet sa na parallel LSTM
    """
    # from tensorflow.keras.layers import Input, LSTM, concatenate, Dense, Lambda
    # from tensorflow.keras.models import Model
    # input_ = Input(shape=(n_timesteps,  n_features), name='input')

    # middle_split = int(n_timesteps/2)

    # input_1 = train_data[:, :middle_split, :]
    # input_2 = train_data[:, middle_split:, :]
    # print(input_1.shape)
    # lstm1 = LSTM(256, name='lstm1')(input_1)
    # lstm2 = LSTM(256, name='lstm2')(input_2)
    # concat = concatenate([lstm1, lstm2]) 
    # output = Dense(n_outputs, activation='tanh', name='dense')(concat)
    # model = Model(inputs=input_, outputs=output)
    # print(model.summary())
    pass

def build_model_Attention_LSTM(n_timesteps, n_features,n_outputs):
    """
    https://github.com/philipperemy/keras-attention-mechanism
    """
    model_input = Input(shape=(n_timesteps,  n_features))
    x = LSTM(64, return_sequences=True)(model_input)
    x = Attention(32)(x)
    x = Dense(n_outputs)(x)
    model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())
    return model
