import keras
from keras.models import Sequential
from keras.layers import *
from keras.layers.wrappers import *
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import CSVLogger, EarlyStopping
import tensorflow as tf


def build_model_BasicLSTM(pastHistory,number_of_features_x, number_of_predictions_y ):
    model = Sequential()
    model.add(
            LSTM(
                32,
                return_sequences=True,
                input_shape=(pastHistory, number_of_features_x)
            )
    )
    model.add(
        LSTM(
            16, 
            activation='relu'
        )
    )
    model.add(Dense(number_of_predictions_y))

    model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae')
    
    print(model.summary())
    return model