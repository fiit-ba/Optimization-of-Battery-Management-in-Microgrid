import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
    
def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction,STEP):
    plt.figure(figsize=(18, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()
    

def validation_plots(history, number_of_plots, val_data, model,STEP):
    plot_train_history(history, 'Multi-Step Training and validation loss')
    
    for x, y in val_data.take(number_of_plots):
        multi_step_plot(x[0], y[0], model.predict(x)[0],STEP)