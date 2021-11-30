import pandas as pd
import numpy as np
import tensorflow as tf

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


def preprocess_data_for_LSTM(df,features,past_history, future_target, STEP)
    features = df[features_considered]
    features.index = df['date']
    
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std
    
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)
    print (x_train_multi.shape,
           y_train_multi.shape,
           'Single window of past history : {}'.format(x_train_multi[0].shape),
           'Target usage to predict : {}'.format(y_train_multi[0].shape),
           sep='\n')
    
    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
    return train_data_multi, val_data_multi