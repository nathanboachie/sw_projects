
# # Hot Wire RNN
#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

#%%

#Import data, U for velocity and sample rate as frequency
import h5py
with h5py.File('../resource/asnlib/publicdata/hot_wire_data.mat', 'r') as f:
    U = list(f['U'])
    freq = list(f['samp_rate'])

#%%
def prep_dataset(U, freq, split_ratio):
    ###
    #Create time array, split it into 10000 first steps and store within arrays, then create train valid split
    time_1 = (np.arange(0,len(U[1]))*(1/freq[0]))[0:10000]
    x = U[1][0:10000]
    n = len(x)
    x_train = x[0:int(n*split_ratio)]
    x_valid = x[int(n*split_ratio):]
    time_train = time_1[0:int(n*split_ratio)]
    time_valid = time_1[int(n*split_ratio):]
    ###
    return time_train, x_train, time_valid, x_valid
split_ratio = 0.8
time_train, x_train, time_valid, x_valid = prep_dataset(U,freq,split_ratio)
#%%
#Create windowed dataset with window size, batch size and shuffle buffer for training and validation
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    ###
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1,shift = 1, stride = 1, drop_remainder = True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    ###
    return dataset
window_size = 60
batch_size = 32
shuffle_buffer_size = 1000
train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
test_dataset = windowed_dataset(x_valid, window_size, batch_size, 1)
#%%
#Build a good infrastructure for LSTM model, includes lambda functions for creating a 3D tensor and scaling the output to be in the same range as the input
def build_model():
    ###
    model  = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis = -1), input_shape = [None]),
    tf.keras.layers.LSTM(100, return_sequences = False),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*10.0)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.compile(optimizer=optimizer,loss = tf.keras.losses.Huber(),metrics = ['mape'])
    ###
    return model
model = build_model()
model.summary()
#%%
#Train model using optimal learning rate and no. of epochs
def train_model(train_dataset):
    start_time = time.time()
    ###
    history = model.fit(train_dataset, epochs = 30, validation_data = test_dataset)
    ###
    end_time = time.time()
    runtime = end_time-start_time
    return runtime, history, model#train using optimal learning rate and no. of epochs
runtime, history, model = train_model(train_dataset)
#%%
# Visualize your loss using this cell
epochs = range(len(history.history['loss']))
plt.plot  ( epochs, history.history['loss'], label = 'Training')
# plt.plot  ( epochs, history.history['val_loss'], label = 'Validation')
plt.title ('Training and validation loss')
plt.xlabel('epochs')
plt.legend()
plt.ylim([0,0.01])

#%%
#Model prediction and visualisation
forecast = model.predict(test_dataset)
# Plot your results alongside ground truth
plt.figure(figsize=(10, 6))
plt.plot(time_valid[window_size:],x_valid[window_size:], label='data')
plt.plot(time_valid[window_size:],forecast, label='RNN prediction on validation data')
plt.xlabel('time step')
plt.ylabel('label')
plt.title('RNN prediction')
plt.legend()
plt.ylim([9,9.5])
plt.xlim([0.145,0.15])
