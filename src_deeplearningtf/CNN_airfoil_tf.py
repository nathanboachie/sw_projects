


#%%
#Import all the necessary libraries to for convolutional neural network
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random

#%%
#Code here will read a CSV containing file paths along with other data, later discarded
#Create a function to read a jpg image, then using the filenames search through folders in a directory and load images existing 
#Then add images to the dataframe and create a train test set

import cv2
df = pd.read_csv('airfoils_Re_03.csv')
#Image loading function
def image_loader(filename):
    image = cv2.imread(filename,0)
    return image
filenames = df['fpath']
images = []
datadir = os.getcwd()
for filename in filenames:
    file_path = os.path.join(datadir,filename)
    if os.path.exists(file_path):
        image = image_loader(file_path)
        images.append(image)
    
df['Image'] = images
df.reset_index(drop=True, inplace=True)

my_df_train = df.sample(frac = 0.8, random_state = 0)
my_df_test = df.drop(my_df_train.index)

#%%

# Create a dataset using flow_from_dataframe, with no real image augmentation, shuffling and class mode on sparse as there are multiple classes
# Create a train and validation generator to generate a train and validation set from a training set of data, later used for training  
def create_dataset(my_df):
    ###
    my_df['AoA'] = my_df['AoA'].astype(str)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2) 
    train_generator = datagen.flow_from_dataframe(dataframe=my_df, x_col = 'fpath', y_col = 'AoA', target_size = (256,256), color_mode = 'grayscale', class_mode = 'sparse', batch_size = 32, shuffle = True, subset = 'training')
    valid_generator = datagen.flow_from_dataframe(dataframe=my_df, x_col = 'fpath', y_col = 'AoA', target_size = (256,256), color_mode = 'grayscale', class_mode = 'sparse', batch_size = 32, shuffle = True, subset = 'validation')  
    ###
    return train_generator, valid_generator
train_generator, valid_generator = create_dataset(my_df_train)
print(my_df_train.shape)
print(my_df_test.shape)
#%%

#Build a model with 16 3x3 filters in the first conv layer, max pooling with 2x2, another conv layer with double the filters, flattening layer, 2 fully connected layers of 64 nodes and a classification layer
#Using cross entropy loss and Adam momentum optimiser build a model, also with relu activation function --> works well with pictures as theyre between 0 and 1 
###
def build_model(im_width, im_height):
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3,3), activation = 'relu', input_shape = (im_width, im_height, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(8, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
       
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(7, activation = 'softmax')
        ])
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4),
                  metrics = ['accuracy'])
    return model
    
im_width  = train_generator.next()[0].shape[1]
im_height = train_generator.next()[0].shape[2]
model = build_model(im_width, im_height)
model.summary()

#%%

#Train a model with 10 epochs, which feels like overkill for simple data
def train_model(train_generator, valid_generator):
    start_time = time.time()
    ###
    history = model.fit(train_generator, epochs = 10, validation_data = valid_generator,verbose = 1)
    
    ###
    end_time = time.time()
    runtime = end_time - start_time
    
    return runtime, history, model
training_time, history, model = train_model(train_generator, valid_generator)
print('Training time: ', round(training_time/60,2), 'mins.')
