#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd
import numpy as np
import random
import tqdm
import glob
import cv2
import os

# other dependencies
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation,BatchNormalization


# SETTING THE DIRECTORY 

# In[2]:


os.getcwd()


# In[3]:


os.chdir("E:\\IISC NV Gaze\\Sample")
#E:\IISC NV Gaze\NV Gaz
#e\nvgaze_real_dataset_vr
os.getcwd()


# In[4]:


#2nd
img = image.load_img("New_train\\combi\\01_new\\000001.jpg")
plt.imshow(img)
cv2.imread("New_train\\combi\\01_new\\000001.jpg").shape


# In[5]:


#SAMPLE DISPLAY CSV DATA 
train = pd.read_csv("New_train\\combi\\01_new.csv")
test = pd.read_csv("New_train\\combi\\12_new.csv")
print(train.head())
print(test.head())


# In[6]:


tf.config.list_physical_devices('GPU')


# LEFT AND RIGHT EYE SEGREGATE DATA FROM ORIGINAL CSV 

# LEFT EYE

# In[ ]:


#Preprocessing - Segregating data based on left and Right eyes
import csv
#Left Eye
with open(f'New_train\\combi\\{num1}.csv', 'r') as infile, open(f'New_train\\combi\\{num1}_new.csv', 'a', newline='') as outfile:
    results = csv.DictReader(infile)
    writer = csv.writer(outfile)

    # write column headers to output file
    writer.writerow(
        ['imagefile', 'eye', 'gaze_x', 'gaze_y']
    )
    for result in results:
        eye = result.get('eye')
        # if value already exists in the list, skip writing it whole row to output file
        if eye == 'L':
            writer.writerow([
            result.get('imagefile'),
            result.get('eye'),
            result.get('gaze_x'),
            result.get('gaze_y')
            ])
        else :
            continue


# RIGHT EYE

# In[7]:


#Right eye 
with open('Train\\01.csv', 'r') as infile, open('eye_R_01.csv', 'a', newline='') as outfile:
    results = csv.DictReader(infile)
    writer = csv.writer(outfile)

    # write column headers to output file
    writer.writerow(
        ['imagefile', 'eye', 'gaze_x', 'gaze_y']
    )
    for result in results:
        eye = result.get('eye')
        # if value already exists in the list, skip writing it whole row to output file
        if eye == 'R':
            writer.writerow([
            result.get('imagefile'),
            eye,
            result.get('gaze_x'),
            result.get('gaze_y')
            ])
        else :
            continue


# CSV FILES SAMPLING STARTS 

# TRAIN :

# In[ ]:


#Preprocessing - Train dataset taking only 2 top images and 2 bottom images 
#for a batch of repeating gaze_x and gaze_y values

import csv

with open('New_train\\14_new\\14.csv', 'r') as infile, open('New_train\\14_new\\unique_first_second_14.csv', 'a', newline='') as outfile:
    # this list will hold unique gaze_x and gaze_y values
    gaze_xs = []
    gaze_ys = []
    results = csv.DictReader(infile)
    writer = csv.writer(outfile)
    
    # write column headers to output file
    writer.writerow(
        ['imagefile', 'eye', 'gaze_x', 'gaze_y']
    )
    
    run_once = 0
    for result in results:
        gaze_x = result.get('gaze_x')
        gaze_y = result.get('gaze_y')
       
        if gaze_x in gaze_xs and gaze_y in gaze_ys:
                if run_once == 0:
                     writer.writerow([
                     result.get('imagefile'),
                     result.get('eye'),
                     gaze_x,
                     gaze_y
                     ])
                     run_once = 1
                continue
        run_once = 0

        writer.writerow([
        result.get('imagefile'),
        result.get('eye'),
        gaze_x,
        gaze_y
        ])
        # add the value to the list to so as to be skipped subsequently
        gaze_xs.append(gaze_x)
        gaze_ys.append(gaze_y)

infile.close()
outfile.close()


# In[13]:


os.getcwd()


# In[10]:


n = input ("Enter number of values in intervals to take in csv files for sampling:")
n = int(n)


# In[11]:


print(n)


# In[230]:


num1 = input ("Enter folder number for extracting left images: ")


# In[231]:


#Next approach - for left eye images as input(half) and selecting only 1/20 of that 
#Preprocessing - Train dataset taking images in batches of 20 - ratio 1/20
df = pd.read_csv(f'Train\\{num1}_new.csv', header=0, skiprows=lambda i: i % n != 0)
# every 100th line = 1% of the lines
# df = pd.read_csv(filename, header=0, skiprows=lambda i: i % n != 0)

#z = df.drop_duplicates(keep='last',subset=['gaze_x', 'gaze_y'])
df.to_csv(f'Left_sampled_10\\{num1}_new.csv',index=False)


# In[70]:


#Preprocessing - Train dataset taking only the bottom image 
#for a batch of repeating gaze_x and gaze_y values

import csv

df = pd.read_csv('New_train\\14_new\\14.csv')
y = df.drop_duplicates(keep='last',subset=['gaze_x', 'gaze_y'])
y.to_csv('New_train\\14_new\\unique_last_14.csv',index=False)    

#Removing the last values in every 300 batch - ***original file altered
# Open details file and get a unique set of links
details_csv = csv.DictReader(open('New_train\\14_new\\unique_last_14.csv','r'))
details = set(i.get('imagefile') for i in details_csv)

# Open master file and only retain the data not in the set
master_csv = csv.DictReader(open('New_train\\14_new\\14.csv','r'))
master = [i for i in master_csv if i.get('imagefile') not in details]

# Overwrite master file with the new results
with open('New_train\\14_new\\14.csv','w') as file:
    writer = csv.DictWriter(file, master[0].keys(), lineterminator='\n')
    writer.writeheader()
    writer.writerows(master)
    
file.close()


# In[71]:


# second last val after deleting the last ones and running keep=last again
df = pd.read_csv('New_train\\14_new\\14.csv')
z = df.drop_duplicates(keep='last',subset=['gaze_x', 'gaze_y'])
z.to_csv('New_train\\14_new\\unique_second_last_14.csv',index=False)


# In[72]:


#merge csv files - merge all to get the 4 - first 2 and last 2
import pandas as pd

a = pd.read_csv("New_train\\14_new\\unique_first_second_14.csv")
b = pd.read_csv("New_train\\14_new\\unique_second_last_14.csv")
c = pd.read_csv("New_train\\14_new\\unique_last_14.csv")

combined_csv = pd.concat([a, b, c])
combined_csv.to_csv("New_train\\14_new\\14_new.csv", index=False)


# In[41]:


df = pd.read_csv("New_train\\14_new\\14_new.csv",delimiter=',')
df.head()


# TEST : 

# In[42]:


#Preprocessing - Train dataset taking only 2 top images and 2 bottom images 
#for a batch of repeating gaze_x and gaze_y values

import csv

with open('New_test\\12_new\\12.csv', 'r') as infile, open('New_test\\12_new\\unique_first_second_12.csv', 'a', newline='') as outfile:
    # this list will hold unique gaze_x and gaze_y values
    gaze_xs = []
    gaze_ys = []
    results = csv.DictReader(infile)
    writer = csv.writer(outfile)
    
    # write column headers to output file
    writer.writerow(
        ['imagefile', 'eye', 'gaze_x', 'gaze_y']
    )
    
    run_once = 0
    for result in results:
        gaze_x = result.get('gaze_x')
        gaze_y = result.get('gaze_y')
        if gaze_x in gaze_xs and gaze_y in gaze_ys:
                if run_once == 0:
                     writer.writerow([
                     result.get('imagefile'),
                     result.get('eye'),
                     gaze_x,
                     gaze_y
                     ])
                     run_once = 1
                continue
        run_once = 0

        writer.writerow([
        result.get('imagefile'),
        result.get('eye'),
        gaze_x,
        gaze_y
        ])
        # add the value to the list to so as to be skipped subsequently
        gaze_xs.append(gaze_x)
        gaze_ys.append(gaze_y)

infile.close()
outfile.close()


# In[44]:


#Preprocessing - Train dataset taking only the bottom image 
#for a batch of repeating gaze_x and gaze_y values

import csv

df = pd.read_csv('New_test\\12_new\\12.csv')
y = df.drop_duplicates(keep='last',subset=['gaze_x', 'gaze_y'])
y.to_csv('New_test\\12_new\\unique_last_12.csv',index=False)    

#Removing the last values in every 300 batch - ***original file altered
# Open details file and get a unique set of links
details_csv = csv.DictReader(open('New_test\\12_new\\unique_last_12.csv','r'))
details = set(i.get('imagefile') for i in details_csv)

# Open master file and only retain the data not in the set
master_csv = csv.DictReader(open('New_test\\12_new\\12.csv','r'))
master = [i for i in master_csv if i.get('imagefile') not in details]

# Overwrite master file with the new results
with open('New_test\\12_new\\12.csv','w') as file:
    writer = csv.DictWriter(file, master[0].keys(), lineterminator='\n')
    writer.writeheader()
    writer.writerows(master)
    
file.close()


# In[45]:


# second last val after deleting the last ones and running keep=last again
df = pd.read_csv('New_test\\12_new\\12.csv')
z = df.drop_duplicates(keep='last',subset=['gaze_x', 'gaze_y'])
z.to_csv('New_test\\12_new\\unique_second_last_12.csv',index=False)


# In[46]:


# merge csv files - merge all to get the 4 - fist 2 and last 2
import pandas as pd

a = pd.read_csv("New_test\\12_new\\unique_first_second_12.csv")
b = pd.read_csv("New_test\\12_new\\unique_second_last_12.csv")
c = pd.read_csv("New_test\\12_new\\unique_last_12.csv")

combined_csv = pd.concat([a, b, c])
combined_csv.to_csv("New_test\\12_new\\12_new.csv", index=False)


# PREPROCESSING ENDS

# PREPARING X TRAIN AND Y TRAIN

# X TRAIN :

# In[6]:


os.chdir("E:\\IISC NV Gaze\\Sample")
#NV Gaze
#nvgaze_real_dataset_vr
os.getcwd()


# In[7]:


num = input ("Enter folder number for test :")


# In[8]:


print("Folders for Validation ")
val1 = input ("Enter folder number 1 for validation :")
val2 = input ("Enter folder number 2 for validation :")
val3 = input ("Enter folder number 3 for validation :")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# PERFORMING TRAINING AND TESTING ON ONE SINGLE FOLDER

# In[9]:


#Training on one sub and testing on one sub splitiing the csv into test and train

import pandas as pd
import numpy as np

df = pd.read_csv(f"New_train\\combi\\{num}_new.csv", sep = ',')

msk = np.random.rand(len(df)) <= 0.75

train = df[msk]
test = df[~msk]
print(f"{num}_new.csv")
print(train)
print(test)


# In[10]:


import pandas as pd

# read DataFrame
data = pd.read_csv(f"New_train\\combi\\{num}_new.csv", sep = ',')

# no of csv files with row size
k = 2
size = 11219 

for i in range(k):
	df = data[size*i:size*(i+1)]
	df.to_csv(f'Customers_{i+1}.csv', index=False)

print(f"{num}_new.csv")
train = pd.read_csv("Customers_1.csv")
print("Train\n",train)

test = pd.read_csv("Customers_2.csv")
print("Tes\n",test)


# In[13]:


print(train.shape[0])
print(test.shape[0])


# In[12]:


print(num)


# In[ ]:


#New code for X_train for single folder
train_image = []

from keras.preprocessing import image
from tqdm import tqdm
import os
for i in tqdm(range(train.shape[0])):
    img = image.load_img(f"New_train\\combi\\{num}\\"+ train['imagefile'][i], target_size=(127, 127,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)

X_train = np.array(train_image)


# In[ ]:


# Y_train initialise 
l = X_train.shape[0]
print(l)
Y_train = np.zeros(shape = [l,2],dtype = np.float32)


# In[ ]:


#preparing Y_train

gaze_x = train['gaze_x'].tolist()
gaze_y = train['gaze_y'].tolist()

for i in range(l):
    Y_train[i,0] = gaze_x[i];
    Y_train[i,1] = gaze_y[i];

print(Y_train.shape)


# In[15]:


#New code for X_test for single folder
test_image = []

from keras.preprocessing import image
from tqdm import tqdm
import os
for i in tqdm(range(test.shape[0])):
    img = image.load_img(f"New_train\\combi\\{num}\\"+ test['imagefile'][i], target_size=(127, 127,3))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)

X_test = np.array(test_image)


# In[16]:


# Y_test initialise
m = X_test.shape[0]
print(m)
Y_test = np.zeros(shape = [m,2],dtype = np.float32)


# In[17]:


#preparing Y_test

gaze_x = test['gaze_x'].tolist()
gaze_y = test['gaze_y'].tolist()

for i in range(m):
    Y_test[i,0] = gaze_x[i];
    Y_test[i,1] = gaze_y[i];

print(Y_test.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


#new approach - for loop for taking both X_train and Y_train together in a loop 
# X_train -- take images in a list - append them and at last - convert it into a numpy array

from keras.preprocessing import image
from tqdm import tqdm
import os

train_image = []
li = []
path = 'New_train\\combi'

prefixes = [f'{num}', f'{val1}', f'{val2}', f'{val3}']

for filename in os.listdir(path):
    if not filename.startswith(tuple(prefixes)):
        if filename.endswith('.csv'): 
            r = filename.split(".")[0]
            print(r)
            train = pd.read_csv(f"New_train\\combi\\{filename}",sep=',')
            li.append(train)
            for i in tqdm(range(train.shape[0])):
                img = image.load_img(f"New_train\\combi\\{r}\\"+ train['imagefile'][i], target_size=(127,127,3))
                img = image.img_to_array(img)
                img = img/255
                train_image.append(img)

X_train = np.array(train_image)
df = pd.concat(li, axis=0, ignore_index=True)
# last 2 columns of dataframe - gaze_x and y
train =  df.iloc[: , -2:]


# In[111]:


print("train shape: ", train.shape)
print("X_train shape: ", X_train.shape)


# In[112]:


print(num)


# Y TRAIN :

# In[113]:


l = X_train.shape[0]
print(l)
Y_train = np.zeros(shape = [l,2],dtype = np.float32)


# In[114]:


#preparing Y_train

gaze_x = train['gaze_x'].tolist()
gaze_y = train['gaze_y'].tolist()

for i in range(l):
    Y_train[i,0] = gaze_x[i];
    Y_train[i,1] = gaze_y[i];

print(Y_train.shape)


# PREPARING X TEST AND Y TEST 

# X TEST :

# In[115]:


path = r"New_train\\combi"
filename = num + "_new.csv"
file_path = r"%s\%s" % (path, filename)
#file_path = os.path.join(path, filename)
test = pd.read_csv(file_path)
print(f"{num}_new.csv")
print(test.head())


# In[116]:


## Loading X_test with images matching automatically csv values
from keras.preprocessing import image
from tqdm import tqdm

test_image = []

for i in tqdm(range(test.shape[0])):
    img = image.load_img(f'New_train\\combi\\{num}_new\\'+test['imagefile'][i], target_size=(127,127,3))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)

X_test = np.array(test_image)
print(X_test.shape)


# Y TEST :

# In[117]:


m = X_test.shape[0]
Y_test = np.zeros(shape = [m,2])


# In[118]:


#preparing Y_test

gaze_x = test['gaze_x'].tolist()
gaze_y = test['gaze_y'].tolist()

for i in range(m):
    Y_test[i,0] = gaze_x[i];
    Y_test[i,1] = gaze_y[i];

print(Y_test.shape)


# PREPARING X VAL AND Y VAL

# X VAL

# In[120]:


#new approach - for loop for taking both X_train and Y_train together in a loop 
# X_train -- take images in a list - append them and at last - convert it into a numpy array

from keras.preprocessing import image
from tqdm import tqdm
import os

val_image = []
li = []
path = 'New_train\\combi'

prefixes = [f'{val1}', f'{val2}', f'{val3}']

for filename in os.listdir(path):
    if filename.startswith(tuple(prefixes)):
        if filename.endswith('.csv'): 
            r = filename.split(".")[0]
            print(r)
            val = pd.read_csv(f"New_train\\combi\\{filename}",sep=',')
            li.append(val)
            for i in tqdm(range(val.shape[0])):
                img = image.load_img(f"New_train\\combi\\{r}\\"+ val['imagefile'][i], target_size=(127,127,3))
                img = image.img_to_array(img)
                img = img/255
                val_image.append(img)

X_val = np.array(val_image)
df = pd.concat(li, axis=0, ignore_index=True)
# last 2 columns of dataframe - gaze_x and y
val =  df.iloc[: , -2:]


# In[121]:


print("val shape: ", val.shape)
print("X_val shape: ", X_val.shape)


# Y_val :

# In[122]:


p = X_val.shape[0]
print(p)
Y_val = np.zeros(shape = [p,2],dtype = np.float32)


# In[123]:


#preparing Y_val

gaze_x = val['gaze_x'].tolist()
gaze_y = val['gaze_y'].tolist()

for i in range(p):
    Y_val[i,0] = gaze_x[i];
    Y_val[i,1] = gaze_y[i];

print(Y_val.shape)


# In[ ]:





# In[5]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[ ]:


print(X_val.shape)
print(Y_val.shape)


# COLAB FUNCTIONALITY EXTRAS - NPY IMPORT DIRECTLY - IN CASE OF LARGE FILES 

# In[33]:


#OPTIONAL - converting into .npy file(binary) for accessing in colab
from numpy import asarray
from numpy import save


# In[34]:


# save('X_train.npy', X_train)


# In[35]:


# save('Y_train.npy', Y_train)


# In[36]:


# save('X_test.npy', X_test)


# In[37]:


# save('Y_test.npy', Y_test)


# CREATING THE MODEL

# In[ ]:


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D

# from tensorflow.keras.layers import Activation,BatchNormalization

# model = Sequential() 

# model.add(Conv2D(24, (9, 9), strides=(2, 2), activation='relu', input_shape=(293,293,1))) # output dim = 284x284x24    
# model.add(Dropout(0.1))

# model.add(Conv2D(36, (7, 7), strides=(2, 2), activation='relu'))                        # channel dimensions = 279x279x36    
# model.add(Dropout(0.1))

# model.add(Conv2D(52, (5, 5), strides=(2, 2), activation='relu'))                     # channel dimensions = 275x275x52
# model.add(Dropout(0.1))

# model.add(Conv2D(80, (5, 5), strides=(2, 2), activation='relu'))                    # channel dimensions = 271x271x80    
# model.add(Dropout(0.1))

# model.add(Conv2D(124, (3, 3), strides=(2, 2), activation='relu'))                    # channel dimensions = 269x269x124
# model.add(Dropout(0.1))

# model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu'))                    # channel dimensions = 267x267x256
# model.add(Dropout(0.1))

# model.add(Conv2D(512, (3, 3), strides=(2, 2), activation='relu'))#10                     # channel dimensions = 265x265x512
# model.add(Dropout(0.1))

# model.add(Flatten())

# model.add(Dense(2, activation='linear'))

# model.summary()

# from tensorflow.keras.optimizers import Adam

# model.compile(loss='mse', optimizer=Adam(lr = 0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,), metrics=['mse'])


# In[ ]:


IMG_SIZE = 127

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])


# In[ ]:


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])


# In[ ]:


# model = tf.keras.Sequential([
#   # Add the preprocessing layers you created earlier.
#   resize_and_rescale,
#   data_augmentation,
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   # Rest of your model.
# ])


# In[18]:


# Creating the model

img_input = layers.Input(shape = (127, 127,3))
x = resize_and_rescale
x = data_augmentation
x = layers.Conv2D(32,3, activation='relu',strides = 2)(img_input)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(48,3, activation='relu',strides = 2)(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(72,3, activation='relu',strides = 2)(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(108,3, activation='relu',strides = 2)(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(162,3, activation='relu',strides = 2)(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(243,3, activation='relu',strides = 2)(x)
x = layers.Dropout(0.1)(x)
x = layers.Flatten()(x)
output = Dense(2, activation='linear')(x)
model = Model(img_input,output)
model.summary()


# In[19]:


model.compile(loss='mean_squared_error',
              optimizer = Adam(lr = 0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08),
              metrics=['mse']); 


# FITTING THE MODEL

# In[40]:


cnt = 0


# In[41]:


c = 0


# In[42]:


print(num)


# In[49]:


# filepath=f"New_train\\Trained models\\OneRealSub_Model_Best_{num}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

#model.fit(X_train,Y_train,epochs=30,verbose=1, callbacks=callbacks_list, validation_data = (X_test,Y_test))


model.fit(X_train,Y_train,epochs=30,verbose=1, validation_data = (X_test,Y_test))


# In[55]:


Y_pred = model.predict(X_test)


# In[56]:


from sklearn import preprocessing
import math


# In[57]:


def convert_to_unit_vector(angles):
    x = -np.cos(angles[:, 0]) * np.sin(angles[:, 1])
    y = -np.sin(angles[:, 0])
    z = -np.cos(angles[:, 1]) * np.cos(angles[:, 0])
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return [x, y, z]


# In[58]:


def compute_angle_error(Y_pred, Y_test):
    pred_x, pred_y, pred_z = convert_to_unit_vector(Y_pred)
    label_x, label_y, label_z = convert_to_unit_vector(Y_test)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    result = np.where(angles > 1)
    if(result != []):
        angles[result] = 1
    result = np.where(angles < -1)
    if(result != []):
        angles[result] = -1
    return np.arccos(angles) * 180 / np.pi


# In[59]:


print(np.mean(compute_angle_error(Y_pred, Y_test)))


# In[ ]:


18.059011696523985
(9746, 127, 127, 3)
(9746, 2)
(1485, 127, 127, 3)
(1485, 2)
(1961, 127, 127, 3)
(1961, 2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[169]:


# cnt +=1
# print(cnt)
# print("epochs:",cnt*50)


# In[33]:


#checking the model performance on train
Y_pred_train = model.predict(X_train)
print(Y_pred_train)
print("\n\n",Y_train)


# In[36]:


#checking the model performance on test
Y_pred = model.predict(X_test)
print(Y_pred)
print("\n\n",Y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[150]:


# Y_test(just to reset the values again - undo adding gaze_z)

m = X_test.shape[0]
Y_test = np.zeros(shape = [m,2])

gaze_x = test['gaze_x'].tolist()
gaze_y = test['gaze_y'].tolist()

for i in range(m):
    Y_test[i,0] = gaze_x[i];
    Y_test[i,1] = gaze_y[i];

print(Y_test.shape)


# In[40]:


## adding 3rd column with Gaze_z=0
print("Original Y_pred with Gaze_x and Gaze_y only\n", Y_pred.shape)
n = len(Y_pred[:,0])
abc = np.zeros((n,1))
Y_pred = np.append(Y_pred, abc, axis=1)
print("New Y_pred with Gaze_z added\n", Y_pred.shape)
print(Y_pred)

print("Original Y_test with Gaze_x and Gaze_y only\n", Y_test.shape)
n = len(Y_test[:,0])
abc = np.zeros((n,1))
Y_test = np.append(Y_test, abc, axis=1)
print("New Y_test with Gaze_z added\n", Y_test.shape)
print(Y_test)


# In[30]:


## Normalisation 
from sklearn import preprocessing
Y_pred = preprocessing.normalize(Y_pred,axis=0)
print("Normalised Y_pred\n", Y_pred)
Y_test = preprocessing.normalize(Y_test,axis=0)
print("\nNormalised Y_test\n", Y_test)


# In[151]:


print("Y_pred", Y_pred.shape)
print(Y_pred)

print("\n\n Y_test", Y_test.shape)
print(Y_test)


# In[160]:


acc = 0
accuracy = []
n = len(Y_pred[:,0])
for i in range(0,n):
    x = Y_pred[i,0]
    y = Y_pred[i,1]
    #z = Y_pred[i,2]
    #norm_test = math.sqrt(x*x + y*y + z*z)
    norm_test = math.sqrt(x*x + y*y)

    x1 = Y_test[i,0]
    y1 = Y_test[i,1]
    #z1 = Y_test[i,2]
    #norm_pred = math.sqrt(x1*x1 + y1*y1 + z1*z1)
    norm_pred = math.sqrt(x1*x1 + y1*y1)

    #angle = (x*x1 + y*y1 + z*z1) / (norm_pred*norm_test)
    angle = (x*x1 + y*y1) / (norm_pred*norm_test)
    acc = (math.acos(angle)*180)/math.pi
    accuracy.append(acc)
    print(acc)


# In[159]:


print(np.mean(accuracy))


# In[39]:


print(num)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[158]:


#older method
from sklearn.metrics import r2_score
r2_score(Y_test, Y_pred)


# In[421]:


print(num)


# In[422]:


model.save(f'New_train\\Trained models\\model_{num}.h5')


# In[161]:


# 01 : 0.7491996433616275
# 02 : 0.7815731003527686
# 03 : 0.8667820096522285
# 04 : 0.6361859914808377
# 05 : 0.8473087323729944
# 06 : 0.7947980705011899
# 07 : 0.6584988471516254
# 08 : 0.6730434829614429
# 09 : 0.5310961116102058
# 10 : 0.9364082329157989
# 11 : 0.8947433263024386
# 12 : 0.0
# 13 : 0.9014300127773702
# 14 : 0.9329610298878344


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[162]:


# #new X_test for checking 12th folder

# path = r"New_train\\combi"
# filename = num + "_sample.csv"
# file_path = r"%s\%s" % (path, filename)
# test = pd.read_csv(file_path)
# print(f"{num}_new.csv")
# print(test.head())


# ## Loading X_test with images matching automatically csv values
# from keras.preprocessing import image
# from tqdm import tqdm

# test_image = []

# for i in tqdm(range(test.shape[0])):
#     img = image.load_img(f'New_train\\combi\\{num}_new\\'+test['imagefile'][i], target_size=(127,127,3))
#     img = image.img_to_array(img)
#     img = img/255
#     test_image.append(img)

# X_test = np.array(test_image)
# print(X_test.shape)


# In[163]:


# #new Y_test for checking 12th folder

# m = X_test.shape[0]
# Y_test = np.zeros(shape = [m,2])

# gaze_x = test['gaze_x'].tolist()
# gaze_y = test['gaze_y'].tolist()

# for i in range(m):
#     Y_test[i,0] = gaze_x[i];
#     Y_test[i,1] = gaze_y[i];

# print(Y_test.shape)


# In[164]:


# # Angular accuracy Working code 2
# ac=0
# ab = np.sum(np.multiply(Y_test, Y_pred), axis=1)
# norm_test = np.linalg.norm(Y_test, axis=1)
# norm_pred = np.linalg.norm(Y_pred, axis=1)
# similarity = np.divide(ab, np.multiply(norm_test, norm_pred))
# ac = (np.arccos(similarity)*180)/math.pi
# print(ac) 


# In[165]:


## accuracy working code 3
# print(np.linalg.norm(Y_test))
# print(np.linalg.norm(Y_pred))
# ab = np.sum((np.multiply(Y_test, Y_pred)),axis=1)

# ang = np.degrees(np.arccos(np.sum((np.multiply(Y_test, Y_pred)),axis=1)/ (np.linalg.norm(Y_test) * np.linalg.norm(Y_pred))))
# print(ang)


# In[166]:


# #angle calculation phi and theta working code 
# import math
# n = len(Y_pred[:,0])
# abc = np.zeros((n,1))
# print(np.append(Y_pred, abc, axis=1).shape)
# Y_pred = np.append(Y_pred, abc, axis=1)
# print(Y_pred)
# b = np.linalg.norm(Y_pred)
# print(b)
# Y_pred = (Y_pred/b)
# x = Y_pred[:,0]
# y = Y_pred[:,1]
# z = Y_pred[:,2]
# theta = []
# phi = []
# for i in range(0,n):
#     theta.append(math.asin(-y[i]))
#     phi.append(math.atan2(-x[i],-z[i]))
# print(len(phi))
# print(len(theta))


# In[167]:


# #sample display angle in degrees
# theta_ang1 = math.asin(y1[0])*(180 / math.pi)
# phi_ang1 = math.atan2(x1[0],z1[0])*(180 / math.pi)
# print(theta_ang1)
# print(phi_ang1)


# In[168]:


##GROUND TRUTH VECTOR CONVERSION TO ANGLE THETA AND PHI
##Y TEST

# print("Original Y_test with Gaze_x and Gaze_y only\n", Y_test.shape)
# n = len(Y_test[:,0])
# abc1 = np.zeros((n,1))
# Y_test = np.append(Y_test, abc1, axis=1)
# print("New Y_test with Gaze_z added\n", Y_test.shape)
# print(Y_test)


# In[169]:


# x1 = Y_test[:,0]
# y1 = Y_test[:,1]
# z1 = Y_test[:,2]
# print(len(y))


# In[170]:


# p_theta = math.cos(mean(theta))
# p_phi = math.cos(mean(phi))

# print(p_theta)
# print(p_phi)


# In[171]:


# from statistics import mean
# print(len(phi))
# print(mean(theta))
# print(mean(phi))


# In[172]:


# g_theta = math.cos(mean(theta1))
# g_phi = math.cos(mean(phi1))

# print(g_theta)
# print(g_phi)


# In[173]:


# #sample display angle in degrees
# theta_ang = math.asin(y[0])*(180 / math.pi)
# phi_ang = math.atan2(x[0],z[0])*(180 / math.pi)
# print(theta_ang)
# print(phi_ang)


# In[174]:


# val1 = Y_pred[:,0][0]
# val2 = Y_pred[:,1][0]

