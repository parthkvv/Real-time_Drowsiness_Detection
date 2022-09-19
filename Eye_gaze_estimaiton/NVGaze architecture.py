from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Activation,BatchNormalization

model = Sequential() 

model.add(Conv2D(24, (9, 9), strides=(2, 2), activation='relu', input_shape=(293,293,1))) # output dim = 284x284x24    
model.add(Dropout(0.1))

model.add(Conv2D(36, (7, 7), strides=(2, 2), activation='relu'))                        # channel dimensions = 279x279x36    
model.add(Dropout(0.1))

model.add(Conv2D(52, (5, 5), strides=(2, 2), activation='relu'))                     # channel dimensions = 275x275x52
model.add(Dropout(0.1))

model.add(Conv2D(80, (5, 5), strides=(2, 2), activation='relu'))                    # channel dimensions = 271x271x80    
model.add(Dropout(0.1))

model.add(Conv2D(124, (3, 3), strides=(2, 2), activation='relu'))                    # channel dimensions = 269x269x124
model.add(Dropout(0.1))

model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu'))                    # channel dimensions = 267x267x256
model.add(Dropout(0.1))

model.add(Conv2D(512, (3, 3), strides=(2, 2), activation='relu'))#10                     # channel dimensions = 265x265x512
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(2, activation='linear'))

model.summary()

from tensorflow.keras.optimizers import Adam

model.compile(loss='mse', optimizer=Adam(lr = 0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,), metrics=['mse'])