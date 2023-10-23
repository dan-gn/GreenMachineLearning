from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

class KDNN():

    def kdnn_model():
        model = Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu')) 
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
