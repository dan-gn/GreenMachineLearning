from keras.models import Sequential
from keras.layers import Dense

class KDNN():
    def kdnn_model():
        model = Sequential()
        model.add(Dense(64, input_shape=(None, 4), activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid')) 
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model
