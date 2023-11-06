from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense

class KDNN():

    def kdnn_model():
        model = Sequential()

        # First Layer
        model.add(Dense(64, activation='sigmoid'))

        # Second Layer
        model.add(Dense(1536, activation='sigmoid'))

        # Third Layer
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))

        # Fourth Layer
        # model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='sigmoid')) 
        model.add(Dense(64, activation='sigmoid'))

        model.add(Dense(1, activation='sigmoid'))
        opt = optimizers.Adam(learning_rate=0.03)
        model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])

        return model