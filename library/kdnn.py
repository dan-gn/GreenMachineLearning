from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense

class KDNN_v1():
 
    def kdnn_model():
        model = Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=['accuracy'])
        return model
 
class KDNN_v2():
 
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
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))

        model.add(Dense(1, activation='sigmoid'))
        opt = optimizers.Adam(learning_rate=0.03)
        model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
 
        return model

class KDNN_v3():
 
    def kdnn_model():
        model = Sequential()
 
        # First Layer
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))

        # Second Layer
        model.add(Dense(1536, activation='relu'))
        model.add(Dense(1))

        # Third Layer
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))

        # Fourth Layer
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))
        opt = optimizers.Adam(learning_rate=0.03)
        model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
 
        return model
    
class KDNN_v4():
    
    def kdnn_model():

        def residual_block(model, units):
            model.add(Dense(units, activation='sigmoid'))
            model.add(Dense(units, activation='sigmoid'))
            return model

        # Initialize the Sequential model
        model = Sequential()

        # Build the ResNet with specified architectures
        model.add(Dense(64, activation='sigmoid'))

        model = residual_block(model, 64)
        model.add(Dense(1536, activation='sigmoid'))

        model = residual_block(model, 1536)
        model.add(Dense(64, activation='sigmoid'))

        model = residual_block(model, 64)
        model.add(Dense(64, activation='sigmoid'))

        model = residual_block(model, 64)
        model.add(Dense(256, activation='sigmoid'))

        model = residual_block(model, 256)
        model.add(Dense(128, activation='sigmoid'))

        model = residual_block(model, 128)
        model.add(Dense(64, activation='sigmoid'))

        model = residual_block(model, 64)

        # Output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        opt = optimizers.Adam(learning_rate=0.03)
        model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])

        return model
    
class KDNN_v5():
    
    def kdnn_model():

        def residual_block(model, units):
            model.add(Dense(units, activation='sigmoid'))
            model.add(Dense(units, activation='sigmoid'))
            return model

        # Initialize the Sequential model
        model = Sequential()

        # Build the ResNet with specified architectures
        model.add(Dense(256, activation='sigmoid'))

        model = residual_block(model, 256)
        model.add(Dense(128, activation='sigmoid'))

        model = residual_block(model, 128)
        model.add(Dense(64, activation='sigmoid'))

        model = residual_block(model, 64)

        # Output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        opt = optimizers.Adam(learning_rate=0.03)
        model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])

        return model