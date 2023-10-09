from keras.models import Sequential

model = Sequential()

model.add(Dense(64, input_shape=(None, 1), activation='sigmoid'))

model.add(Dense(64, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

history = model.fit(X, y, epochs=20, batch_size=1)