import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.model_selection import train_test_split

with open('cnn_data.pickle', 'rb') as f:
    data = pickle.load(f)

X = np.array(data['outputs'])  # (samples, 90, 126)
y = np.array(data['inputs'])  # (samples, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(90, 126)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=4)

model.save('cnn_lstm_model.h5')