import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.sequence import pad_sequences

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Pad sequences
max_length = max(len(seq) for seq in data_dict['data'])
data_padded = pad_sequences(data_dict['data'], maxlen=max_length, dtype='float32', padding='post', value=0.0)
print(f"Padded data shape: {data_padded.shape}")

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(data_dict['labels'])
labels_onehot = to_categorical(labels_encoded)

# Split data
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels_onehot, test_size=0.2, stratify=data_dict['labels'])

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
class_weights = dict(zip(np.unique(labels_encoded), class_weights))

# Model
model = Sequential()
model.add(Input(shape=(max_length, 126)))
model.add(LSTM(256))  # Increased from 128
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),  # Increased patience
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)]

model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), class_weight=class_weights, callbacks=callbacks)

model.save('model.h5')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)