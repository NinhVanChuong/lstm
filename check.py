import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping



actions = np.array(['dat','cat','dua','nhat'])

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join('Hand_Data', action))).astype(int):
        window = []
        for frame_num in range(8, 15):
            res = np.load(os.path.join('Hand_Data', action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=50,
    restore_best_weights=True,
)
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(7,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)
model.fit(X_train, y_train, epochs=2000, validation_data=(X_test, y_test),
    callbacks=[early_stopping])
model.summary()
model.save('new_model_23.h5')