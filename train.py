import os
import numpy as np
import librosa

# Path to your fear audio dataset
dataset_path = 'fear'  # Assuming you're running from "C:\security alert"

# Audio processing constants
SAMPLE_RATE = 22050      # Common standard
DURATION = 3             # In seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MFCC = 13            # Number of MFCCs to extract

# Output lists
features = []
labels = []

# Loop through each .wav file in the dataset
for file_name in os.listdir(dataset_path):
    if file_name.lower().endswith('.wav'):
        file_path = os.path.join(dataset_path, file_name)
        print(f"Processing: {file_path}")
        
        # Load the audio file
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Pad or trim to fixed length
        if len(signal) < SAMPLES_PER_TRACK:
            signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), mode='constant')
        else:
            signal = signal[:SAMPLES_PER_TRACK]
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
        mfcc = mfcc.T  # Transpose to shape (time_steps, n_mfcc)
        
        # Append to dataset
        features.append(mfcc)
        labels.append('fear')  # All are fear sounds for now

print(f"\n✅ Extracted features from {len(features)} audio files.")
import pickle

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Save using pickle
data = {
    "features": features,
    "labels": labels
}

with open("fear_features.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Features and labels saved to fear_features.pkl")
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Define the path to the 'fear' dataset
dataset_path = 'fear'

# Parameters
SAMPLE_RATE = 22050
NUM_MFCC = 13  # Number of MFCCs to extract
SAMPLES_PER_TRACK = 22050  # 1 second of audio at 22,050Hz

# Pick a few files to visualize their MFCCs
file_names = os.listdir(dataset_path)
sample_files = file_names[:5]  # Choose the first 5 files (you can adjust this)

# Visualize MFCC of first few files
plt.figure(figsize=(12, 10))

for i, file_name in enumerate(sample_files):
    if file_name.endswith('.wav'):
        file_path = os.path.join(dataset_path, file_name)
        
        # Load the audio file
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
        
        # Plot MFCC
        plt.subplot(3, 2, i+1)
        librosa.display.specshow(mfcc, x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'MFCC for {file_name}')

plt.tight_layout()
plt.show()
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

# Parameters
SAMPLE_RATE = 22050
NUM_MFCC = 13
SAMPLES_PER_TRACK = 22050  # 1 second of audio at 22,050Hz

# Load preprocessed features
with open("fear_features.pkl", "rb") as f:
    data = pickle.load(f)

X = data['features']
y = data['labels']

# Convert labels to numeric (since only 'fear', it's 1 class)
y_encoded = np.zeros(len(y))  # all 0s for fear class

# Pad/truncate sequences if needed
max_len = 130  # typical length, based on MFCC size
X_padded = np.array([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant') if x.shape[0] < max_len else x[:max_len] for x in X])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Reshape for RNN input
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Build model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(max_len, 13)))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train and save history for visualization
history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_split=0.2)

# Save the model
model.save("fear_model.h5")
print("✅ Model trained and saved as fear_model.h5")

# Visualization: Loss and Accuracy Graphs
# Plot training & validation loss values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Function to preprocess the audio file (same as training preprocessing)
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    # Extracting features (e.g., MFCC)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# Load the trained model
model = load_model('fear_model.h5')

# Function to make prediction
def predict_fear(audio_file):
    # Extract features from the audio file
    features = extract_features(audio_file)
    # Reshape for the model input (add an extra dimension for batch size)
    features = np.reshape(features, (1, 130, 1))  # Adjust if needed to match input shape

    # Make the prediction
    prediction = model.predict(features)

    # If prediction > 0.5, consider it as "fear"
    if prediction > 0.5:
        print("Security Alert: Fear Detected!")
    else:
        print("No Fear Detected.")

# Test the function with a new audio file
audio_file = 'test.wav'  # Provide the path to the fear audio file
predict_fear(audio_file)

