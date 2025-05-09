import os
import librosa
import numpy as np

DATASET_PATH = "dataset"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

features = []
labels = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if os.path.isdir(label_path):
        for filename in os.listdir(label_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(label_path, filename)
                try:
                    mfcc = extract_features(file_path)
                    features.append(mfcc)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

X = np.array(features)
y = np.array(labels)

# Save inside the dataset folder
np.save(os.path.join(DATASET_PATH, "features.npy"), X)
np.save(os.path.join(DATASET_PATH, "labels.npy"), y)


print("✅ Preprocessing done. Features and labels saved to 'dataset/' folder.")
import numpy as np
import matplotlib.pyplot as plt

# Load features and labels
features = np.load('dataset/features.npy')
labels = np.load('dataset/labels.npy')

# Plot the first MFCC vector
plt.figure(figsize=(8, 4))
plt.plot(features[0])
plt.title(f"MFCC Features of Sample 0 — Label: {labels[0]}")
plt.xlabel("MFCC Coefficients")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load features and labels
X = np.load('dataset/features.npy')
y = np.load('dataset/labels.npy')

# Encode the string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Define a simple MLP model
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save the model and label encoder
model.save("speech_model.h5")
np.save("label_classes.npy", le.classes_)

print("✅ Model trained and saved as 'speech_model.h5'")
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model and label encoder
model = load_model("speech_model.h5")
le = LabelEncoder()
le.classes_ = np.load('label_classes.npy', allow_pickle=True)

# Load the features and labels for evaluation (test set)
X_test = np.load('dataset/features.npy')
y_test = np.load('dataset/labels.npy')   # Replace with your test set labels

# Convert labels to numerical format
y_test_encoded = le.transform(y_test)

# Model evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred_classes, target_names=le.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model("speech_model.h5")

# Load the label encoder (this should have been saved during training)
le = LabelEncoder()
le.classes_ = np.load('label_classes.npy', allow_pickle=True)

# Function to extract MFCC features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Function to make predictions on a new audio file
def predict_audio(file_path):
    # Extract features from the audio file
    features = extract_features(file_path)

    # Reshape the feature array to match model input
    features = features.reshape(1, -1)  # Reshape for single prediction

    # Predict the class (audio label) for the input file
    prediction = model.predict(features)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = le.inverse_transform(predicted_class)

    print(f"Predicted label for the audio file '{file_path}': {predicted_label[0]}")

# Example usage – predict the label for a new file
audio_file = "audio.wav"  # Replace with your test audio file path
predict_audio(audio_file)

