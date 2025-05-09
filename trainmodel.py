import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Dataset folder path
DATASET_PATH = "dataset"

# Digit label mapping
DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
label_map = {digit: idx for idx, digit in enumerate(DIGITS)}

print("Setup complete: Libraries imported and label map ready.")
X = []
y = []

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Process each digit class
for digit in DIGITS:
    folder = os.path.join(DATASET_PATH, digit)
    if not os.path.exists(folder):
        continue
    files = os.listdir(folder)[:100]  # limit to 100 files per class
    for file in files:
        path = os.path.join(folder, file)
        try:
            mfcc_features = extract_mfcc(path)
            X.append(mfcc_features)
            y.append(label_map[digit])
        except Exception as e:
            print(f"Error processing {file}: {e}")

X = np.array(X)
y = np.array(y)

print("Feature extraction complete.")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Model trained. Accuracy on test set: {acc * 100:.2f}%")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DIGITS)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Digit Classification")
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DIGITS)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Digit Classification")
plt.show()
# --- Predict a Custom Audio File ---
test_audio_path = "audio.wav"

if os.path.exists(test_audio_path):
    try:
        mfcc = extract_mfcc(test_audio_path)
        mfcc = mfcc.reshape(1, -1)
        prediction = model.predict(mfcc)[0]
        predicted_digit = DIGITS[prediction]
        print(f"Predicted Digit: {predicted_digit}")
    except Exception as e:
        print(f"Error processing the file: {e}")
else:
    print("Test audio file not found. Please check the path.")

from sklearn.metrics import classification_report

# Print classification report
report = classification_report(y_test, y_pred, target_names=DIGITS)
print("Classification Report:\n")
print(report)





