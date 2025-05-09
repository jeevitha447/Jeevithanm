import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model (adjust path as needed)
model = load_model('fear_model.h5')

# Function to extract features from an audio file
def extract_features(audio_file):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # n_mfcc set to 13 (or any other value that matches the model)

        # We need to reshape the data into a sequence for LSTM
        mfccs = np.transpose(mfccs)  # Transpose to match the expected input shape (time_steps, features)

        print(f"Extracted features shape: {mfccs.shape}")  # For debugging
        
        return mfccs
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

# Function to predict whether the audio contains fear or not
def predict_fear(audio_file):
    features = extract_features(audio_file)
    if features is not None:
        try:
            # Reshape to (1, time_steps, features) for the LSTM input
            features = np.reshape(features, (1, features.shape[0], features.shape[1]))  # Adjust shape as needed
            
            # Make a prediction
            prediction = model.predict(features)
            
            # Assuming a binary classification model
            if prediction > 0.5:  # If prediction is greater than 0.5, consider it "fear"
                print(" NO !Fear detected! Triggering security alert.")
            else:
                print(" fear detected.")
        except Exception as e:
            print(f"Error in prediction: {e}")
    else:
        print("No features extracted. Cannot make prediction.")

# Example usage (call this function with an audio file)
if __name__ == "__main__":
    # Provide the path to your audio file for testing
    audio_file = 'test.wav'  # Replace with the path of your test audio file
    predict_fear(audio_file)
