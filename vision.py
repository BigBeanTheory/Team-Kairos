import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

database = pd.read_csv('HACER_dataset.csv')
# Convert the DataFrame to a NumPy array
database = database.to_numpy()
# Save the NumPy array to a .npy file
np.save('HACER_dataset.npy', database)
# Load the database of videos of emotional imbalances
database = np.load('HACER_dataset.npy', allow_pickle=True)
# Convert the database to a pandas DataFrame
database = pd.DataFrame(database)

# Split the database into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(database, np.arange(database.shape[0]), test_size=0.2)
# Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the model
model = SVC()
model.fit(X_train, y_train)

# Create a function to detect actions on a camera related to a imbalance event
def extract_features(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to a consistent size
    resized = cv2.resize(gray, (224, 224))
    # Calculate the histogram of oriented gradients
    features = cv2.HOGDescriptor((224, 224), (16, 16), (1, 1), 8)
    features = features.compute(resized)
    # Return the HOG features
    return features

def detect_imbalance_event(frame):
    # Extract features from the frame
    features = extract_features(frame)
    # Scale the features
    features = scaler.transform(features)
    # Predict the action
    action = model.predict(features)
    return action

# Detect actions on a camera related to a imbalance event
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Detect the action
    action = detect_imbalance_event(frame)
    # Send a notification to the user if any emotional imbalance is detected
    if action == 'imbalance event':
        print("Imbalance Detected")
    # Display the frame
    cv2.imshow('frame', frame)
    # Wait for a key press
    if cv2.waitKey(1):
        break
# Release the camera
cap.release()
# Destroy all windows
cv2.destroyAllWindows()