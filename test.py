import numpy as np
from aeon.datasets import load_classification
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
X, y = load_classification("AbnormalHeartbeat")

# Encode the labels if they are not already integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded)

# Reshape and trim the data as before
trim_size = 65 * 4
X_train_trimmed = X_train[:, :, :trim_size]
X_test_trimmed = X_test[:, :, :trim_size]
X_train_reshaped = X_train_trimmed.reshape(-1, 1, 65, 4)
X_test_reshaped = X_test_trimmed.reshape(-1, 1, 65, 4)

# Save the datasets in pickle format
data = {
    'X_train': X_train_reshaped,
    'y_train': y_train,
    'X_test': X_test_reshaped,
    'y_test': y_test
}

with open('Dataset/abnormal_heartbeat.pkl', 'wb') as file:
    pickle.dump(data, file)
