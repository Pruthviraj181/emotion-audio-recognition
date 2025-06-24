import joblib
import numpy as np

# Load model
model = joblib.load('models/emotion_xgb_model.pkl')

# Example test input — replace with real test MFCCs
X_test_sample = np.random.rand(1, 40)

# Predict
y_pred = model.predict(X_test_sample)
print("Predicted emotion:", y_pred)


