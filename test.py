import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Paths ---
features_csv_path = "paste your test feature extraction (.csv)"
model_path = "paste your trained output file (.pkl)"
output_predictions_csv = r"C:\Users\KOMAL YADAV\Downloads\LITS17\Coverted_image\test\output\predicted_results.csv"

# --- Load extracted test features ---
df = pd.read_csv(features_csv_path)

# --- Prepare test data ---
X_test = df[[col for col in df.columns if col.startswith('feat_')]].values
y_test = df['tumor_present'].values

# --- Load trained model ---
try:
    model = joblib.load(model_path)
    print(" Model loaded successfully.")
except Exception as e:
    print(f" Error loading model: {e}")
    exit()

# --- Make predictions ---
y_pred = model.predict(X_test)

# --- Evaluation ---
print("\n Model Evaluation on Test Set:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Save full predictions to CSV ---
output_df = df.copy()
output_df['Predicted'] = y_pred
output_df['Actual'] = y_test
output_df.to_csv(output_predictions_csv, index=False)
print(f"\n All predictions saved to: {output_predictions_csv}")
