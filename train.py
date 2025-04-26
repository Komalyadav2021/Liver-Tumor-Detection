import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# --- Paths ---
features_csv_input = "paste your extracted features (.csv)"
model_output_path = os.path.join(os.path.dirname(features_csv_input), "random_forest_model_from_deep_features1.pkl")

# --- Load Feature CSV ---
df = pd.read_csv(features_csv_input)

# --- Prepare Data ---
feature_cols = [col for col in df.columns if col.startswith('feat_')]
X = df[feature_cols].values
y = df['tumor_present'].values

# --- Train/Test Split ---
stratify_param = y if len(np.unique(y)) > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_param
)

# --- Train Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = rf.predict(X_test)
print("\nModel Evaluation:\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Save Model ---
joblib.dump(rf, model_output_path)
print(f"\n Trained model saved to: {model_output_path}")

# --- Test Reload ---
try:
    model_loaded = joblib.load(model_output_path)
    print(" Model reloaded successfully after saving.")
except Exception as e:
    print(f" Failed to reload model: {e}")
