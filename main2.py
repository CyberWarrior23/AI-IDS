import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

# ----------------------------- #
# Configuration
# ----------------------------- #
RAW_DATA_FOLDER = os.path.join("data", "archive (1)")
TRAIN_PATH = os.path.join("data", "CIC_IDS_2017_train.csv")
TEST_PATH = os.path.join("data", "CIC_IDS_2017_test.csv")
MODEL_PATH = os.path.join("models", "ids_model.pkl")


# ----------------------------- #
# Step 1: Load and Merge Data
# ----------------------------- #
def load_and_merge_csvs(RAW_DATA_FOLDER):
    csv_files = [os.path.join(RAW_DATA_FOLDER, file) for file in os.listdir(RAW_DATA_FOLDER) if file.endswith(".csv")]
    df_list = [pd.read_csv(file, low_memory=False) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] Loaded {len(csv_files)} CSV files.")
    return df

# ----------------------------- #
# Step 2: Preprocess Data
# ----------------------------- #
def preprocess_data(df):
    print("[DEBUG] Available columns BEFORE cleaning:", df.columns.tolist())

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    print("[DEBUG] Available columns AFTER cleaning:", df.columns.tolist())

    # Drop unnecessary columns
    drop_cols = ["Flow ID", "Timestamp", "Fwd Header Length.1"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Check if "Label" exists
    if "Label" not in df.columns:
        raise ValueError("❌ Column 'Label' not found in the dataset. Please check your input CSV files.")

    # Encode labels
    df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)

    return df


# ----------------------------- #
# Step 3: Split Dataset
# ----------------------------- #
def split_and_save_dataset(df, train_path, test_path):
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=42)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[INFO] Train & Test datasets saved to /data.")

# ----------------------------- #
# Step 4: Feature Scaling
# ----------------------------- #
def feature_engineering(df):
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Replace infinite values with NaNs
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("[DEBUG] Total NaNs before cleaning:", X.isna().sum().sum())

    # Drop rows with any NaNs
    X.dropna(inplace=True)
    y = y[X.index]  # realign y

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# ----------------------------- #
# Step 5: Train Model
# ----------------------------- #
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"[INFO] Model trained with accuracy: {acc:.4f}")
    
    return clf

# ----------------------------- #
# Step 6: Save Model
# ----------------------------- #
def save_model(model, model_path, scaler=None, feature_names=None):
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names
    }, model_path)
    print(f"[INFO] Model, Scaler & Feature Names saved at {model_path}")



# ----------------------------- #
# Main Execution
# ----------------------------- #
if __name__ == "__main__":
    print("[START] IDS Pipeline Running...")

    df_raw = load_and_merge_csvs(RAW_DATA_FOLDER)
    df_clean = preprocess_data(df_raw)
    split_and_save_dataset(df_clean, TRAIN_PATH, TEST_PATH)

    X_scaled, y, scaler = feature_engineering(df_clean)
    model = train_model(X_scaled, y)
    save_model(model, "Models/ids_model.pkl", scaler=scaler, feature_names=list(df_clean.drop("Label", axis=1).columns))



    print("[SUCCESS] All steps completed successfully.")
