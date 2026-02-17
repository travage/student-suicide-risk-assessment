import pandas as pd
from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    # Set path to original data here
    original_data = pd.read_csv("../data/raw/Final_SP_dataSet.csv", keep_default_na=False)
    # Preprocess data
    df = preprocess_data(original_data)

    # Save cleaned data to new CSV if desired
    # df.to_csv('../data/processed/cleaned_data.csv', index=False)

    # Set features (X) and target (y)
    TARGET = 'suicide_attempt'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Training/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        test_size=0.2,
        stratify=y)

    # Train model
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    rf.fit(X_train, y_train)

    # Save model
    joblib.dump(rf, "../models/random_forest_model.pkl")
    joblib.dump(X_train.columns.tolist(), "../models/feature_columns.pkl")

    print("Model training complete and saved.")

    # Test loading
    loaded_model = joblib.load("../models/random_forest_model.pkl")
    loaded_features = joblib.load("../models/feature_columns.pkl")

    print("Model and feature columns loaded successfully.")

    # Quick prediction test
    # sample_prediction = loaded_model.predict(X_test[:5])
    # print("Sample predictions:", sample_prediction)

if __name__ == "__main__":
    main()