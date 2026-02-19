import joblib
import pandas as pd

MODEL = joblib.load('../models/random_forest_model.pkl')
FEATURE_COLUMNS = joblib.load('../models/feature_columns.pkl')

CLASS_TO_LABEL = {
    0: 'Low Risk',
    1: 'Medium Risk',
    2: 'High Risk'
}

def predict_from_input(input_data: dict) -> dict:
    """
    Runs Random Forest classifier on input data
    :param input_data: Survey responses as a dictionary where keys are feature names and values are the corresponding responses.
    :return: Dictionary containing the predicted class, label, and probabilities for each class.
    """

    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])

    # Align to training feature schema
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Convert all values to numeric
    # Nan values will be converted to 0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    pred_class = int(MODEL.predict(df)[0])
    pred_label = CLASS_TO_LABEL.get(pred_class, 'Unknown Risk Level')

    result = {
        'class': pred_class,
        'label': pred_label
    }

    # Add class probabilities to result
    probs = MODEL.predict_proba(df)[0]
    result['probabilities'] = {int(c): float(p) for c, p in zip(MODEL.classes_, probs)}

    return result

def encode_nominal(prefix: str, value: str, feature_columns: list[str]) -> dict:
    """
    Encodes a nominal feature into a one-hot encoded dictionary.
    :param prefix: The prefix for the feature (e.g. mental_support)
    :param value: The value of the nominal feature to encode (e.g. Loneliness)
    :param feature_columns: List of all feature columns used in training to detect the categories
    :return: A dictionary with one-hot encoded values for the nominal feature
    """
    categories = detect_categories(prefix, feature_columns)
    encoded = {f"{prefix}_{cat}": 1 if cat == value else 0 for cat in categories}
    return encoded


def detect_categories(feature_name: str, feature_columns: list[str]) -> list[str]:
    """
    Detects the categories for a nominal feature based on the feature columns used in training.
    :param feature_name: Name of the nominal feature
    :param feature_columns: List of all feature columns used in training
    :return: List of detected categories for the nominal feature
    """
    categories = []
    prefix = f"{feature_name}_"
    for col in feature_columns:
        if col.startswith(prefix):
            categories.append(col[len(prefix):])
    return categories

# if __name__ == "__main__":
#     example = {"age": 20, "stress_level": 2, "mental_support_Loneliness": 1}
#     print(predict_from_input(example))
