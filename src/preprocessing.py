import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Convert column names to lowercase and snake case
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Any age above 100 is likely an error, so it will be replaced with the median age of the data
    # Context: Found age of 221 in original data
    median_age = df.loc[df['age'] < 100, 'age'].median()
    df.loc[df['age'] >= 100, 'age'] = median_age

    # Clean strings
    for col in df.columns:
        if col == 'age':
            continue
        else:
            df[col] = df[col].str.strip().str.title()

    # Replace invalid answers in depression_level column to 'Missing'
    # Context: Found invalid answer of 'pf' in depression_level column in original data
    df.loc[~df['depression_level'].isin(['Sometimes', 'Often', 'Always']), 'depression_level'] = 'Missing'

    # Ordinal columns
    ordinal_mappings = {
        'stress_level': {
            'Low': 0,
            'Moderate': 1,
            'High': 2
        },
        'academic_performance': {
            'Excellent': 0,
            'Good': 1,
            'Average': 2,
            'Poor': 3
        },
        'health_condition': {
            'Normal': 0,
            'Fair': 1,
            'Abnormal': 2
        },
        'depression_level': {
            'Sometimes': 0,
            'Often': 1,
            'Always': 2
        },
        'anxiety_level': {
            'Sometimes': 0,
            'Often': 1,
            'Always': 2
        },
        'self_harm_story': {
            'No': 0,
            'Yes': 1
        },
        'suicide_attempt': {
            'Never Thought': 0,
            'Thought': 1,
            'Attempted': 2
        }
    }

    # Apply ordinal mappings to the dataframe
    for column, mapping in ordinal_mappings.items():
        df[column] = df[column].map(mapping)

    # Set missing values in depression_level column to -1 and convert to int
    df['depression_level'] = df['depression_level'].fillna(-1).astype(int)

    nominal_columns = ['gender', 'relationship_condition', 'family_problem', 'mental_support']
    # One-hot encoding of nominal columns; preserving all dummy columns to preserve interpretability
    df = pd.get_dummies(df, columns=nominal_columns, drop_first=False)
    # Some categories are multiword and therefore introduce spaces in column names
    df.columns = df.columns.str.replace(' ', '_')

    return df