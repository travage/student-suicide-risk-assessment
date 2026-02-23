import streamlit as st

import sys
import os
# Get the absolute path to the project root (one level up)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add project root to Python path
sys.path.append(PROJECT_ROOT)
# Directory where charts are
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")

from src.predict import predict_from_input, encode_nominal

# Category mappings for ordinal features
STRESS_MAPPING = {'Low': 0, 'Moderate': 1, 'High': 2}
ACADEMIC_MAPPING = {'Excellent': 0, 'Good': 1, 'Average': 2, 'Poor': 3}
HEALTH_MAPPING = {'Normal': 0, 'Fair': 1, 'Abnormal': 2}
DEPRESSION_MAPPING = {'Sometimes': 0, 'Often': 1, 'Always': 2}
ANXIETY_MAPPING = {'Sometimes': 0, 'Often': 1, 'Always': 2}
SELF_HARM_MAPPING = {'No': 0, 'Yes': 1}

# Mapping of predicted class to human-readable label
CLASS_TO_LABEL = {
    0: 'Low Risk',
    1: 'Medium Risk',
    2: 'High Risk'
}

st.markdown("<h1 style='text-align: center;'>Student Suicide Risk Assessor</h1>", unsafe_allow_html=True)

# Initialize session state for form counter to force form reset
if 'form_counter' not in st.session_state:
    st.session_state.form_counter = 0

# Form to enter in survey responses - use key with counter to force recreate on clear
with st.form(key=f'survey_form_{st.session_state.form_counter}', clear_on_submit=False):
    st.write('**Please enter survey responses below**')

    age = st.number_input('Age:', min_value=18, max_value=99, value=None, help='Accepted age range is 18 - 99')
    gender = st.selectbox('Gender', ['Male', 'Female'], index=None)
    # Divide main questions, submit button, and clear button over three columns
    c1, c2, c3 = st.columns(3)
    with c1:
        stress_level = st.selectbox('Stress Level', ['High', 'Moderate', 'Low'], index=None)
        relationship_condition = st.selectbox('Relationship Condition', ['In A Relationship', 'Single', 'Breakup'], index=None)
        anxiety_level = st.selectbox('Anxiety Level', ['Always', 'Often', 'Sometimes'], index=None)
        submit_button = st.form_submit_button('Submit responses')
    with c2:
        academic_performance = st.selectbox('Academic Performance', ['Excellent', 'Good', 'Average', 'Poor'], index=None)
        family_problem = st.selectbox('Family Problem', ['None', 'Financial', 'Parental Conflict'], index=None)
        mental_support = st.selectbox('Mental Support', ['Family', 'Friends', 'Loneliness'], index=None)
    with c3:
        health_condition = st.selectbox('Health Condition', ['Normal', 'Fair', 'Abnormal'], index=None)
        depression_level = st.selectbox('Depression Level', ['Always', 'Often', 'Sometimes'], index=None)
        self_harm_story = st.selectbox('Self-Harm Story', ['Yes', 'No'], index=None)
        clear_button = st.form_submit_button('Clear form')

# Handle clear button - increment counter to force form recreation
if clear_button:
    st.session_state.form_counter += 1
    st.rerun()

if submit_button:
    all_responses = [
        age,
        gender,
        stress_level,
        relationship_condition,
        anxiety_level,
        academic_performance,
        family_problem,
        mental_support,
        health_condition,
        depression_level,
        self_harm_story,
    ]

    total_questions = len(all_responses)
    answered_questions = sum(v is not None for v in all_responses)

    # Require at least half of the questions to be answered to make a prediction
    if answered_questions < total_questions // 2:
        st.error(f"Please answer at least {total_questions // 2} of the questions before submitting.")
    else:
        # Build features dictionary
        features = {}

        # Numerical / ordinal features
        features['age'] = age
        features['stress_level'] = STRESS_MAPPING[stress_level] if stress_level else None
        features['academic_performance'] = ACADEMIC_MAPPING[academic_performance] if academic_performance else None
        features['health_condition'] = HEALTH_MAPPING[health_condition] if health_condition else None
        features['depression_level'] = DEPRESSION_MAPPING[depression_level] if depression_level else None
        features['anxiety_level'] = ANXIETY_MAPPING[anxiety_level] if anxiety_level else None
        features['self_harm_story'] = SELF_HARM_MAPPING[self_harm_story] if self_harm_story else None

        # Nominal / dummy features
        features.update(encode_nominal('gender', gender))
        features.update(encode_nominal('relationship_condition', relationship_condition))
        features.update(encode_nominal('family_problem', family_problem))
        features.update(encode_nominal('mental_support', mental_support))

        # Call prediction function
        result = predict_from_input(features)

        # Display results
        st.subheader('Machine Learning Model Risk Assessment')
        if result['label'] == 'Low Risk':
            st.write('Predicted suicide risk level: **:green[Low Risk]**')
        elif result['label'] == 'Medium Risk':
            st.write('Predicted suicide risk level: **:orange[Medium Risk]**')
        elif result['label'] == 'High Risk':
            st.write('Predicted suicide risk level: **:red[High Risk]**')

        if 'probabilities' in result:
            st.write('Class probabilities:')
            for cls, prob in result['probabilities'].items():
                st.write(f"- {CLASS_TO_LABEL[cls]}: {prob:.2%}")

        # Display visualizations
        st.subheader('Model Insights and Data Overview')
        st.image(
            os.path.join(VISUALS_DIR, "suicide_attempt_distribution.png"),
            caption="Distribution of Suicide Attempt Categories"
        )

        st.image(
            os.path.join(VISUALS_DIR, "top_10_feature_importances.png"),
            caption="Top 10 Feature Importances (Random Forest)"
        )

        st.image(
            os.path.join(VISUALS_DIR, "distribution_by_stress_level.png"),
            caption="Distribution of Suicide Attempt Categories by Stress Level"
        )