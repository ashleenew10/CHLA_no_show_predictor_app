import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open("best_no_show_model.pkl", "rb") as file:
        return pickle.load(file)

best_model = load_model()

# Load the trained encoders
@st.cache_resource
def load_encoders():
    with open("NEW_no_show_encoder.pkl", "rb") as encoder_file:
        return pickle.load(encoder_file)

encoder_dict = load_encoders()

# Load the cleaned 2024 dataset
@st.cache_data
def load_data():
    return pd.read_csv("NEW_CLEAN_CHLA_clean_data_2024_Appointments.csv")

df_2024 = load_data()

# Expected categorical features that need encoding
category_col = ['ZIPCODE', 'CLINIC', 'IS_REPEAT', 'APPT_TYPE_STANDARDIZE', 
                'ETHNICITY_STANDARDIZE', 'RACE_STANDARDIZE']

# Expected features in the trained model
expected_features = best_model.feature_names_in_

# Function to preprocess input data
def preprocess_input(df):
    """
    Encodes categorical features and ensures input matches model training.
    """
    df = df.copy()

    # Encode categorical variables using stored encoders
    for col in category_col:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 
                encoder_dict[col].transform([x])[0] if x in encoder_dict[col].classes_ 
                else encoder_dict[col].transform(['Unknown'])[0])

    # Ensure dataset has all features the model expects
    missing_cols = set(expected_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0  # Default fill for missing columns

    # Reorder columns to match model training
    df = df[expected_features]
    
    return df

# Function to make predictions
def predict_no_show(features):
    """
    Predicts No-Show status and probability.
    """
    y_prob = best_model.predict_proba(features)[:, 1]  # Probability of No-Show
    y_pred = np.where(y_prob >= 0.5, "No-Show", "Show-Up")

    return y_pred, y_prob.round(2)

# Streamlit App
def main():
    st.title("CHLA Patient No-Show Prediction")
    st.write("Select clinic and appointment date range to view no-show predictions.")

    # User inputs
    clinic_name = st.selectbox("Select Clinic", df_2024["CLINIC"].unique())

    # Get available date range
    min_date = pd.to_datetime(df_2024["BOOK_DATE"]).min()
    max_date = pd.to_datetime(df_2024["APPT_DATE"]).max()

    start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End Date", min_value=start_date, max_value=max_date, value=max_date)

    # Predict Button
    if st.button("Get Predictions"):
        df_filtered = df_2024[(df_2024["CLINIC"] == clinic_name) & 
                              (pd.to_datetime(df_2024["APPT_DATE"]) >= pd.to_datetime(start_date)) & 
                              (pd.to_datetime(df_2024["APPT_DATE"]) <= pd.to_datetime(end_date))]

        if df_filtered.empty:
            st.warning("No appointments found for the selected clinic and date range.")
        else:
            # Keep necessary identifiers for output
            output_data = df_filtered[["MRN", "APPT_ID", "APPT_DATE", "HOUR_OF_DAY"]].copy()

            # Preprocess data for prediction
            X_input = preprocess_input(df_filtered)

            # Get predictions
            y_pred, y_prob = predict_no_show(X_input)

            # Add predictions to output
            output_data["No-Show Prediction"] = y_pred
            output_data["Probability of No-Show"] = y_prob

            # Display results
            st.subheader("Predicted No-Show Appointments")
            st.dataframe(output_data)

if __name__ == "__main__":
    main()


