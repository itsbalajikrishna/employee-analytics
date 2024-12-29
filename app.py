import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Train the Attrition Model
def train_attrition_model(data):
    # Check if "Attrition" column exists
    if "Attrition" not in data.columns:
        raise ValueError("The dataset must contain an 'Attrition' column.")
    
    # Prepare target and features
    target = data["Attrition"].astype("category").cat.codes
    features = data.drop(columns=["Attrition", "EmployeeNumber", "Name"], errors="ignore")
    
    # Encode categorical variables
    label_encoders = {}
    for column in features.select_dtypes(include="object").columns:
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column])
        label_encoders[column] = le
    
    # Scale numeric features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train a RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, scaler, label_encoders


# Prediction Function
def predict_attrition(test_data, model, scaler, label_encoders):
    # Check if "Name" exists and store it for final output
    names = test_data["Name"] if "Name" in test_data.columns else None
    
    # Drop irrelevant columns, keeping only features used for prediction
    test_data = test_data.drop(columns=["Attrition", "EmployeeNumber", "Name"], errors="ignore").copy()
    
    # Apply label encoding to categorical columns
    for column, le in label_encoders.items():
        if column in test_data.columns:
            test_data[column] = le.transform(test_data[column])
    
    # Scale numeric features
    features = scaler.transform(test_data)
    
    # Predict probabilities and outcomes
    probabilities = model.predict_proba(features)[:, 1]  # Probability of Attrition = Yes
    predictions = model.predict(features)
    
    # Add predictions and probabilities to the test data
    test_data["Prediction"] = np.where(predictions == 1, "Yes", "No")
    test_data["Attrition Probability"] = probabilities
    
    # Add back the "Name" column if it exists
    if names is not None:
        test_data.insert(0, "Name", names.values)
    
    # Return the sorted DataFrame
    return test_data.sort_values(by="Attrition Probability", ascending=False)


# Configure the page
st.set_page_config(page_title="Employee Analytics", layout="wide")

# Set up the upload directory
UPLOAD_DIR = "data"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


# Helper function to save uploaded file
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, "uploaded_data.xlsx")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# Helper function to load the uploaded file
def load_uploaded_file():
    file_path = os.path.join(UPLOAD_DIR, "uploaded_data.xlsx")
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    return None


# Title of the App
st.title("Employee Analytics")

# Sidebar for Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Choose a section",
    options=["Upload Data", "Attrition", "Performance", "Predict"],
)

# Load previously uploaded file if available
uploaded_data = load_uploaded_file()

# Upload Section
if menu == "Upload Data":
    st.subheader("Upload Employee Data")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file:
        save_uploaded_file(uploaded_file)
        st.success("File uploaded successfully!")
    if uploaded_data is not None:
        st.write("Uploaded Data Preview:")
        st.dataframe(uploaded_data.head())
    else:
        st.warning("No data uploaded yet. Please upload an Excel file.")

# Attrition Section
elif menu == "Attrition":
    st.subheader("Attrition Analysis")
    if uploaded_data is not None:
        st.write("Analyzing attrition data...")

        # Key Features for Visualization
        key_features = [
            "DistanceFromHome",
            "Gender",
            "MonthlyIncome",
            "JobRole",
            "JobSatisfaction",
            "NumCompaniesWorked",
            "PercentSalaryHike",
            "PerformanceRating",
            "YearsSinceLastPromotion",
        ]

        # Visualizations
        st.write("### Key Feature Visualizations Against Attrition")
        for feature in key_features:
            if feature in uploaded_data.columns:
                st.write(f"#### {feature} vs Attrition")
                
                # Use Plotly for interactive visualizations
                if uploaded_data[feature].dtype == "object":
                    fig = px.histogram(
                        uploaded_data,
                        x=feature,
                        color="Attrition",
                        barmode="group",
                        title=f"{feature} vs Attrition",
                        labels={"Attrition": "Attrition", feature: feature},
                    )
                else:
                    fig = px.box(
                        uploaded_data,
                        x="Attrition",
                        y=feature,
                        title=f"{feature} vs Attrition",
                        labels={"Attrition": "Attrition", feature: feature},
                    )
                
                # Display the interactive plot
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload data in the 'Upload Data' section.")

# Performance Section
elif menu == "Performance":
    st.subheader("Performance Analysis")
    if uploaded_data is not None:
        st.write("Analyzing performance data...")

        # Distribution of Performance Ratings
        st.write("### Performance Rating Distribution")
        if "PerformanceRating" in uploaded_data.columns:
            # Plotly histogram for PerformanceRating
            fig = px.histogram(
                uploaded_data,
                x="PerformanceRating",
                title="Performance Rating Distribution",
                labels={"PerformanceRating": "Performance Rating"},
                nbins=5,
                color_discrete_sequence=["blue"],
            )
            st.plotly_chart(fig, use_container_width=True)

        # Categorize employees into Low, Medium, and High Performers
        st.write("### Performance Categories")
        if "PerformanceRating" in uploaded_data.columns:
            low_performers = uploaded_data[uploaded_data["PerformanceRating"] <= 2]
            medium_performers = uploaded_data[uploaded_data["PerformanceRating"] == 3]
            high_performers = uploaded_data[uploaded_data["PerformanceRating"] >= 4]

            st.write(f"Low Performers: {len(low_performers)}")
            st.write(f"Medium Performers: {len(medium_performers)}")
            st.write(f"High Performers: {len(high_performers)}")

            # Display data for High Performers
            st.write("#### High Performers Overview")
            st.dataframe(high_performers[["Name", "JobSatisfaction", "YearsSinceLastPromotion"]])

        # Features influencing performance
        st.write("### Features Impacting Performance")
        features_to_analyze = [
            "JobSatisfaction",
            "YearsSinceLastPromotion",
            "PercentSalaryHike",
            "TrainingTimesLastYear",
        ]

        for feature in features_to_analyze:
            if feature in uploaded_data.columns:
                st.write(f"#### {feature} vs Performance Rating")
                fig = px.box(
                    uploaded_data,
                    x="PerformanceRating",
                    y=feature,
                    title=f"{feature} vs Performance Rating",
                    labels={"PerformanceRating": "Performance Rating", feature: feature},
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload data in the 'Upload Data' section.")

# Predict Section
elif menu == "Predict":
    st.subheader("Prediction Analysis")
    if uploaded_data is not None:
        # Train the model on the uploaded data
        st.write("Training attrition model...")
        model, scaler, label_encoders = train_attrition_model(uploaded_data)
        st.success("Model trained successfully!")

        # Upload test data
        st.write("### Upload Test Data for Predictions")
        st.markdown(
            """
            **Note:** The test Excel sheet must include the following columns:
            - `Name`: Employee's name
            - `Gender`: Employee's gender
            - `JobSatisfaction`: Satisfaction level at the job
            - `PerformanceRating`: Employee's performance rating
            - `YearsSinceLastPromotion`: Years since the employee's last promotion
            
            Ensure the column names match exactly to avoid errors.
            """
        )
        test_file = st.file_uploader("Upload a test Excel file", type=["xlsx"], key="test_file")
        if test_file:
            test_data = pd.read_excel(test_file)
            st.write("Test Data Preview:")
            st.dataframe(test_data)


            # Make predictions
            st.write("### Prediction Results")
            predictions = predict_attrition(test_data, model, scaler, label_encoders)
            if "Name" in predictions.columns:
                st.dataframe(predictions[["Name", "Attrition Probability", "Prediction"]])
            else:
                st.dataframe(predictions[["Attrition Probability", "Prediction"]])

            # Highlight top 10 employees likely to leave
            st.write("### Top 10 Employees Most Likely to Leave")
            top_10 = predictions.head(10)
            if "Name" in top_10.columns:
                st.dataframe(top_10[["Name", "Attrition Probability", "Prediction"]])
            else:
                st.dataframe(top_10[["Attrition Probability", "Prediction"]])

     
    else:
        st.warning("Please upload data in the 'Upload Data' section.")
