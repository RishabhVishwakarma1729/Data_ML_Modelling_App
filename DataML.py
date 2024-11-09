# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import io

st.title("Machine Learning App with Regression and Classification")

# File upload
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    # Load file
    try:
        data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.write("### Dataset Preview", data.head())
        
        # Dataset info
        if st.checkbox("Show Dataset Info"):
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            if st.checkbox("Show Null Values Count"):
                st.write(data.isnull().sum())
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.warning("Please upload a CSV or Excel file to continue.")

# Data Preprocessing Options
if uploaded_file:
    columns_to_process = st.multiselect("Select Columns for Imputation or Drop", data.columns)

    # Imputation
    impute_option = st.selectbox("Select Imputation Method", ["None", "Mean", "Median", "Mode", "Drop NaN Rows"])
    if impute_option != "None":
        for col in columns_to_process:
            if impute_option == "Mean":
                data[col].fillna(data[col].mean(), inplace=True)
            elif impute_option == "Median":
                data[col].fillna(data[col].median(), inplace=True)
            elif impute_option == "Mode":
                data[col].fillna(data[col].mode()[0], inplace=True)
            elif impute_option == "Drop NaN Rows":
                data.dropna(subset=columns_to_process, inplace=True)

    # Label Encoding
    if st.checkbox("Apply Label Encoding"):
        label_encoder = LabelEncoder()
        categorical_columns = st.multiselect("Select Categorical Columns for Label Encoding", data.select_dtypes(include=['object']).columns)
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])

    # Scaling
    scale_option = st.selectbox("Select Scaling Method", ["None", "Standard Scaling", "MinMax Scaling"])
    if scale_option != "None":
        scaler = StandardScaler() if scale_option == "Standard Scaling" else MinMaxScaler()
        data[data.select_dtypes(include=np.number).columns] = scaler.fit_transform(data.select_dtypes(include=np.number))
    
    st.write("### Preprocessed Data", data.head())

# Model Selection
if uploaded_file:
    target_column = st.selectbox("Select Target Column", data.columns)
    feature_columns = st.multiselect("Select Feature Columns", [col for col in data.columns if col != target_column])

    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        model_type = st.selectbox("Choose a Model Type", ["Regression", "Classification"])
        
        if model_type == "Regression":
            regressor = st.selectbox("Select Regression Model", ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression", "Logistic Regression"])
        else:
            classifier = st.selectbox("Select Classification Model", ["Decision Tree", "Random Forest", "SVM", "Naive Bayes", "k-NN"])

        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == "Regression":
                if regressor == "Logistic Regression":
                    model = LogisticRegression()
                elif regressor == "Simple Linear Regression":
                    model = LinearRegression()
                elif regressor == "Multiple Linear Regression":
                    model = LinearRegression()
                elif regressor == "Polynomial Regression":
                    degree = st.slider("Polynomial Degree", 2, 5)
                    poly_features = PolynomialFeatures(degree=degree)
                    X_poly_train = poly_features.fit_transform(X_train)
                    X_poly_test = poly_features.transform(X_test)
                    model = LinearRegression()
                    model.fit(X_poly_train, y_train)
                    predictions = model.predict(X_poly_test)
                    st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")
                    st.write(f"R^2 Score: {r2_score(y_test, predictions)}")
            else:
                if classifier == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif classifier == "Random Forest":
                    model = RandomForestClassifier(n_estimators=10)
                elif classifier == "SVM":
                    model = SVC()
                elif classifier == "Naive Bayes":
                    model = GaussianNB()
                elif classifier == "k-NN":
                    model = KNeighborsClassifier()
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                st.write(f"Accuracy Score: {accuracy_score(y_test, predictions)}")
                plt.figure(figsize=(10, 6))
                ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
                plt.title(f"{classifier}: Confusion Matrix")
                st.pyplot(plt)
else:
    st.warning("Please select target and feature columns.")
