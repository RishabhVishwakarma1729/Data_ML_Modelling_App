# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

st.title("Machine Learning App with Regression and Classification")

# File upload
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    # Load file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Data Preview:")
    st.write(data.head())

    # Imputation options
    impute_option = st.selectbox("Select Imputation Method", ["None", "Mean", "Median", "Mode"])
    if impute_option != "None":
        for col in data.select_dtypes(include=np.number).columns:
            if impute_option == "Mean":
                data[col].fillna(data[col].mean(), inplace=True)
            elif impute_option == "Median":
                data[col].fillna(data[col].median(), inplace=True)
            elif impute_option == "Mode":
                data[col].fillna(data[col].mode()[0], inplace=True)

    # Scaling options
    scale_option = st.selectbox("Select Scaling Method", ["None", "Standard Scaling", "MinMax Scaling"])
    if scale_option != "None":
        scaler = StandardScaler() if scale_option == "Standard Scaling" else MinMaxScaler()
        data[data.select_dtypes(include=np.number).columns] = scaler.fit_transform(data.select_dtypes(include=np.number))
    
    st.write("Preprocessed Data:")
    st.write(data.head())
    
    # Model type selection
    model_type = st.selectbox("Choose a Model Type", ["Regression", "Classification"])

    if model_type == "Regression":
        regressor = st.selectbox("Select Regression Model", ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression", "Logistic Regression"])
    else:
        classifier = st.selectbox("Select Classification Model", ["Decision Tree", "Random Forest", "SVM", "Naive Bayes", "k-NN"])

    # Splitting data
    target_column = st.selectbox("Select Target Column", data.columns)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "Regression":
            if regressor == "Simple Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Visualization for Simple Linear Regression
                if X.shape[1] == 1:  # Only works for single feature
                    plt.figure(figsize=(10, 6))
                    plt.scatter(X_test, y_test, color="blue", label="Actual")
                    plt.plot(X_test, predictions, color="red", label="Predicted")
                    plt.title("Simple Linear Regression: Actual vs Predicted")
                    plt.xlabel("Feature")
                    plt.ylabel("Target")
                    plt.legend()
                    st.pyplot(plt)
                
            elif regressor == "Multiple Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Visualization for Multiple Linear Regression
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
                plt.scatter(range(len(predictions)), predictions, color="red", label="Predicted")
                plt.title("Multiple Linear Regression: Actual vs Predicted")
                plt.legend()
                st.pyplot(plt)

            elif regressor == "Polynomial Regression":
                poly = PolynomialFeatures(degree=2)
                X_poly_train = poly.fit_transform(X_train)
                X_poly_test = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_poly_train, y_train)
                predictions = model.predict(X_poly_test)

                # Visualization for Polynomial Regression
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
                plt.scatter(range(len(predictions)), predictions, color="red", label="Predicted")
                plt.title("Polynomial Regression: Actual vs Predicted")
                plt.legend()
                st.pyplot(plt)

            elif regressor == "Logistic Regression":
                model = LogisticRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Visualization for Logistic Regression
                plt.figure(figsize=(10, 6))
                ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
                plt.title("Logistic Regression: Confusion Matrix")
                st.pyplot(plt)

        else:
            if classifier == "Decision Tree":
                model = DecisionTreeClassifier()
            elif classifier == "Random Forest":
                model = RandomForestClassifier()
            elif classifier == "SVM":
                model = SVC()
            elif classifier == "Naive Bayes":
                model = GaussianNB()
            elif classifier == "k-NN":
                model = KNeighborsClassifier()

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Display accuracy
            st.write(f"Accuracy Score: {accuracy_score(y_test, predictions)}")

            # Confusion Matrix for Classification
            plt.figure(figsize=(10, 6))
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
            plt.title(f"{classifier}: Confusion Matrix")
            st.pyplot(plt)
