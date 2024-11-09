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
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

# Show dataset info
if st.checkbox("Show Dataset Info"):
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    # Show number of null values
    if st.checkbox("Show Null Values Count"):
        st.write(data.isnull().sum())

    st.write("Data Preview:")
    st.write(data.head())

    # Column selection for imputation or drop
    columns_to_process = st.multiselect("Select Columns for Imputation or Drop", data.columns)
    
    # Imputation options
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

    # Label Encoding option
    if st.checkbox("Apply Label Encoding"):
        label_encoder = LabelEncoder()
        categorical_columns = st.multiselect("Select Categorical Columns for Label Encoding", data.select_dtypes(include=['object']).columns)
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])

    # Scaling options
    scale_option = st.selectbox("Select Scaling Method", ["None", "Standard Scaling", "MinMax Scaling"])
    if scale_option != "None":
        scaler = StandardScaler() if scale_option == "Standard Scaling" else MinMaxScaler()
        data[data.select_dtypes(include=np.number).columns] = scaler.fit_transform(data.select_dtypes(include=np.number))
    
    st.write("Preprocessed Data:")
    st.write(data.head())

   # Option to choose target column
target_column = st.selectbox("Select Target Column", data.columns)

# Option to choose feature columns, excluding the target column
feature_columns = st.multiselect(
    "Select Feature Columns",
    [col for col in data.columns if col != target_column]
)
    # Model type selection
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
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Logistic Regression Visualization
                if X.shape[1] == 1:  # Visualization only works for single feature
                    x_range = np.linspace(X_test.min().values[0], X_test.max().values[0], 100)
                    y_prob = model.predict_proba(x_range.reshape(-1, 1))[:, 1]

                    plt.figure(figsize=(10, 6))
                    plt.scatter(X_test, y_test, color="blue", label="Actual")
                    plt.plot(x_range, y_prob, color="red", label="Predicted Probability")
                    plt.title("Logistic Regression Probability Curve")
                    plt.xlabel("Feature")
                    plt.ylabel("Probability of Positive Class")
                    plt.legend()
                    st.pyplot(plt)

        else:
            if classifier == "Decision Tree":
                model = DecisionTreeClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Decision Tree Visualization
                plt.figure(figsize=(12, 8))
                plot_tree(model, filled=True, feature_names=feature_columns, class_names=True)
                plt.title("Decision Tree Visualization")
                st.pyplot(plt)

            elif classifier == "Random Forest":
                model = RandomForestClassifier(n_estimators=10)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Random Forest Visualization
                plt.figure(figsize=(15, 10))
                for i, tree in enumerate(model.estimators_[:3]):  # Visualize first 3 trees
                    plt.subplot(1, 3, i + 1)
                    plot_tree(tree, filled=True, feature_names=feature_columns, class_names=True)
                    plt.title(f"Random Forest Tree {i+1}")
                st.pyplot(plt)

            elif classifier == "k-NN":
                model = KNeighborsClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # k-NN Visualization for Binary Classification (2D Visualization)
                if X.shape[1] == 2:  # Only works with 2 feature columns
                    h = 0.02
                    x_min, x_max = X_test.iloc[:, 0].min() - 1, X_test.iloc[:, 0].max() + 1
                    y_min, y_max = X_test.iloc[:, 1].min() - 1, X_test.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    plt.figure(figsize=(10, 6))
                    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
                    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, s=40, cmap=plt.cm.coolwarm, edgecolor='k')
                    plt.xlabel(feature_columns[0])
                    plt.ylabel(feature_columns[1])
                    plt.title("k-NN Classification Decision Boundary")
                    st.pyplot(plt)
            
            # Display Confusion Matrix for Classification
            st.write(f"Accuracy Score: {accuracy_score(y_test, predictions)}")
            plt.figure(figsize=(10, 6))
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
            plt.title(f"{classifier}: Confusion Matrix")
            st.pyplot(plt)

# This code provides all the requested visualizations based on the selected algorithm.

    

