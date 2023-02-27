# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
# Import EDA
from eda import eda
# Load data
dataset = load_iris()
# Create dataframe with iris data
print(dataset)
data = dataset.data
target_names = dataset.target_names # Classes
feature_names = dataset.feature_names # Columns
target = dataset.target # Output
df = pd.DataFrame(data, columns = feature_names)
# Make target a series
target = pd.Series(target)
# Streamlit
# Set up App

st.set_page_config(page_title="EDA and ML Dashboard", 
                   layout="centered",
                   initial_sidebar_state="auto")
# Add title and markdown decription
st.title("EDA and Prdictive Modelling Dashboard")

# define sidebar and sidebar options
options = ["EDA", "Predictive Modelling"]
selected_option = st.sidebar.selectbox("Select an option", options)
# Do EDA
if selected_option == "EDA":
    # Call/invoke EDA function from ead.py
    eda(df, target_names, feature_names, target)         
# Predictive Modelling
elif selected_option == "Predictive Modelling":
    st.subheader("Predictive Modelling")
    st.write("Choose a transform type and Model from the otpions below:")
    X = df.values
    Y = target.values
    
    test_proportion = 0.30
    seed = 5
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_proportion, random_state=seed)
    transform_options = ["None", 
                         "StandardScaler", 
                         "Normalizer", 
                         "MinMaxScaler"]
    transform = st.selectbox("Select data transform",
                             transform_options)
    if transform == "StandardScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif transform == "Normalizer":
        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif transform == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train
        X_test = X_test
    classifier_list = ["LogisticRegression",
                       "SVM",
                       "DecisionTree",
                       "KNeighbors",
                       "RandomForest"]
    classifier = st.selectbox("Select classifier", classifier_list)
    # Add option to select classifiers
    # Add LogisticRegression
    if classifier == "LogisticRegression":
        st.write("Here are the results of a logistic regression model:")
        solver_value = st.selectbox("Select solver",
                                    ["lbfgs",
                                     "liblinear",
                                     "newton-cg",
                                     "newton-cholesky"])
        model = LogisticRegression(solver=solver_value)
        model.fit(X_train, y_train)
        # Make prediction
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average = "weighted")
        # Display dresults
        st.write(f'Accuracy: {accuracy}')
        st.write(f'Precision: {precision}')
        st.write(f'Recall: {recall}')
        st.write(f'F1-Score: {f1}')
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
    elif classifier == "DecisionTree":
        st.write("Here are the results of a logistic regression model:")
        
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        # Make prediction
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average = "weighted")
        # Display dresults
        st.write(f'Accuracy: {accuracy}')
        st.write(f'Precision: {precision}')
        st.write(f'Recall: {recall}')
        st.write(f'F1-Score: {f1}')
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
           