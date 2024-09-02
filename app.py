import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


st.set_page_config(page_title="Employee Attrition Prediction",layout='wide')

# Set title for the Streamlit app
st.title("### Employee Attrition Prediction")

# Upload CSV file
st.subheader("Only Uploaded WA_Fn-UseC_-HR-Employee-Attrition.csv ")
uploaded_file = st.file_uploader("WA_Fn-UseC_-HR-Employee-Attrition.csv", type="csv")
if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Data Preprocessing
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
    df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Encoding categorical columns
    df = df.join(pd.get_dummies(df['BusinessTravel'])).drop('BusinessTravel', axis=1)
    df = df.join(pd.get_dummies(df['Department'], prefix='Department')).drop('Department', axis=1)
    df = df.join(pd.get_dummies(df['EducationField'], prefix='EducationField')).drop('EducationField', axis=1)
    df = df.join(pd.get_dummies(df['JobRole'], prefix='JobRole')).drop('JobRole', axis=1)
    df = df.join(pd.get_dummies(df['MaritalStatus'], prefix='MaritalStatus')).drop('MaritalStatus', axis=1)
    
    # Drop unnecessary columns
    df = df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1)
    
    # Display the first few rows of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())




    # Train Random Forest model
    x, y = df.drop('Attrition', axis=1), df['Attrition']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    model.fit(x_train, y_train)
     

    st.subheader("Random Forest model")
    # Display model accuracy
    accuracy = model.score(x_test, y_test)
    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Feature Importance
    sorted_importances = dict(sorted(zip(model.feature_names_in_, model.feature_importances_), 
                                     key=lambda x: x[1], reverse=True))
    
    # Plot Feature Importance
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(sorted_importances.keys(), sorted_importances.values())
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Display histograms of the dataset
    st.subheader("Dataset Histograms")
    fig, ax = plt.subplots(figsize=(20, 15))
    df.hist(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    

    st.markdown("<h2 style='text-align: center;'>Decision Tree model</h2>", unsafe_allow_html=True)

        
    st.subheader("Decision Tree model")
    # Train Decision Tree model
    x, y = df.drop('Attrition', axis=1), df['Attrition']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)

    # Display model accuracy
    accuracy = model.score(x_test, y_test)
    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Feature Importance
    sorted_importances = dict(sorted(zip(x.columns, model.feature_importances_), key=lambda x: x[1], reverse=True))

    # Plot Feature Importance
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(sorted_importances.keys(), sorted_importances.values())
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # # Display histograms of the dataset
    # st.subheader("Dataset Histograms")
    # fig, ax = plt.subplots(figsize=(20, 15))
    # df.hist(ax=ax)
    # plt.tight_layout()
    # st.pyplot(fig)    


    st.subheader("Dataset Scatter")
    plt.figure(figsize=(20,10))
    plt.scatter(sorted_importances.keys(), sorted_importances.values())
    plt.xticks(rotation=45, ha='right')
    plt.show()

    st.markdown("<h2 style='text-align: center;'>Train Logistic Regression Model</h2>", unsafe_allow_html=True)


    # st.subheader("Train Logistic Regression model") 

    x, y = df.drop('Attrition', axis=1), df['Attrition']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    # Display model accuracy
    accuracy = model.score(x_test, y_test)
    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Predict probabilities and compute ROC curve
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    st.subheader("ROC Curve")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend()
    st.pyplot(fig)

    # Display Confusion Matrix
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    st.pyplot(fig)

    # # Display histograms of the dataset
    # st.subheader("Dataset Histograms")
    # fig, ax = plt.subplots(figsize=(20, 15))
    # df.hist(ax=ax)
    # plt.tight_layout()
    # st.pyplot(fig)

    # Box Plots
    st.subheader("Box Plots")
    features = df.columns.difference(['Attrition'])
    fig, ax = plt.subplots(figsize=(20, 15))

    # Replace 'Salary' with other features if needed
    sns.boxplot(data=df, x='Attrition', y='Age', ax=ax)  
    plt.title('Box Plot of Age by Attrition')
    st.pyplot(fig)

    # Class Distribution Plot
    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Attrition', data=df, palette='viridis')
    plt.title('Class Distribution for Attrition')
    st.pyplot(fig)
