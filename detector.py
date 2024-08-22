import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import joblib

def preprocess_data(data):
    # Fill missing values
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    object_columns = data.select_dtypes(include=['object']).columns
    for col in object_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Encode categorical variables
    data['Sex'] = data['Sex'].map({'M': 0, 'F': 1})
    data['Ascites'] = data['Ascites'].map({'N': 0, 'Y': 1})
    data['Hepatomegaly'] = data['Hepatomegaly'].map({'N': 0, 'Y': 1})
    data['Spiders'] = data['Spiders'].map({'N': 0, 'Y': 1})
    data['Edema'] = data['Edema'].map({'N': 0, 'S': 1, 'Y': 2})
    data['Status'] = data['Status'].map({'C': 0, 'CL': 1, 'D': 2})
    
    return data

def train_and_evaluate(X_train, X_test, y_train, y_test, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Cross-validation score
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    return model, y_pred, accuracy, report, importances, indices, cross_val_scores

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, labels={"x": "Predicted Label", "y": "True Label"},
                    title="Confusion Matrix", color_continuous_scale="Blues")
    return fig

st.title('Liver Cirrhosis Stage Detection')
st.markdown("This app helps to predict liver cirrhosis stages based on patient data. Upload your CSV file, choose model parameters, and see the results.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(data.head())
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Define features and target
    X = data.drop('Stage', axis=1)
    y = data['Stage']
    
    # One-hot encoding for categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
    X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    X = X.drop(categorical_cols, axis=1).join(X_encoded)
    
    # Feature selection
    feature_options = X.columns.tolist()
    selected_features = st.multiselect("Select features to include:", feature_options, default=feature_options)
    X = X[selected_features]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model parameters
    n_estimators = st.slider('Number of trees in RandomForest:', min_value=10, max_value=200, value=100, step=10)
    
    # Train and evaluate
    with st.spinner('Training the model...'):
        model, y_pred, accuracy, report, importances, indices, cross_val_scores = train_and_evaluate(X_train, X_test, y_train, y_test, n_estimators)
    
    # Save model
    joblib.dump(model, 'model.pkl')
    st.success('Model trained and saved successfully!')
    
    # Display results
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text(f"Classification Report:\n{report}")
    
    # Cross-validation results
    st.write(f"Cross-Validation Scores: {cross_val_scores}")
    st.write(f"Mean Cross-Validation Score: {np.mean(cross_val_scores):.2f}")
    
    # Plot feature importances
    fig, ax = plt.subplots()
    ax.bar(range(X.shape[1]), importances[indices], align="center")
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(X.columns[indices], rotation=90)
    ax.set_title("Feature Importances")
    st.pyplot(fig)
    
    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(y_test, y_pred)
    st.plotly_chart(cm_fig)
