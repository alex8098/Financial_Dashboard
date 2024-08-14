import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y, data.feature_names

def get_classifier(clf_name, params):
    if clf_name == "SVM":
        return SVC(C=params["C"], kernel=params["kernel"])
    elif clf_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"])
    elif clf_name == "Random Forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
    else:
        return KNeighborsClassifier(n_neighbors=params["K"])

def run():
    st.title("Machine Learning Playground")

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")

    # Dataset selection
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

    # Model selection
    classifier_name = st.sidebar.selectbox("Select Classifier", ("SVM", "Decision Tree", "Random Forest", "KNN"))

    # Load dataset
    X, y, feature_names = get_dataset(dataset_name)

    # Display dataset info
    st.write(f"## {dataset_name} Dataset")
    st.write("Shape of dataset:", X.shape)
    st.write("Number of classes:", len(np.unique(y)))

    # Parameter selection based on classifier
    params = {}
    if classifier_name == "SVM":
        params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        params["kernel"] = st.sidebar.selectbox("Kernel", ("rbf", "linear"))
    elif classifier_name == "Decision Tree":
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 10)
    elif classifier_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 10)
    else:
        params["K"] = st.sidebar.slider("K", 1, 15)

    # Get classifier
    clf = get_classifier(classifier_name, params)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and predict
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Classifier: {classifier_name}")
    st.write(f"Accuracy: {acc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=np.unique(y), y=np.unique(y),
                       title="Confusion Matrix", color_continuous_scale="Viridis")
    st.plotly_chart(fig_cm)

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write("Classification Report:")
    st.dataframe(df_report)

    # PCA Visualization
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig_pca = px.scatter(x=x1, y=x2, color=y,
                         labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                         title="PCA Visualization")
    st.plotly_chart(fig_pca)

    # Feature Importance (for tree-based models)
    if classifier_name in ["Decision Tree", "Random Forest"]:
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig_imp = go.Figure(go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h'
        ))
        fig_imp.update_layout(title="Feature Importances", xaxis_title="Importance", yaxis_title="Features")
        st.plotly_chart(fig_imp)