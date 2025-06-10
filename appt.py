import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# Set wide layout and page title
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\atpt1\Titanic\titanic_data.csv")
    df = df.dropna()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    return df

@st.cache_resource
def train_models(df):
    X = df[['Pclass', 'Age', 'SibSp', 'Sex']]
    y = df['Survived']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    tree_model = DecisionTreeClassifier(max_leaf_nodes=50, random_state=42)
    tree_model.fit(X_train, y_train)

    y_log_proba = log_model.predict_proba(X_valid)[:, 1]
    y_tree_proba = tree_model.predict_proba(X_valid)[:, 1]

    fpr_log, tpr_log, _ = roc_curve(y_valid, y_log_proba)
    fpr_tree, tpr_tree, _ = roc_curve(y_valid, y_tree_proba)

    auc_log = roc_auc_score(y_valid, y_log_proba)
    auc_tree = roc_auc_score(y_valid, y_tree_proba)

    return log_model, tree_model, (fpr_log, tpr_log, auc_log), (fpr_tree, tpr_tree, auc_tree)

# Load data and models
df = load_data()
log_model, tree_model, roc_log, roc_tree = train_models(df)

# Sidebar: Inputs
st.sidebar.title("🎫 Passenger Input")
pclass = st.sidebar.selectbox("🚢 Passenger Class", [1, 2, 3])
age = st.sidebar.slider("🎂 Age", 0, 80, 30)
sibsp = st.sidebar.number_input("👨‍👩‍👧‍👦 Siblings/Spouses Aboard", 0, 8, 0)
sex = st.sidebar.radio("⚧️ Sex", ["Male", "Female"], horizontal=True)

# Main area
st.title("🚢 Titanic Survival Prediction Dashboard")
st.markdown("Predict whether a Titanic passenger would survive based on input values, and visualize model performance.")

# Prediction Section
st.markdown("## 🔍 Prediction Result")

sex_encoded = 1 if sex == "Female" else 0
input_data = np.array([[pclass, age, sibsp, sex_encoded]])

log_prob = log_model.predict_proba(input_data)[0][1]
tree_pred = tree_model.predict(input_data)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("🧠 Logistic Regression", f"{log_prob:.2%} survival probability")

with col2:
    st.metric("🌲 Decision Tree Prediction", "✅ Survived" if tree_pred == 1 else "❌ Did Not Survive")

# ROC Curve Plot
st.markdown("## 📊 ROC Curve Comparison")

fpr_log, tpr_log, auc_log = roc_log
fpr_tree, tpr_tree, auc_tree = roc_tree

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {auc_log:.2f})")
ax.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {auc_tree:.2f})", linestyle="--")
ax.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")
st.pyplot(fig)
# Prepare prediction result for download
prediction_df = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Sex": [sex],
    "LogisticRegression_Probability": [round(log_prob, 4)],
    "DecisionTree_Prediction": ["Survived" if tree_pred == 1 else "Did Not Survive"]
})
st.markdown("### 📥 Download Prediction")

csv = prediction_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📄 Download as CSV",
    data=csv,
    file_name='titanic_prediction_result.csv',
    mime='text/csv'
)
