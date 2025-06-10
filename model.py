import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Load dataset
df = pd.read_csv(r"C:\Users\atpt1\Titanic\titanic_data.csv")
# Drop rows with missing values
df_clean = df.dropna()

# Encode categorical variables
df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})

# Select features
features = ['Pclass', 'Age', 'SibSp', 'Sex']
X = df_clean[features]
y = df_clean['Survived']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Predictions
y_pred_log = log_model.predict(X_valid)
y_proba_log = log_model.predict_proba(X_valid)[:, 1]

# Coefficients
coeff_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': log_model.coef_[0]
})
print(coeff_df)
tree_model = DecisionTreeClassifier(max_leaf_nodes=50, random_state=42)
tree_model.fit(X_train, y_train)

# Predictions
y_pred_tree = tree_model.predict(X_valid)
y_proba_tree = tree_model.predict_proba(X_valid)[:, 1]
def evaluate_model(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

# Evaluate both models
print("Logistic Regression:")
evaluate_model(y_valid, y_pred_log, y_proba_log)

print("\nDecision Tree:")
evaluate_model(y_valid, y_pred_tree, y_proba_tree)
# Example: Pclass=2, Age=30, SibSp=1, Sex=female (1)
sample_input = np.array([[2, 30, 1, 1]])
log_prob = log_model.predict_proba(sample_input)[0][1]
tree_pred = tree_model.predict(sample_input)[0]

print(f"Logistic Regression Probability of Survival: {log_prob:.8f}")
print(f"Decision Tree Prediction: {'Survived' if tree_pred == 1 else 'Did Not Survive'}")

