# Task 5: Decision Trees and Random Forests
# Dataset: Heart Disease Dataset (or any classification dataset)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# -------------------------
# Step 1: Load Dataset
# -------------------------
# Download Heart Disease dataset (update path if needed)
file_path = r"C:\Users\Divya P\Downloads\heart.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Step 2: Decision Tree Classifier
# -------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n--- Decision Tree Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"],
          filled=True, max_depth=3)  # limit depth in plot for clarity
plt.show()

# -------------------------
# Step 3: Control Overfitting
# -------------------------
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred_pruned = dt_pruned.predict(X_test)

print("\n--- Pruned Decision Tree Results (max_depth=4) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_pruned))

# -------------------------
# Step 4: Random Forest
# -------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
plt.title("Feature Importances (Random Forest)")
plt.show()

# -------------------------
# Step 5: Cross-validation
# -------------------------
cv_scores_dt = cross_val_score(dt_pruned, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)

print("\nCross-validation Accuracy (Decision Tree):", cv_scores_dt.mean())
print("Cross-validation Accuracy (Random Forest):", cv_scores_rf.mean())
