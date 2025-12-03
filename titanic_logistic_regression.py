"""
Titanic Survival Prediction using Logistic Regression

Steps:
1. Load data
2. Clean & preprocess
3. Split train/test
4. Train Logistic Regression
5. Evaluate with Accuracy & Classification Report
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
# Use your own file path here (Kaggle Titanic train.csv or similar)
data = pd.read_csv("titanic.csv")

# Basic cleaning: drop rows with too many missing key values
data = data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
data.dropna(inplace=True)

# 2. Separate features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# 3. Define numeric and categorical columns
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

# 4. Preprocessing: scale numeric, one-hot encode categorical
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. Create model
model = LogisticRegression(max_iter=1000)

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Build full pipeline: preprocessing + model
from sklearn.pipeline import Pipeline

clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# 8. Train
clf.fit(X_train, y_train)

# 9. Predict & evaluate
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
