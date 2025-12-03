"""
House Price Prediction using Linear Regression

Steps:
1. Load data
2. Select useful features
3. Preprocess (numeric + categorical)
4. Train Linear Regression
5. Evaluate with RMSE
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Load dataset
# Use your own file path (e.g. Kaggle House Prices train.csv)
data = pd.read_csv("house_prices.csv")

# Select a small subset of features for simplicity
cols = [
    "SalePrice",          # target
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "Neighborhood",       # categorical
]
data = data[cols].dropna()

# 2. Separate features and target
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

numeric_features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
categorical_features = ["Neighborhood"]

# 3. Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 4. Define model
model = LinearRegression()

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.pipeline import Pipeline

reg = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# 6. Train
reg.fit(X_train, y_train)

# 7. Predict & evaluate
y_pred = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")
