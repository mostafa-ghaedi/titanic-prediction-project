# titanic_app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("D:/machine_learning/titanic_final/src/titanic.csv")
df["Age"] = df.groupby("Pclass")["Age"].transform(lambda group: group.fillna(group.mean()))
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].fillna("S").map({"S": 0, "C": 1, "Q": 2})

# Select features and target
features = df[["Age", "Fare", "Pclass", "Sex", "FamilySize", "IsAlone", "Embarked"]]
target = df["Survived"]

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse_original = mean_squared_error(y_test, y_pred)
r2_original = r2_score(y_test, y_pred)
print(f"Original MSE: {mse_original:.4f}, R^2: {r2_original:.4f}")

# Optimize by removing features (remove Embarked)
features_optimized = features_scaled_df.drop(columns=["Embarked"])
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(features_optimized, target, test_size=0.2, random_state=42)
model_opt = LinearRegression()
model_opt.fit(X_train_opt, y_train_opt)
y_pred_opt = model_opt.predict(X_test_opt)
mse_optimized = mean_squared_error(y_test_opt, y_pred_opt)
r2_optimized = r2_score(y_test_opt, y_pred_opt)
print(f"Optimized MSE (without Embarked): {mse_optimized:.4f}, R^2: {r2_optimized:.4f}")

# Visualize coefficients
coef_df = pd.DataFrame({"Feature": features.columns, "Coefficient": model.coef_})
plt.figure(figsize=(10, 6))
plt.bar(coef_df["Feature"], coef_df["Coefficient"], color="green")
plt.title("Feature Coefficients")
plt.xticks(rotation=45)
plt.show()