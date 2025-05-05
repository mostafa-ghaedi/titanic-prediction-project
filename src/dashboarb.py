# dashboard.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score

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

# Train model
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Streamlit dashboard
st.title("Titanic Survival Prediction Dashboard")

# Show sample data
st.subheader("Sample Data")
st.write(df.head())

# Correlation heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(features_scaled_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
fig, ax = plt.subplots()
ax.scatter([1, 2, 3, 4], [1, 2, 3, 4])
st.pyplot(fig)

# Interactive filter by Sex
st.subheader("Survival Prediction by Gender")
sex = st.selectbox("Select Gender", ["All", "male", "female"])
if sex == "male":
    filtered_df = df[df["Sex"] == 0]
elif sex == "female":
    filtered_df = df[df["Sex"] == 1]
else:
    filtered_df = df
filtered_features = filtered_df[["Age", "Fare", "Pclass", "Sex", "FamilySize", "IsAlone", "Embarked"]]
filtered_target = filtered_df["Survived"]
X_filtered = scaler.fit_transform(filtered_features)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_filtered, filtered_target, test_size=0.2, random_state=42)
model_f = LinearRegression()
model_f.fit(X_train_f, y_train_f)
y_pred_f = model_f.predict(X_test_f)
st.write(f"MSE for {sex}: {mean_squared_error(y_test_f, y_pred_f):.4f}")

# Show coefficients
st.subheader("Feature Coefficients")
coef_df = pd.DataFrame({"Feature": features.columns, "Coefficient": model.coef_})
st.bar_chart(coef_df.set_index("Feature")["Coefficient"])

# Run with: streamlit run dashboard.py