import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import streamlit as st

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("D:/machine_learning/titanic_final/src/titanic.csv")
    df["Age"] = df.groupby("Pclass")["Age"].transform(lambda group: group.fillna(group.mean()))
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].fillna("S").map({"S": 0, "C": 1, "Q": 2})
    return df

# Load data
df = load_data()

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

# Binarize predictions for accuracy
threshold = 0.5
y_pred_binary = (y_pred >= threshold).astype(int)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_binary)

# Streamlit Dashboard
st.title("🚢 Titanic Survival Prediction Dashboard")
st.markdown("This dashboard allows you to explore the Titanic dataset, analyze feature correlations, and predict survival using a Linear Regression model. Use the filters below to interact with the data!")

# Sidebar for filters
st.sidebar.header("Filter Options")
sex_filter = st.sidebar.selectbox("Filter by Gender", ["All", "Male", "Female"])
pclass_filter = st.sidebar.selectbox("Filter by Passenger Class", ["All", 1, 2, 3])
age_range = st.sidebar.slider("Age Range", min_value=0, max_value=80, value=(0, 80))
fare_range = st.sidebar.slider("Fare Range", min_value=0, max_value=500, value=(0, 500))

# Apply filters
filtered_df = df.copy()
if sex_filter != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == (0 if sex_filter == "Male" else 1)]
if pclass_filter != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass_filter]
filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]
filtered_df = filtered_df[(filtered_df["Fare"] >= fare_range[0]) & (filtered_df["Fare"] <= fare_range[1])]

# Display filtered data
st.subheader("📊 Filtered Dataset")
st.write(f"Showing {len(filtered_df)} passengers after applying filters.")
st.dataframe(filtered_df)

# Train model on filtered data if enough samples
if len(filtered_df) > 10:  # Ensure enough data to train
    filtered_features = filtered_df[["Age", "Fare", "Pclass", "Sex", "FamilySize", "IsAlone", "Embarked"]]
    filtered_target = filtered_df["Survived"]
    X_filtered = scaler.fit_transform(filtered_features)
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_filtered, filtered_target, test_size=0.2, random_state=42)
    model_f = LinearRegression()
    model_f.fit(X_train_f, y_train_f)
    y_pred_f = model_f.predict(X_test_f)
    y_pred_f_binary = (y_pred_f >= threshold).astype(int)
    filtered_mse = mean_squared_error(y_test_f, y_pred_f)
    filtered_r2 = r2_score(y_test_f, y_pred_f)
    filtered_accuracy = accuracy_score(y_test_f, y_pred_f_binary)
else:
    filtered_mse = "N/A (Not enough data)"
    filtered_r2 = "N/A (Not enough data)"
    filtered_accuracy = "N/A (Not enough data)"

# Display model metrics
st.subheader("📈 Model Performance")
st.write(f"**Overall Model (All Data):**")
st.write(f"- Mean Squared Error (MSE): {mse:.4f}")
st.write(f"- R² Score: {r2:.4f}")
st.write(f"- Accuracy (threshold={threshold}): {accuracy:.4f}")
st.write(f"**Filtered Model:**")
st.write(f"- Filtered MSE: {filtered_mse}")
st.write(f"- Filtered R²: {filtered_r2}")
st.write(f"- Filtered Accuracy: {filtered_accuracy}")

# Feature coefficients
st.subheader("⚖️ Feature Importance (Coefficients)")
coef_df = pd.DataFrame({"Feature": features.columns, "Coefficient": model.coef_})
st.bar_chart(coef_df.set_index("Feature")["Coefficient"])

# Correlation matrix as a table (since heatmap is not directly supported by Streamlit)
st.subheader("🔗 Correlation Matrix")
corr_matrix = features_scaled_df.corr()
st.dataframe(corr_matrix.style.format("{:.2f}").background_gradient(cmap="coolwarm", vmin=-1, vmax=1))

# Summary statistics
st.subheader("📝 Summary Statistics")
summary_stats = filtered_df[["Age", "Fare", "Pclass", "FamilySize"]].describe()
st.dataframe(summary_stats)

# Survival rate by class
st.subheader("📉 Survival Rate by Passenger Class")
if not filtered_df.empty:
    survival_by_class = filtered_df.groupby("Pclass")["Survived"].mean().reset_index()
    survival_by_class.columns = ["Passenger Class", "Survival Rate"]
    st.bar_chart(survival_by_class.set_index("Passenger Class")["Survival Rate"])
else:
    st.write("No data available for this filter combination.")