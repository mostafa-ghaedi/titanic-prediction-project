# complete_titanic_dashboard.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import streamlit as st
import matplotlib.pyplot as plt

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Streamlit Dashboard
st.title("🚢 Titanic Survival Prediction Dashboard")
st.markdown("A comprehensive dashboard to explore Titanic survival predictions using Logistic Regression. Use the sidebar to filter data and analyze results!")

# Sidebar for filters
st.sidebar.header("🔍 Filter Options")
sex_filter = st.sidebar.selectbox("Select Gender", ["All", "Male", "Female"])
pclass_filter = st.sidebar.selectbox("Select Passenger Class", ["All", 1, 2, 3])
age_range = st.sidebar.slider("Select Age Range", min_value=0, max_value=80, value=(0, 80))
fare_range = st.sidebar.slider("Select Fare Range", min_value=0, max_value=500, value=(0, 500))
embarked_filter = st.sidebar.selectbox("Select Embarked Port", ["All", "S", "C", "Q"])

# Apply filters
filtered_df = df.copy()
if sex_filter != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == (0 if sex_filter == "Male" else 1)]
if pclass_filter != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass_filter]
if embarked_filter != "All":
    filtered_df = filtered_df[filtered_df["Embarked"] == {"S": 0, "C": 1, "Q": 2}[embarked_filter]]
filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]
filtered_df = filtered_df[(filtered_df["Fare"] >= fare_range[0]) & (filtered_df["Fare"] <= fare_range[1])]

# Train model on filtered data if enough samples
if len(filtered_df) > 10:
    filtered_features = filtered_df[["Age", "Fare", "Pclass", "Sex", "FamilySize", "IsAlone", "Embarked"]]
    filtered_target = filtered_df["Survived"]
    X_filtered = scaler.fit_transform(filtered_features)
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_filtered, filtered_target, test_size=0.2, random_state=42)
    model_f = LogisticRegression(max_iter=200)
    model_f.fit(X_train_f, y_train_f)
    y_pred_f = model_f.predict(X_test_f)
    y_pred_f_prob = model_f.predict_proba(X_test_f)[:, 1]
    filtered_accuracy = accuracy_score(y_test_f, y_pred_f)
    filtered_conf_matrix = confusion_matrix(y_test_f, y_pred_f)
    filtered_fpr, filtered_tpr, _ = roc_curve(y_test_f, y_pred_f_prob)
    filtered_roc_auc = auc(filtered_fpr, filtered_tpr)
else:
    filtered_accuracy = "N/A (Not enough data)"
    filtered_conf_matrix = "N/A (Not enough data)"
    filtered_roc_auc = "N/A (Not enough data)"

# Display filtered data
st.subheader("📋 Filtered Passenger Data")
st.write(f"Displaying {len(filtered_df)} passengers based on selected filters.")
st.dataframe(filtered_df)

# Model Performance Metrics
st.subheader("📊 Model Performance Metrics")
st.write(f"**Overall Model:**")
st.write(f"- Accuracy: {accuracy:.4f}")
st.write(f"**Filtered Model:**")
st.write(f"- Accuracy: {filtered_accuracy}")

# Confusion Matrix
st.subheader("⚙️ Confusion Matrix")
st.write("**Overall Model Confusion Matrix:**")
st.write(pd.DataFrame(conf_matrix, index=["Actual Not Survived", "Actual Survived"], columns=["Predicted Not Survived", "Predicted Survived"]))
if filtered_accuracy != "N/A (Not enough data)":
    st.write("**Filtered Model Confusion Matrix:**")
    st.write(pd.DataFrame(filtered_conf_matrix, index=["Actual Not Survived", "Actual Survived"], columns=["Predicted Not Survived", "Predicted Survived"]))

# Feature Importance
st.subheader("⚖️ Feature Importance (Coefficients)")
coef_df = pd.DataFrame({"Feature": features.columns, "Coefficient": model.coef_[0]})
st.bar_chart(coef_df.set_index("Feature")["Coefficient"])

# Correlation Heatmap
st.subheader("🔗 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = pd.DataFrame(features_scaled, columns=features.columns).corr()
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_matrix.columns)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")
plt.colorbar(im)
ax.set_title("Feature Correlation Heatmap")
st.pyplot(fig)

# ROC Curve
st.subheader("📉 Receiver Operating Characteristic (ROC) Curve")
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve (Overall Model)')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)
if filtered_accuracy != "N/A (Not enough data)":
    st.subheader("📉 Filtered ROC Curve")
    fig_roc_f, ax_roc_f = plt.subplots(figsize=(8, 6))
    ax_roc_f.plot(filtered_fpr, filtered_tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {filtered_roc_auc:.2f})')
    ax_roc_f.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc_f.set_xlim([0.0, 1.0])
    ax_roc_f.set_ylim([0.0, 1.05])
    ax_roc_f.set_xlabel('False Positive Rate')
    ax_roc_f.set_ylabel('True Positive Rate')
    ax_roc_f.set_title('ROC Curve (Filtered Model)')
    ax_roc_f.legend(loc="lower right")
    st.pyplot(fig_roc_f)

# Survival Statistics
st.subheader("📊 Survival Statistics")
survival_rate = filtered_df["Survived"].mean()
st.write(f"Survival Rate for Filtered Data: {survival_rate:.2%}")
if not filtered_df.empty:
    survival_by_class = filtered_df.groupby("Pclass")["Survived"].mean().reset_index()
    survival_by_class.columns = ["Passenger Class", "Survival Rate"]
    st.bar_chart(survival_by_class.set_index("Passenger Class")["Survival Rate"])
    survival_by_sex = filtered_df.groupby("Sex")["Survived"].mean().reset_index()
    survival_by_sex["Sex"] = survival_by_sex["Sex"].map({0: "Male", 1: "Female"})
    survival_by_sex.columns = ["Gender", "Survival Rate"]
    st.bar_chart(survival_by_sex.set_index("Gender")["Survival Rate"])

# Run with: streamlit run complete_titanic_dashboard.py