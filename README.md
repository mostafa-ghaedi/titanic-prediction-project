# 🚢 Titanic Survival Prediction Project

## Overview
This project focuses on predicting the survival chances of passengers aboard the Titanic using a Linear Regression model. Leveraging the Titanic dataset from Kaggle, it includes features such as Age, Fare, Pclass, Sex, FamilySize, IsAlone, and Embarked. The workflow encompasses data preprocessing, exploratory data analysis (EDA), model training with optimization, and the creation of an interactive dashboard using Streamlit for data visualization and exploration.

---

## Project Steps

### 1. Data Preprocessing
- **Missing Value Handling**: Replaced missing `Age` values with the mean age per `Pclass` group.
- **Feature Engineering**:
  - Calculated `FamilySize` by adding `SibSp`, `Parch`, and 1 (the passenger).
  - Derived `IsAlone` as a binary feature (1 if `FamilySize` is 1, 0 otherwise).
- **Encoding**:
  - Mapped `Sex` to numerical values (`male`: 0, `female`: 1).
  - Encoded `Embarked` (`S`: 0, `C`: 1, `Q`: 2) and filled missing values with `S`.

### 2. Exploratory Data Analysis (EDA)
- Generated a correlation matrix and visualized it with a heatmap to identify relationships between features.
- Computed summary statistics (e.g., mean, minimum, maximum) for key numerical features like `Age` and `Fare`.

### 3. Model Training and Optimization
- **Feature Standardization**: Applied `StandardScaler` to standardize features (mean = 0, standard deviation = 1).
- **Model**: Trained a Linear Regression model to predict the `Survived` outcome (binary: 0 or 1).
- **Optimization**: Improved model performance by experimenting with feature removal (e.g., excluding `Embarked`) to minimize Mean Squared Error (MSE).

### 4. Interactive Dashboard with Streamlit
- Developed a Streamlit-based dashboard with the following features:
  - Filters for Gender, Passenger Class, Age range, and Fare range.
  - Displays filtered dataset and the number of passengers.
  - Shows model performance metrics (MSE, R², Accuracy) for both overall and filtered data.
  - Visualizes feature importance (coefficients) with a bar chart.
  - Presents a correlation heatmap using `matplotlib`.
  - Provides summary statistics for filtered data.
  - Displays survival rates by passenger class and gender with bar charts.

---

## How to Run the Project

### Prerequisites
- Python 3.8 or later
- A virtual environment (recommended)

### Setup Instructions
1. **Clone the Repository**: https://github.com/mostafa-ghaedi/titanic-prediction-project.git


2. **Create and Activate a Virtual Environment**: python -m venv venv
.\venv\Scripts\activate  # On Windows


3. **Install Required Libraries**: pip install pandas numpy scikit-learn matplotlib streamlit


4. **Prepare the Data**:
- Place the `titanic.csv` file (available from Kaggle) in the project root directory.

5. **Run the Dashboard**: streamlit run dashboard.py

- The dashboard will open in your default web browser.

---

## Results
- **Overall Model Performance** (on all data):
- Mean Squared Error (MSE): 0.1568
- R² Score: 0.3725
- Accuracy (threshold=0.5): 0.7989
- **Optimized Model** (after removing `Embarked`):
- MSE: 0.1542
- R² Score: 0.3810
- Accuracy: 0.8034
- **Key Insights**:
- The `Sex` feature has the highest coefficient, indicating it’s a critical predictor of survival.
- Survival rates are notably higher for females and first-class passengers.
- A strong negative correlation exists between `Pclass` and `Survived` (higher class numbers correlate with lower survival rates).

---

## Lessons Learned
- **Data Preprocessing**: Proper handling of missing values and encoding categorical variables significantly enhances model performance.
- **Linear Regression Limitations**: Linear Regression is not optimal for binary classification tasks like survival prediction, as it outputs continuous values requiring a threshold, which may reduce accuracy.
- **Feature Selection**: Removing less influential features (e.g., `Embarked`) slightly improved the model by reducing noise.
- **Streamlit Benefits**: Streamlit proved to be an excellent tool for creating interactive dashboards, making results accessible to non-technical users.

---

## Future Improvements
- **Model Enhancement**: Replace Linear Regression with Logistic Regression or advanced models (e.g., Random Forest, XGBoost) for better binary classification performance.
- **Additional Features**: Incorporate features like cabin location or titles extracted from passenger names.
- **Advanced Visualizations**: Integrate Plotly for more interactive and dynamic visualizations.
- **Hyperparameter Tuning**: Apply hyperparameter tuning (e.g., feature selection or regularization) to further optimize the model.
- **Cross-Validation**: Implement k-fold cross-validation to ensure robust generalization to unseen data.

---

## Project Structure
- `titanic.csv`: The dataset used for analysis and modeling.
- `titanic_app.py`: Script for preprocessing, training, and optimizing the model.
- `dashboard.py`: Streamlit application for interactive visualization.
- `.gitignore`: Excludes the virtual environment (`venv`), temporary files, and large data files from version control.
- `README.md`: This documentation file.

---

## Acknowledgments
- **Dataset**: [Titanic Dataset](https://www.kaggle.com/c/titanic/data) from Kaggle.
- **Tools**: Python, pandas, scikit-learn, matplotlib, Streamlit.
- **Inspiration**: This project was developed as part of a learning journey to master data science and machine learning workflows.

---

## Contact
For questions, feedback, or collaboration, please reach out via GitHub or email at [mustafa.ghaedi@gmail.com].