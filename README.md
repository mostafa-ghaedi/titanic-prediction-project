# 🚢 Titanic Survival Dashboard

## Overview
This project predicts Titanic passenger survival using a Logistic Regression model. It includes an interactive Streamlit dashboard with filters, performance metrics, and visualizations.

## How to Run
1. Activate virtual environment:.\venv\Scripts\activate

2. Install dependencies: pip install pandas numpy scikit-learn matplotlib streamlit

3. Run the dashboard:streamlit run complete_titanic_dashboard.py 

- Ensure `titanic.csv` is in the project folder.

## Features
- Filters for Gender, Pclass, Age, Fare, and Embarked.
- Displays accuracy, confusion matrix, ROC curve, and correlation heatmap.
- Shows survival rates by class and gender.

## Results
- Accuracy: ~0.80 (varies by data split)
- ROC AUC: ~0.85

## Contact
For questions, contact [mustafa.ghaedi@gmail.com].