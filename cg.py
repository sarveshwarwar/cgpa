import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px # type: ignore

# App title
st.title("🎓 Advanced CGPA Prediction App")

# Sidebar Inputs
st.sidebar.header("Enter Student Details")
attendance = st.sidebar.slider("Attendance (%)", 50, 100, 85)
study_hours = st.sidebar.slider("Daily Study Hours", 0, 10, 4)
projects = st.sidebar.number_input("Completed Projects", 0, 20, 2)
internship = st.sidebar.slider("Internship Duration (months)", 0, 12, 3)
backlogs = st.sidebar.number_input("Number of Backlogs", 0, 10, 0)
sem_avg = st.sidebar.slider("Average Semester Marks (%)", 50, 100, 75)

# Sample dataset (In real case, load actual data)
np.random.seed(42)
data = pd.DataFrame({
    'Attendance': np.random.randint(60, 100, 100),
    'StudyHours': np.random.randint(1, 8, 100),
    'Projects': np.random.randint(0, 10, 100),
    'InternshipMonths': np.random.randint(0, 12, 100),
    'Backlogs': np.random.randint(0, 5, 100),
    'SemMarks': np.random.randint(55, 95, 100),
    'CGPA': np.random.uniform(5, 10, 100)
})

# Train model
X = data.drop("CGPA", axis=1)
y = data["CGPA"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction
input_data = np.array([[attendance, study_hours, projects, internship, backlogs, sem_avg]])
predicted_cgpa = model.predict(input_data)[0]

# Display results
st.subheader("📌 Predicted CGPA:")
st.success(f"{predicted_cgpa:.2f}")

# Model accuracy
accuracy = r2_score(y_test, model.predict(X_test))
st.write(f"**Model Accuracy:** {accuracy*100:.2f}%")

# Feature importance chart
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance")
st.plotly_chart(fig)
