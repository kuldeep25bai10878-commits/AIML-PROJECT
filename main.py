import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Linear curve
from sklearn.preprocessing import PolynomialFeatures # Quadratic curve
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data load
df = pd.read_csv("StudentPerformanceFactors.csv").dropna()

# 2. Features and Target
features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions']
X = df[features]
y = df['Exam_Score']

# 3. Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. --- EFFICIENCY UPGRADE: MODEL TRAINING---
# Model : Degree 4
poly_4 = PolynomialFeatures(degree=4)
X_train_p4 = poly_4.fit_transform(X_train)
X_test_p4 = poly_4.transform(X_test)

model_4 = LinearRegression().fit(X_train_p4, y_train)
p4_pred = model_4.predict(X_test_p4)

#5. Model Accuracy
print(f"{'Model Type':<15} | {'R² Score (%)':<15} | {'MSE':<10}")
print("-" * 45)
print(f"{'Degree 4':<15} | {r2_score(y_test, p4_pred)*100:<15.2f} | {mean_squared_error(y_test, p4_pred):<10.2f}")

# 6. Prediction Logic 
print("\n--- Student Score Predictor ---")
try:
    h = float(input("Enter weekly study hours: "))
    a = float(input("Enter attendance percentage: "))
    b = float(input("Enter previous score: "))
    t = float(input("Enter number of tutoring sessions: "))

    # Passing data as -> DataFrame -> names consistent
    user_input = pd.DataFrame([[h, a, b, t]], columns = features)
    user_input_p4 = poly_4.transform(user_input)
    pred_quad = model_4.predict(user_input_p4) # using quadratic
    
    print(f"{'Degree 4':<15} | {np.clip(pred_quad[0], 0, 100):>14.2f}")
    
except ValueError:
    print("Please enter valid numerical values.")

# 7. Visualizing
plt.figure(figsize=(10, 4))

# 8. Plotting

# Scatter for the dots
plt.scatter(df['Hours_Studied'],
            df['Exam_Score'], color='orange', alpha=0.2, 
            label='Actual Data', s=10)

# Data Variables
x_smooth = np.linspace(df['Hours_Studied'].min(), df['Hours_Studied'].max(), 300)
avg_attendance = df['Attendance'].mean()
avg_prev = df['Previous_Scores'].mean()
avg_tutor = df['Tutoring_Sessions'].mean()

plot_data = pd.DataFrame({
    'Hours_Studied': x_smooth,
    'Attendance': [avg_attendance]*300,
    'Previous_Scores': [avg_prev]*300,
    'Tutoring_Sessions': [avg_tutor]*300
})

# Plot for Quad Curve
plt.plot(x_smooth, model_4.predict(poly_4.transform(plot_data)), 
         color='blue', label='Quartic Curve (d=4)',
         linewidth=2, linestyle='--')

plt.title("Correlation Analysis: Study Impact on Academic Performance", fontweight='bold')
plt.xlabel("Weekly Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig("Correlation_Graph.png", dpi=300)
plt.show()
