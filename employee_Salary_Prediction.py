import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv(r"C:\Users\THANMAI REDDY\Downloads\adult 3.csv")

# -----------------------------
# 2. Data Cleaning
# -----------------------------
data['workclass'] = data['workclass'].replace({'?': 'Others'})
data['occupation'] = data['occupation'].replace({'?': 'Others'})
data = data[(data['workclass'] != 'Without-pay') & (data['workclass'] != 'Never-worked')]

# Drop redundant column
data = data.drop(columns=['education'])

# -----------------------------
# 3. Features and Target
# -----------------------------
X = data.drop(columns=['income'])
y = data['income']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# -----------------------------
# 4. Preprocessing Pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# -----------------------------
# 5. Candidate Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_model = None
best_score = 0
accuracies = {}

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 7. Train and Evaluate Models
# -----------------------------
for name, clf in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_score:
        best_score = acc
        best_model = pipe

# -----------------------------
# 8. Save Best Model
# -----------------------------
joblib.dump(best_model, "best_model.pkl")
print(f"✅ Best model saved with accuracy {best_score:.4f}")

# -----------------------------
# 9. Visualizations
# -----------------------------

# Model Accuracy Comparison
plt.figure(figsize=(8,5))
plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue','lightgreen','salmon'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.0)
plt.show()

# Feature Distribution Example: Age
plt.figure(figsize=(8,5))
plt.hist(data['age'], bins=20, color='skyblue', edgecolor='black')
plt.title("Age Distribution of Employees")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Salary Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="Set2")
plt.title("Income Class Distribution")
plt.xlabel("Salary Class")
plt.ylabel("Count")
plt.show()
