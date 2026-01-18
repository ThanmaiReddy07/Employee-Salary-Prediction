import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\THANMAI REDDY\Downloads\adult 3.csv")

# Clean missing values
data['workclass'] = data['workclass'].replace({'?': 'Others'})
data['occupation'] = data['occupation'].replace({'?': 'Others'})
data = data[(data['workclass'] != 'Without-pay') & (data['workclass'] != 'Never-worked')]

# Drop redundant column
data = data.drop(columns=['education'])

# Features and target
X = data.drop(columns=['income'])
y = data['income']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# Full pipeline: preprocessing + model
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit pipeline
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save pipeline
joblib.dump(pipe, "best_model.pkl")
print("✅ Saved trained pipeline as best_model.pkl")
