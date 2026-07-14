# Import libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("dataset.csv")

# Encode categorical columns
categorical_columns = [
    "Gender",
    "Employment_Type",
    "House_Owned",
    "Education",
    "Marital_Status",
    "Approved"
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder

# Separate features and target
X = data.drop("Approved", axis=1)
y = data["Approved"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dictionary of models
models = {

    "Logistic Regression": LogisticRegression(),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

best_accuracy = 0
best_model = None
best_model_name = ""

print("=" * 50)

# Train every model
for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name} Accuracy : {accuracy:.2f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

print("=" * 50)

print("Best Model :", best_model_name)

print("Accuracy :", best_accuracy)

# Save best model
joblib.dump(best_model, "models/model.pkl")

print("\nModel Saved Successfully!")