# Import required libraries
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully")
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove missing values (if any)
data = data.dropna()

# List of categorical columns
categorical_columns = [
    "Gender",
    "Employment_Type",
    "House_Owned",
    "Education",
    "Marital_Status",
    "Approved"
]

# Dictionary to store encoders
encoders = {}

# Encode categorical columns
for column in categorical_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder

print("\nCategorical Data Encoded Successfully!")

# Separate input and output
X = data.drop("Approved", axis=1)
y = data["Approved"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save encoders
joblib.dump(encoders, "models/encoders.pkl")

print("\nPreprocessing Completed Successfully!")

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))