import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/model.pkl")

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Load encoders
encoders = joblib.load("models/encoders.pkl")

# -----------------------------
# Sample Applicant Details
# -----------------------------

applicant = {

    "Gender": "Male",

    "Age": 30,

    "Income": 55000,

    "Employment_Years": 6,

    "Loan_Amount": 150000,

    "Credit_Score": 740,

    "Existing_Loan": 25000,

    "Past_Due_Payments": 0,

    "Credit_Inquiries": 1,

    "Employment_Type": "Salaried",

    "House_Owned": "Yes",

    "Education": "Graduate",

    "Marital_Status": "Single"

}

# Convert dictionary into DataFrame
input_data = pd.DataFrame([applicant])

# Encode categorical columns
categorical_columns = [

    "Gender",

    "Employment_Type",

    "House_Owned",

    "Education",

    "Marital_Status"

]

for column in categorical_columns:

    input_data[column] = encoders[column].transform(input_data[column])

# Scale data
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)

# Display Result
if prediction[0] == 1:

    print("Credit Card Status : APPROVED")

else:

    print("Credit Card Status : REJECTED")