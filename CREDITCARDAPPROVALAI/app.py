from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask application
app = Flask(__name__)

# Load saved files
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    applicant = {

        "Gender": request.form["Gender"],

        "Age": int(request.form["Age"]),

        "Income": float(request.form["Income"]),

        "Employment_Years": int(request.form["Employment_Years"]),

        "Loan_Amount": float(request.form["Loan_Amount"]),

        "Credit_Score": int(request.form["Credit_Score"]),

        "Existing_Loan": float(request.form["Existing_Loan"]),

        "Past_Due_Payments": int(request.form["Past_Due_Payments"]),

        "Credit_Inquiries": int(request.form["Credit_Inquiries"]),

        "Employment_Type": request.form["Employment_Type"],

        "House_Owned": request.form["House_Owned"],

        "Education": request.form["Education"],

        "Marital_Status": request.form["Marital_Status"]

    }

    input_data = pd.DataFrame([applicant])

    categorical_columns = [
        "Gender",
        "Employment_Type",
        "House_Owned",
        "Education",
        "Marital_Status"
    ]

    for column in categorical_columns:
        input_data[column] = encoders[column].transform(input_data[column])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

# Get prediction probability
    probability = model.predict_proba(input_scaled)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 1:
        result = "Credit Card Approved ✅"
    else:
        result = "Credit Card Rejected ❌"

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)