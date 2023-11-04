import pickle
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd

app = Flask('predict_loan')

# Load the model and the preprocessors
model = pickle.load(open('logreg.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Define the columns that should exist after one-hot encoding during training
# These should be the columns you ended up with in your training dataset after one-hot encoding
expected_columns = [
    'Gender', 'Married', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
    'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
]


def preprocess_input(data):
    # Assume data is a dictionary containing the input from the client.
    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Apply label encoding
    binary_columns = ['Gender', 'Married', 'Self_Employed', 'Education']
    for col in binary_columns:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = le.transform(df[col])

    # Apply one-hot encoding
    # For simplicity, assume 'Dependents' and 'Property_Area' are the columns to be one-hot encoded.
    df = pd.get_dummies(df, columns=['Dependents', 'Property_Area'])

    # Check that all the columns expected during training are present
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder the columns to match the training dataframe
    df = df[expected_columns]

    # Scale numerical features
    numerical_columns = ['ApplicantIncome',
                         'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Return preprocessed DataFrame
    return df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request.
        data = request.get_json(force=True)

        # Preprocess the input data
        preprocessed_df = preprocess_input(data)

        # Make prediction
        prediction = model.predict(preprocessed_df)

        # Prepare and send the response
        prediction_response = prediction.tolist()
        return jsonify({'prediction': prediction_response})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
