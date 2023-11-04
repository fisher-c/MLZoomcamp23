import requests

url = "http://localhost:9696/predict"

clients_data = [{
    "Gender": "Male",
    "Married": "No",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "Yes",
    "ApplicantIncome": 10416,
    "CoapplicantIncome": 0,
    "LoanAmount": 187,
    "Loan_Amount_Term": 360,
    "Credit_History": 0,
    "Property_Area": "Urban"
},
    {
    "Gender": "Female",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 4567,
        "CoapplicantIncome": 0,
        "LoanAmount": 100,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Rural"
},
]

for i, client_data in enumerate(clients_data):
    response = requests.post(url, json=client_data).json()
    print(f"Response for client {i+1}: {response}")
