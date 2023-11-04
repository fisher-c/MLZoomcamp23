from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the models
with open('dv.bin', 'rb') as file_in:
    dict_vectorizer = pickle.load(file_in)

with open('model1.bin', 'rb') as file_in:
    logistic_regression_model = pickle.load(file_in)

# in order to send the customer information we need to post its data


@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.json
    # apply the one-hot encoding feature to the customer data
    X = dict_vectorizer.transform([client_data])
    prob = logistic_regression_model.predict_proba(X)[0][1]
    # send back the data in json format to the user
    return jsonify({"probability": prob})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
