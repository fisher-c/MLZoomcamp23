import pickle


def load_file(filename: str):
    with open(filename, 'rb') as file_in:
        return pickle.load(file_in)


dict_vectorizer = load_file('dv.bin')
logistic_regression_model = load_file('model1.bin')

client_data = {"job": "retired", "duration": 445, "poutcome": "success"}

# transform the client's data using the DictVectorizer
X = dict_vectorizer.transform([client_data])

# make a prediction using the logistic regression model
y_pred = logistic_regression_model.predict_proba(X)[0, 1]

print("The probability that this client will get a credit is: \n", y_pred)
