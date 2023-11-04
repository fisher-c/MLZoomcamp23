FROM python:3.10-slim

# Install pipenv
RUN pip install pipenv

# Copy only the necessary Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install the Python dependencies
RUN pipenv install --system --deploy

# Copy your Flask app script, saved model, and any other relevant files
COPY ["predict.py", "logreg.pkl", "scaler.pkl", "label_encoders.pkl", "./"]

# Expose the port the app runs on
EXPOSE 9696

# Run the Flask app with waitress
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
