## Loan Prediction: An End-to-End Machine Learning Project

### Overview

This application is designed to predict loan approval outcomes using machine learning. It provides an automated approach to the traditional loan approval process, enhancing efficiency and reducing bias.


### Problem Statement

In the financial industry, loan approval is a critical process that involves assessing the risk and creditworthiness of loan applicants. The process can be costly and time-consuming, and it's subject to human error and bias. This project aims to build an automated system using machine learning that can predict whether a loan should be approved based on historical data.


### Business Context

In the current scenario, loan eligibility is determined by financial experts who manually review all the aspects of the loan application. This process is not only time-consuming and costly but also prone to human errors and biases. By introducing a machine learning system, we seek to:
- Enhance the speed and efficiency of the loan approval process.
- Minimize the operational costs associated with the manual processing of loan applications.
- Provide a consistent and unbiased assessment of loan applications.

### Dataset

The dataset is available on Kaggle and can be found here: [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/data).

### Workflow

The project workflow includes:

1. Data Exploration and Visualization
2. Data Preparation
3. Model Selection and Training
4. Model Fine-Tuning
5. Solution Presentation
6. Deployment

### Model

A logistic regression model has been trained with historical data and can predict the likelihood of an applicant's loan being approved.

### Deployment

The model is deployed using a Flask web application that is containerized with Docker for easy distribution and deployment.

### Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### Prerequisites

- Python 3.6+
- Docker

#### Installing

- Clone the repository to your local machine
- Navigate to the cloned directory
- Build Docker image: `docker build -t loan_predict_app .`
- Run the Docker container: `docker run -p 9696:9696 loan_predict_app`

To test the API, run the `test.py` script from another terminal window: `python test.py`

For batch testing with multiple inputs, modify the `test.py` script to loop over a list of input dictionaries.
