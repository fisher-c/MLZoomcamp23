import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def read_data(file_path):
    return pd.read_csv(file_path)


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")


def preprocess_data(df):
    # Impute mssing values
    # Impute missing values for categorical variables using mode
    categorical_columns = ['Gender', 'Married',
                           'Dependents', 'Self_Employed', 'Credit_History']
    for column in categorical_columns:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)
    # Impute missing values for numerical variables using median
    numerical_columns = ['LoanAmount', 'Loan_Amount_Term']
    for column in numerical_columns:
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)

    # Dictionary to hold label encoders for each column
    label_encoders = {}

    binary_columns = ['Gender', 'Married',
                      'Self_Employed', 'Loan_Status', 'Education']
    for column in binary_columns:
        # Create a new LabelEncoder for each column
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        # Save each LabelEncoder in the dictionary
        label_encoders[column] = le

    save_model(label_encoders, 'label_encoders.pkl')

    # One-hot encode non-binary categorical columns
    df = pd.get_dummies(df, columns=['Dependents', 'Property_Area'])

    # feature scaling, use standard scaler for numerical features
    scaler = StandardScaler()
    numerical_columns = ['ApplicantIncome',
                         'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    save_model(scaler, 'scaler.pkl')

    return df


def prepare_data_for_training(df, target_column, drop_columns):
    X = df.drop(columns=drop_columns)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    # Define an extended set of hyperparameters and their possible values
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        # 'saga' solver supports both l1 and l2 penalty
        'solver': ['liblinear', 'saga'],
        'max_iter': [50, 100, 200, 300],
        # add class_weight if we suspect class imbalance
        'class_weight': ['balanced', None]
    }

    logreg = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def main():
    train_data = read_data('train_u6lujuX_CVtuZ9i.csv')
    train_preprocessed = preprocess_data(train_data)

    X_train, X_val, y_train, y_val = prepare_data_for_training(
        train_preprocessed,
        target_column='Loan_Status',
        drop_columns=['Loan_ID', 'Loan_Status']
    )

    best_model = train_model(X_train, y_train)

    save_model(best_model, 'logreg.pkl')


if __name__ == "__main__":
    main()
