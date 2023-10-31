import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFECV
import json
import joblib

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Page title
st.title('Titanic Survival Prediction App')

# Data Exploration
st.header('Data Exploration')
st.write('Display the first few rows of the dataset:')
st.write(titanic_data.head())

# Data Preprocessing
st.header('Data Preprocessing')
# Drop unnecessary columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
titanic_data = titanic_data.drop(columns_to_drop, axis=1)

# Handle categorical variables ('Embarked' and 'Sex')
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked', 'Sex'], drop_first=True)

st.write('Modified dataset with dummy variables:')
st.write(titanic_data.head())

from sklearn.preprocessing import StandardScaler

# Feature Scaling
st.header('Feature Scaling')

# Separate the target variable ('Survived') from the features
X = titanic_data.drop('Survived', axis=1)  # Features
y = titanic_data['Survived']  # Target variable

# Remove non-numeric columns (e.g., 'Name', 'Ticket', 'Cabin')
X_numeric = X.select_dtypes(include=['number'])

# Initialize the StandardScaler
scaler = StandardScaler()

# Perform feature scaling on the numeric features (X_numeric)
X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)

# Display the first 5 rows of the scaled dataset
X_scaled.head(5)
 

# Train and Evaluate Models
st.header('Train and Evaluate Models')
#Train and Evaluate Models:
#Import classifiers and train models

from sklearn.model_selection import train_test_split

# Separate the target variable ('Survived') from the features
X = titanic_data.drop('Survived', axis=1)  # Features
y = titanic_data['Survived']  # Target variable

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['number']))
X_test_scaled = scaler.transform(X_test.select_dtypes(include=['number']))

# Create an imputer with the mean strategy
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on the training data
X_train_scaled = imputer.fit_transform(X_train_scaled)

# Transform the test data using the same imputer
X_test_scaled = imputer.transform(X_test_scaled)

# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}

for model_name, model in models.items():
    # Train the model on the scaled training data
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)
    print("=" * 40)

# Model Tuning
st.header('Model Tuning')
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with RandomForestClassifier
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

# Get the best model and its score
best_model = grid.best_estimator_
best_score = grid.best_score_

print(f"Best model: {best_model}")
print(f"Best score: {best_score:.2f}")

# Feature Selection
st.header('Feature Selection')
#feature_selection using RFECV

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Separate the target variable ('Survived') from the features
X = titanic_data.drop(['Survived', 'Name'], axis=1)  # Remove 'Name' column
y = titanic_data['Survived']  # Target variable

# Remove non-numeric columns (e.g., 'Ticket', 'Cabin')
X_numeric = X.select_dtypes(include=['number'])

# Initialize the StandardScaler
scaler = StandardScaler()

# Perform feature scaling on the numeric features (X_numeric)
X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Instantiate the best model (e.g., RandomForestClassifier with tuned hyperparameters)
best_model = RandomForestClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=10)

# Create the RFECV object and fit it to the training data
selector = RFECV(best_model, step=1, cv=5, scoring='accuracy')
selector.fit(X_train, y_train)

# Get the selected features and their ranks
selected_features = X_numeric.columns[selector.support_]
feature_ranks = selector.ranking_

print(f"Selected features: {selected_features}")
print(f"Feature ranks: {feature_ranks}")

# Model Evaluation
st.header('Model Evaluation')
# Convert selected_features to a list
selected_features_list = selected_features.tolist()

# Remove target variable from the list of selected features if it's present
if 'Survived' in selected_features_list:
    selected_features_list.remove('Survived')

# Ensure that X_train and X_test have the same columns
X_train = X_train[:, selector.support_]
X_test = X_test[:, selector.support_]

# Train the best model on the list of selected features
best_model = best_model.fit(X_train, y_train)

# Make predictions on the test set using the trained model
y_pred = best_model.predict(X_test)

# Evaluate the model using accuracy_score
from sklearn.metrics import accuracy_score

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy with selected features: {test_accuracy}")

# Display Confusion Matrix
st.subheader('Confusion Matrix')
# Load the confusion matrix image
confusion_matrix_image = 'confusion_matrix.png'
st.image(confusion_matrix_image, use_column_width=True)

# Save selected features to a JSON file
with open("selected_features.json", "r") as f:
    selected_features_list = json.load(f)
st.write('Selected features:')
st.write(selected_features_list)

# Save the best model to a file
best_model = joblib.load("best_model1.pkl")
st.write('Best model loaded from file:')
st.write(best_model)

# Streamlit app entry point
if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)



