import pandas as pd
import joblib    
import os        

from sklearn.ensemble import RandomForestClassifier  # The machine learning model we'll use
from sklearn.model_selection import train_test_split # Helper function to split data into training and validation sets
from sklearn.metrics import accuracy_score, classification_report # Functions to evaluate model performance


# Read the training data from the CSV file into a pandas DataFrame
train_df = pd.read_csv('data/train.csv')

# Separate the features (X) from the target variable (y)
X = train_df.drop(columns=['y'])
y = train_df['y']

# Convert text-based columns into numerical format using one-hot encoding.
X = pd.get_dummies(X, drop_first=True)


# Split the dataset into two parts: 80% for training and 20% for validation.
# 'stratify=y' ensures that the proportion of 'yes' and 'no' in the target variable is the same in both the training and validation sets.
# 'random_state=0' makes the split reproducible, so you get the same split every time you run the code.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# This entire section is for finding the best hyperparameters for the model.
# It uses RandomizedSearchCV to test different combinations automatically.
# It is commented out because it takes a very long time to run and has already been completed.
# If you want to test different parameters, make sure you comment out 

# param_dist = {
#     'n_estimators': [20, 50, 100],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2'],
#     'bootstrap': [True, False]
# }
# rf = RandomForestClassifier(random_state=0, n_jobs=-1)
# rf_random = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=param_dist,
#     n_iter=15,
#     cv=3,
#     verbose=2,
#     random_state=0,
#     n_jobs=-1
# )
# rf_random.fit(X, y)
# print(rf_random.best_params_)
# print(rf_random.best_score_)


# These are the best parameters discovered from the RandomizedSearchCV process above.
best_params = {
    'n_estimators': 50,
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'bootstrap': True,
    'random_state': 0
}

# Create an instance of the RandomForestClassifier with the best parameters.
# 'n_jobs=-1' tells the model to use all available CPU cores to speed up training.
model = RandomForestClassifier(**best_params, n_jobs=-1)

# Train the model using the training data (X_train, y_train).
model.fit(X_train, y_train)


# Use the trained model to make predictions on the validation data.
y_pred = model.predict(X_val)

# Print the accuracy score to see what percentage of predictions were correct.
print("Validation Accuracy: ", accuracy_score(y_val, y_pred))

# Print a detailed classification report, which includes precision, recall, and F1-score for each class.
print("Classification Report:\n", classification_report(y_val, y_pred))



# Define the directory where the model will be saved.
model_dir = "src/model"

# Create the directory if it doesn't already exist.
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Define the full path for the model file.
model_path = os.path.join(model_dir, "random_forest.pkl")
# Save the trained model object to the file.
joblib.dump(model, model_path)

# Define the path for the training columns file.
columns_path = os.path.join(model_dir, "training_columns.pkl")
# Save the list of column names used during training. This is crucial for ensuring
# that the input data for future predictions has the exact same structure.
joblib.dump(X_train.columns.tolist(), columns_path)

# Save the validation data and predictions for use in Streamlit visualization.
joblib.dump(y_val, os.path.join(model_dir, 'y_val.pkl'))
joblib.dump(y_pred, os.path.join(model_dir, 'y_pred.pkl'))