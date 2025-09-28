import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import os

train_df = pd.read_csv('data/train.csv')

X = train_df.drop(columns=['y'])
y = train_df['y']

X = pd.get_dummies(X, drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
# param_dist = {
#     'n_estimators':[20, 50, 100],
#     'max_depth':[None, 10, 20, 30],
#     'min_samples_split':[2, 5, 10],
#     'min_samples_leaf':[1, 2, 4],
#     'max_features':['sqrt', 'log2'],
#     'bootstrap':[True, False]
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

best_params = {
    'n_estimators':50,
    'max_depth':30,
    'min_samples_split':2,
    'min_samples_leaf':2,
    'max_features':'log2',
    'bootstrap':True,
    'random_state':0
}
model = RandomForestClassifier(**best_params, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Validation Accuracy: ", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

model_dir = "src/model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "random_forest.pkl")
joblib.dump(model, model_path)

columns_path = os.path.join(model_dir, "training_columns.pkl")
joblib.dump(X_train.columns.tolist(), columns_path)

joblib.dump(y_val, os.path.join(model_dir, 'y_val.pkl'))
joblib.dump(y_pred, os.path.join(model_dir, 'y_pred.pkl'))
