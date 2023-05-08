import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('diabetes_prediction_dataset.csv')

# EDA
data.head()
data.dtypes
data.isnull().sum()
data['smoking_history'].value_counts()
data['smoking_history'].shape
idx = data[data['smoking_history'] == 0]
idx.shape
data['diabetes'].value_counts()
data['smoking_history'] =  data['smoking_history'].replace('No Info', 0)

# Fix gender values and smoking_history
data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)

def transform(data):
    result = 5
    if(data=='current'):
        result = 0
    elif(data=='not current'):
        result = 1
    elif(data=='former'):
        result = 2
    elif(data=='ever'):
        result = 3
    elif(data=='never'):
        result = 4
    return(result)

data['smoking_history'] = data['smoking_history'].apply(transform)

# Separate target from predictors
y = data.diabetes
X = data.drop(['diabetes'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# # "Cardinality" means the number of unique values in a column
# # Select categorical columns with relatively low cardinality (convenient but arbitrary)
# categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
#                         X_train_full[cname].dtype == "object"]

# # Select numerical columns
# numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# # Keep selected columns only
# my_cols = categorical_cols + numerical_cols
# X_train = X_train_full[my_cols].copy()
# X_valid = X_valid_full[my_cols].copy()

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder

# # Preprocessing for numerical data
# numerical_transformer = SimpleImputer(strategy='constant')

# # Preprocessing for categorical data
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Bundle preprocessing for numerical and categorical data
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=0)



# # Bundle preprocessing and modeling code in a pipeline
# my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', model)
#                              ])

# # Preprocessing of training data, fit model 
# my_pipeline.fit(X_train, y_train)

# # Preprocessing of validation data, get predictions
# preds = my_pipeline.predict(X_valid)

model.fit(X_train_full, y_train)

preds = model.predict(X_valid_full)


# Evaluate the model using various metrics
accuracy = accuracy_score(y_valid, preds)
precision = precision_score(y_valid, preds)
recall = recall_score(y_valid, preds)
f1 = f1_score(y_valid, preds)

# Print the evaluation metrics
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1))

import joblib
joblib.dump(model, 'diabetes_rfc_model.pkl') 



