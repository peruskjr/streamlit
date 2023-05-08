import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

framingham = pd.read_csv('framingham.csv')

framingham.head()

sns.countplot(x = 'education', data = framingham)

sns.catplot(x='TenYearCHD', y='cigsPerDay', kind='bar', data = framingham)

sns.boxplot(x='TenYearCHD', y='age',hue='currentSmoker',data=framingham)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.boxplot(x='TenYearCHD',y='age',hue='prevalentStroke',data=framingham)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.boxplot(x='TenYearCHD',y='age',hue='diabetes',data=framingham)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.boxplot(x='TenYearCHD',y='totChol',data=framingham)
plt.ylim(80)

sns.catplot(x='TenYearCHD',y='sysBP',kind='bar',data=framingham)

sns.catplot(x='TenYearCHD',y='diaBP',kind='bar',data=framingham)

sns.catplot(x='TenYearCHD',y='BMI',kind='bar',data=framingham)

sns.catplot(x='TenYearCHD',y='BPMeds',kind='bar',data=framingham)

framingham.isnull().any()
framingham = framingham.dropna()

framingham['TenYearCHD'].value_counts()

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

X = framingham.drop('TenYearCHD',axis=1)
y = framingham['TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X_train,y_train)
rf = RandomForestClassifier()
rf.fit(X_over,y_over)

preds = rf.predict(X_test)
print(accuracy_score(y_test,preds))

import joblib
joblib.dump(rf, 'fhs_rf_model.pkl') 

# from sklearn.model_selection import train_test_split

# X = framingham.drop('TenYearCHD',axis=1)
# y = framingham['TenYearCHD']

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35)

# from numpy import mean
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.tree import DecisionTreeClassifier
# from imblearn.pipeline import Pipeline
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler

# oversample = RandomOverSampler(sampling_strategy='minority')
# X_over, y_over = oversample.fit_resample(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X_over,y_over,test_size=0.35)

# steps = [('under', RandomUnderSampler()), ('model', DecisionTreeClassifier())]
# pipeline = Pipeline(steps=steps)

# pipeline.fit(X_train,y_train)

# pipepred = pipeline.predict(X_test)

# from sklearn.metrics import classification_report,accuracy_score
# print(classification_report(y_test,pipepred))

# accuracy_score(y_test,pipepred)

# import joblib
# joblib.dump(pipeline, 'fhs_dtree_model.pkl')
