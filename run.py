
import pandas as pd
import numpy as np
from helper import *

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle


np.seterr(divide = 'ignore') 


data = pd.read_csv('./titanic.csv')
train, test = train_test_split(data, test_size = 0.25, random_state = 42)

 
ordinal_pipline = make_pipeline(
  SimpleImputer(strategy='most_frequent'),
  OrdinalEncoder()
)

family_size_pipline = make_pipeline(
  SimpleImputer(strategy='mean'),
  FunctionTransformer(family_size),
)

name_pipline = make_pipeline(
  SimpleImputer(strategy='constant' , fill_value='unkown'),
  FunctionTransformer(custom_name_transformer),
  OneHotEncoder()
  )

log_pipline = make_pipeline(
  SimpleImputer(strategy='mean'),
  FunctionTransformer(np.log),
  FunctionTransformer(remove_inf),
  StandardScaler()
)
normal_pipline = make_pipeline(
  SimpleImputer(strategy='most_frequent'))

preprocessing  = ColumnTransformer([
  ('Age', log_pipline, ['Age']),
  ('Sex', ordinal_pipline,['Sex']),
  ('Family_size', family_size_pipline, ['Sibsp', 'Parch']),
  ('Embarked', ordinal_pipline, ['Embarked']),
  ('Title', name_pipline, ['Name']),
  ('Fare', log_pipline, ['Fare']),
  ('Pclass', normal_pipline, ['Pclass']),
])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


pipline = make_pipeline(preprocessing, XGBClassifier(
  max_depth=1,
  subsample=1,
  colsample_bytree=1,
  learning_rate=0.6,
  n_estimators=100,
  n_jobs=-1,
  random_state=42,
  objective='binary:logistic'
)) 

pipline.fit(train, le.fit_transform(train['Survived']))


print('Train Accuracy: ', pipline.score(train, le.fit_transform(train['Survived'])))
print('Test Accuracy: ', pipline.score(test, le.fit_transform(test['Survived'])))

pickle.dump(pipline, open('model.pkl', 'wb'))
