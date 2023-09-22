

import pandas as pd
import numpy as np

def orange_MONTHS(age):
  if age <= 8:
    return '1'
  elif age <= 15:
    return '2'
  elif age <= 25:
    return '3'
  elif age <= 40:
    return '4'
  else:
    return '5'

def speed_group(speed):
  if speed <= 60:
    return '1'
  else:
    return '2'  
  
def age_group(age):
  if age <= 23:
    return '1'
  elif age <= 25:
    return '2'
  elif age <= 30:
    return '3'
  elif age <= 35:
    return '4'
  elif age <= 40:
    return '5'
  elif age <= 45:
    return '6'
  elif age <= 50:
    return '7'
  elif age <= 55:
    return '8'
  else:
    return '9'

def preprocessing(data):
  
  # Remove unnecessary columns
  remove_columns = ['OF_PREV_SPEED', 'LAST_LINK_PRIORITY', 'LAST_POWER_VALIDATION', 'LAST_LINK_STATUS', 'LAST_LINK_QUALITY',
                    'Disconnection_TOTAL_MIN_day', 'ID', 'Disconnection_TOTAL_MAX_day', 'Disconnection_TOTAL_SUM_Month',
                    'Disconnection_TOTAL_MEAN_Month']
  data = data.drop(remove_columns, axis=1)


  # Remove rows with lots of missing data
  row_missing_percentage = .25

  data = data.loc[(data.isna().sum(1) / data.shape[1]) <= row_missing_percentage, :]
  data = data.reset_index(drop=True)

  assert not ((data.isna().sum(1) / data.shape[1]) > .25).any(), 'Error'


  # Fill missing data
  categorical_columns = data.select_dtypes('O').columns
  numerical_columns = data.select_dtypes(np.number).columns


  # Categorical data
  governorates_missing = 'West Amman'
  customer_gender = 'M'

  data['GOVERNORATE'].fillna(value=governorates_missing, inplace=True)
  data['CUSTOMER_GENDER'].fillna(value=customer_gender, inplace=True)

  assert not data[categorical_columns].isna().any().any(), 'Error'


  # Numerical data
  ### Median
  numerical_missing_values = {'GB_TOTAL_CONSUMPTION_Month1': 361.79541484406195,
                              'GB_TOTAL_CONSUMPTION_Month2': 354.923208067659,
                              'GB_TOTAL_CONSUMPTION_Month3': 333.67581848939847}

  for num_miss_value in numerical_missing_values.keys():
    data[num_miss_value].fillna(value=numerical_missing_values[num_miss_value], inplace=True)

  assert not data[num_miss_value].isna().any().any(), 'Error'
  

  # Feature engineering
  # data = data[data['GOVERNORATE'].isin(['West Amman', 'East Amman'])]

  data.loc[~data['GOVERNORATE'].isin(['West Amman', 'East Amman']), 'GOVERNORATE'] = 'OTHER'
  data.loc[data['CUSTOMER_GENDER'] == 'U', 'CUSTOMER_GENDER'] = 'M'

  data['CUSTOMER_AGE_YEARS'] = data['CUSTOMER_AGE_MONTHS'] // 12
  data = data[(data['CUSTOMER_AGE_YEARS'] >= 18) & (data['CUSTOMER_AGE_YEARS'] <= 60)]
  data['CUSTOMER_AGE_YEARS'] = data['CUSTOMER_AGE_YEARS'].apply(age_group)
  data['AGE_GENDER'] = data['CUSTOMER_AGE_YEARS'] + data['CUSTOMER_GENDER']
  data.drop([ 'CUSTOMER_AGE_YEARS', 'CUSTOMER_GENDER'], axis=1, inplace=True)

  data['OF_SPEED'] = data['OF_SPEED'].apply(speed_group)

  # data['Customer with orange_MONTHS'] = data['Customer with orange_MONTHS'].apply(orange_MONTHS)

  data['GB_TOTAL_CONSUMPTION'] = (data['GB_TOTAL_CONSUMPTION_Month1'] +
                                  data['GB_TOTAL_CONSUMPTION_Month2'] +
                                  data['GB_TOTAL_CONSUMPTION_Month3']) / 3
  data.drop(['GB_TOTAL_CONSUMPTION_Month1', 'GB_TOTAL_CONSUMPTION_Month2', 'GB_TOTAL_CONSUMPTION_Month3'], axis=1, inplace=True)

  # Encode categorical data
  categorical_columns = ['GOVERNORATE', 'MIGRATION_FLAG', 'AGE_GENDER', 'OF_SPEED']#, 'Customer with orange_MONTHS']
  # data = pd.get_dummies(data, columns=categorical_columns)

  for col in categorical_columns:
        data[col] = pd.factorize(data[col])[0]



  COMMITMENT_MAP = {36: 3, 24: 2, 12: 1}
  data['COMMITMENT'] = data['COMMITMENT'].map(COMMITMENT_MAP)

  # Scale the data
  ### RobustScaler = (Xi-Xmedian) / Xiqr

  data_median = {'GB_TOTAL_CONSUMPTION': 375}
  data_iqr = {'GB_TOTAL_CONSUMPTION': 347}


  for key in data_median.keys():
    data[key] = (data[key] - data_median[key]) / data_iqr[key]
  data.drop(['TARGET'], axis=1, inplace=True)
  return data