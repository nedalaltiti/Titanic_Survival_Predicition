# survivetitanic-be

used the titanic dataset to predict the survival of the titanic passengers

using xgboost with the following parameters

```python
  max_depth=1,
  subsample=1,
  colsample_bytree=1,
  learning_rate=0.6,
  n_estimators=100,
  n_jobs=-1,
  random_state=42,
  objective='binary:logistic'
```

got the following results

```python
    Train Accuracy:  0.8309572301425662
    Test Accuracy:  0.8201219512195121
```

## Installation

```bash
    RUN pip install pandas xgboost scikit-learn numpy
    python web.py
```

## Usage

post request : /

```json
{
  "Age": 48,
  "Sex": "male",
  "Sibsp": 1,
  "Parch": 2,
  "Embarked": "S",
  "Name": "Anderson, Miss. Harry",
  "Fare": 26.55,
  "Pclass": 1
}
```
