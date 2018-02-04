import pandas
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data prep.
sample_data = pandas.read_csv('sample_data.csv')
sample_data['ARR_DELAY'] = sample_data['ARR_DELAY'].apply(
  lambda x: 1 if x >= 5 else 0)

sample_data = pandas.concat([
  sample_data,
  pandas.get_dummies(sample_data['UNIQUE_CARRIER'],
                     drop_first=True,
                     prefix="UNIQUE_CARRIER"),
  pandas.get_dummies(sample_data['ORIGIN'],
                     drop_first=True,
                     prefix="ORIGIN"),
  pandas.get_dummies(sample_data['DEST'],
                     drop_first=True,
                     prefix="DEST"),
  pandas.get_dummies(sample_data['DAY_OF_WEEK'],
                     drop_first=True,
                     prefix="DAY_OF_WEEK"),
  pandas.get_dummies(sample_data['DEP_HOUR'],
                     drop_first=True,
                     prefix="DEP_HOUR")
], axis=1)

sample_data.drop([
  'ORIGIN',
  'DEST',
  'UNIQUE_CARRIER',
  'DAY_OF_WEEK',
  'DEP_HOUR'
], axis=1, inplace=True)


# Model training.
X_train, X_test, y_train, y_test = train_test_split(
  sample_data.drop('ARR_DELAY', axis=1),
  sample_data['ARR_DELAY'],
  test_size=0.30)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

accuracy = accuracy_score(y_test,
                          predictions,
                          normalize=True,
                          sample_weight=None)
print('accuracy: %.2f' % accuracy)

with open('logmodel.pkl', 'wb') as file:
    pickle.dump(logmodel, file, 2)

categories = sample_data.drop('ARR_DELAY', axis=1)
category_index = dict(zip(categories.columns, range(categories.shape[1])))

with open('categories.pkl', 'wb') as file:
    pickle.dump(category_index, file, 2)

