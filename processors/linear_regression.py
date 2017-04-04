import mysql.connector
import matplotlib.pyplot as plt

import tensorflow.contrib.learn as learn
from sklearn import datasets, metrics, preprocessing
import collections
from tensorflow.python.platform import gfile
import numpy as np

config = {
  'user'              :'root',
  'password'          :'123456',
  'host'              :'localhost',
  'database'          :'crunchbase',
  'raise_on_warnings' : True,
}

DEFAULT_ROW_NUM = 10000

def get_funding_total_usd_and_funding_rounds(conn, cursor):
  query = (
    "SELECT c.funding_rounds, "
           "Avg(f.raised_amount_usd) as avg_funding_usd, "
           "c.funding_total_usd "
    "FROM cb_objects AS c "
      "Inner Join cb_funding_rounds AS f "
      "On c.id = f.object_id "
    "WHERE c.entity_type = 'Company' AND "
          "c.funding_total_usd IS NOT NULL "
    "GROUP BY c.id "
    "ORDER BY c.funding_total_usd "
    "LIMIT %s;" % DEFAULT_ROW_NUM
  )

  cursor = conn.cursor()
  cursor.execute(query)

  x = []
  y = []

  for (
    funding_rounds,
    avg_funding_usd,
    funding_total_usd
  ) in cursor:
    x.append([
      int(funding_rounds),
      float(avg_funding_usd)
    ])
    y.append(float(funding_total_usd))

  return [x,y]

# training data
conn = mysql.connector.connect(**config)
cursor = conn.cursor()

data_x, data_y = get_funding_total_usd_and_funding_rounds(conn, cursor)

cursor.close()
conn.close()

x_train = np.zeros((DEFAULT_ROW_NUM, 2), dtype=np.float)
y_train = np.zeros((DEFAULT_ROW_NUM,), dtype=np.float)
# x_train = np.zeros((9,), dtype=np.int)
# y_train = np.zeros((9,), dtype=np.int)

for i in range(0, DEFAULT_ROW_NUM-1):
  x_train[i] = np.asarray(data_x[i], dtype=np.float)
  y_train[i] = np.asarray(data_y[i], dtype=np.float)
  # sample data
  # x_train[i] = np.asarray(i+1, dtype=np.int)
  # y_train[i] = np.asarray(i, dtype=np.int)

Dataset = collections.namedtuple('Dataset', ['x', 'y'])
data = Dataset(x=x_train, y=y_train)

x = preprocessing.StandardScaler().fit_transform(data.x)
feature_columns = learn.infer_real_valued_columns_from_input(x)
regressor = learn.LinearRegressor(feature_columns=feature_columns)
regressor.fit(x, data.y, steps=1000, batch_size=32)
predicted_data = list(regressor.predict(x, as_iterable=True))

print ' \n--- EXAMPLE DATA ---\n'
print data.x
print ' \n--- EXPECTED DATA RESULTS ---\n'
print data.y
print ' \n--- PREDICTED DATA RESULTS ---\n'
print predicted_data

score = metrics.mean_squared_error(predicted_data, data.y)

print ' \n--- MSE: ---\n'
print score

# plt.plot(x_train, a * x_train + b, '--', linewidth=1)
# plt.plot(x_train, y_train, '-', linewidth=2)
# plt.show()
