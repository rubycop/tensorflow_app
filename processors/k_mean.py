import mysql.connector
import matplotlib.pyplot as plt

import tensorflow.contrib.learn as learn
from sklearn import datasets, metrics, preprocessing
import collections
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

config = {
  'user'              :'root',
  'password'          :'123456',
  'host'              :'localhost',
  'database'          :'crunchbase',
  'raise_on_warnings' : True,
}

DEFAULT_ROW_NUM = 100

def get_funding_total_usd_and_funding_rounds(conn, cursor):
  query = (
    "SELECT c.funding_rounds, "
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

  res = []

  for (
    funding_rounds,
    funding_total_usd
  ) in cursor:
    res.append([
      int(funding_rounds),
      int(funding_total_usd)/100
    ])

  return res

# training data
conn = mysql.connector.connect(**config)
cursor = conn.cursor()

data = get_funding_total_usd_and_funding_rounds(conn, cursor)

cursor.close()
conn.close()

clusters_n = 2
iteration_n = 1000

points = tf.constant(data)
centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0,0], [clusters_n, -1]))

points_expanded = tf.expand_dims(points, 0)
centroids_expanded = tf.expand_dims(centroids, 1)

distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
assignments = tf.argmin(distances, 0)

means = []
for c in xrange(clusters_n):
  means.append(tf.reduce_mean(
    tf.gather(points,
      tf.reshape(
        tf.where(
          tf.equal(assignments, c)
        ),[1,-1])
     ),reduction_indices=[1]))

new_centroids = tf.concat(means, 0)
update_centroids = tf.assign(centroids, new_centroids)

init = tf.initialize_all_variables()

with tf.Session() as sess:
  sess.run(init)

  for step in xrange(iteration_n):
    [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])

plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.6)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
plt.show()
