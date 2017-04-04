import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

# Model parameters
a = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

model = a / x + b

# loss
loss = tf.reduce_sum(tf.square(model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = []
y_train = []

with open('data3.csv', 'rb') as f:
  reader = csv.reader(f)
  for row in reader:
    x_train.append(float(row[1]))
    y_train.append(float(row[0]))

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(100):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([a, b, loss], {x:x_train, y:y_train})
print("a: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

plt.plot(x_train, curr_W / x_train + curr_b, '--', linewidth=1)
plt.plot(x_train, y_train, '-', linewidth=2)
plt.show()
