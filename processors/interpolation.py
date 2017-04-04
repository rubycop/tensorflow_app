import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Model parameters
a = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = a * x + b

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4,5]
y_train = [0,1,2,3,4]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
a, b, loss = sess.run([a, b, loss], {x:x_train, y:y_train})
print("a: %s b: %s loss: %s"%(a, b, loss))

plt.plot(x_train, y_train,         '-',  linewidth = 2)
plt.plot(x_train, a * x_train + b, '--', linewidth = 1)

plt.show()
