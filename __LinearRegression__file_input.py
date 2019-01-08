import numpy as np
import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('data_01_test_score.csv', dtype=np.float32, delimiter=',')

#모든 row로 데이터를 받아온다.
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# x_data, y_data 확인
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data, len(y_data))

X = tf.placeholder(dtype=tf.float32, shape=[None, 3])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20010):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y:y_data})

    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction:\n",  hy_val)

#Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print("Other score will be ", sess.run(hypothesis, feed_dict={X:[[60, 70, 110], [90, 100, 80]]}))