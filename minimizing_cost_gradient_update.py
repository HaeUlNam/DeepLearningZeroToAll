import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

# 아무리 큰 값을 넣더라도 Gradient Descent 알고리즘에 따라, Minimize가 잘된다.
# (아래 tf.Variable 값을 변경하며, 적용시켜 보자.)
# 다만, 적당한 learning rate를 설정해주어야 한다.

# W = tf.Variable(tf.random_normal([1]), name='weight')
W = tf.Variable(50000.0)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.2
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)
# Same As
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
# train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20):
    # W 값 변경 후에 cost function 조절
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
