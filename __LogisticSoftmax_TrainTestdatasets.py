import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

# Placeholder
X = tf.placeholder("float", shape=[None, 3])
Y = tf.placeholder("float", shape=[None, 3])

# Variable
W = tf.Variable(tf.random_normal([3, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

# Cost cross entropy
# multi classfication을 하게 되면, 여러 logistic 값들이 합쳐지기에 1이 넘어가게 된다.
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Prediction & Accuracy (Using arg_max)
prediction = tf.arg_max(hypothesis, 1)

# Arg_max는 두번째 인자로 넘겨준 1 차원의 argument 중에서 가장 큰 값을 반환합니다.
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Session run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    #Predict
    print("Predication : ", sess.run(prediction, feed_dict={X: x_test}))
    print("Accuracy : ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

