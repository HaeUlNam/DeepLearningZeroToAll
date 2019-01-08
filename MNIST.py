import tensorflow as tf
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MINIST_data/", one_hot=True)

# class number
nb_classes = 10

# Placeholder
# MNIST image
X = tf.placeholder(tf.float32, [None, 784])
# 0- 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

# Variable


# hypothesis

# cost(cross entropy)

