import os
import numpy as np
import tensorflow as tf
from plotting import show, show_images
from tensorflow.examples.tutorials.mnist import input_data

mnist_path = '/mnt/c/src/tf-playground/mnist'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# setup model
X = tf.placeholder(tf.float32, [None, 784], 'X')
W = tf.Variable(tf.zeros([784, 10]), 'W')
b = tf.Variable(tf.zeros([10]), 'b')
Y = tf.nn.softmax(tf.matmul(X, W) + b)
Y_ = tf.placeholder(tf.float32, [None, 10], 'Y_')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer().run()

# training
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: batch_xs, Y_: batch_ys})

# check accuracy against test set
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))

# gets actual weights
We = W.eval()
# split by columns, then reshape columns to 28x28
Wec = [np.reshape(x, (28, 28)) for x in np.hsplit(We, 10)]
show_images(Wec, 2, ['Class: {}'.format(n) for n in range(10)])
