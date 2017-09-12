from plotting import show
import os
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist_path = '/mnt/c/src/tf-playground/mnist'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# read data
mnist = input_data.read_data_sets(mnist_path, one_hot=True)

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

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
# show weights
We = W.eval()
Wec = [np.reshape(x, (28, 28)) for x in np.hsplit(We, 10)]
show(np.block([
    [Wec[0], Wec[1], Wec[2], Wec[3], Wec[4]], 
    [Wec[5], Wec[6], Wec[7], Wec[8], Wec[9]] 
]))
