from plotting import show, show_images
import os
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist_logs = '/mnt/c/src/tf-playground/logs/mist'
mnist_path = '/mnt/c/src/tf-playground/mnist'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # kills optimization warnings

if tf.gfile.Exists(mnist_logs):
	tf.gfile.DeleteRecursively(mnist_logs)
tf.gfile.MakeDirs(mnist_logs)

# read data
mnist = input_data.read_data_sets(mnist_path, one_hot=True)

batch_size = 100
max_steps = 1000

# setup model
with tf.name_scope('Input'):
	X = tf.placeholder(tf.float32, [None, 784], 'X')
with tf.name_scope('Variables'):
	W = tf.Variable(tf.zeros([784, 10]), 'W')
	b = tf.Variable(tf.zeros([10]), 'b')
with tf.name_scope('cross_entropy'):
	Y = tf.nn.softmax(tf.matmul(X, W) + b)
	Y_ = tf.placeholder(tf.float32, [None, 10], 'Y_')
	with tf.name_scope('total'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
tf.summary.scalar('cross_entropy', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)
mnist_test_set = {X: mnist.test.images, Y_: mnist.test.labels}

sess = tf.InteractiveSession()
init = tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(mnist_logs + '/train', sess.graph)
test_writer = tf.summary.FileWriter(mnist_logs + '/test')

# training
for i in range(max_steps):
	if i % 10 == 0:	# Record summaries and test-set accuracy
		summary, acc = sess.run([merged, accuracy], feed_dict=mnist_test_set)
		test_writer.add_summary(summary, i)
		print('Accuracy at step %s: %s' % (i, acc));
	else:  # Record train set summaries, and train
		batch_xs, batch_ys = mnist.train.next_batch(100)
		mnist_test_set = {X: batch_xs, Y_: batch_ys}
		if i % 100 == 99:  # Record execution stats
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			summary, _ = sess.run([merged, train_step], feed_dict=mnist_test_set, options=run_options)
			train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
			train_writer.add_summary(summary, i)
			print('Adding run metadata for {}'.format(i))
		else:  # Record a summary
			summary, _ = sess.run([merged, train_step], feed_dict=mnist_test_set)
			train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()

# check accuracy against test set
print(sess.run(accuracy, feed_dict=mnist_test_set))

# gets actual weights
We = W.eval()
# split by columns, then reshape columns to 28x28
Wec = [np.reshape(x, (28, 28)) for x in np.hsplit(We, 10)]
show_images(Wec, 2, ['Class: {}'.format(n) for n in range(10)])
