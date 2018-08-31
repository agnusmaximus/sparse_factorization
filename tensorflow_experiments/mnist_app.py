import tensorflow as tf
from app import TfApp
import sys
import tempfile
from tensorflow.examples.tutorials.mnist import input_data
import numpy 

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class MnistApp(TfApp):

    def __init__(self):
      print("Creating mnist app")
      self.create_mnist_graph()

    def create_mnist_graph(self):

      self.sess = tf.Session()

      #########################################
      # The following copied from tf tutorial #
      #########################################      
      data_dir = "./mnist_data_dir"
      mnist = input_data.read_data_sets(data_dir)
      
      # Create the model
      x = tf.placeholder(tf.float32, [None, 784])
      
      # Define loss and optimizer
      y_ = tf.placeholder(tf.int64, [None])
      
      # Build the graph for the deep net
      y_conv, keep_prob = deepnn(x)

      l1_parameter_placeholder = tf.placeholder(dtype=tf.float32)
      l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=0.01, scope=None
      )
      regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, tf.trainable_variables())
      
      with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy) + regularization_penalty
        
        with tf.name_scope('adam_optimizer'):
          train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
          
        with tf.name_scope('accuracy'):
          correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
          correct_prediction = tf.cast(correct_prediction, tf.float32)
          accuracy = tf.reduce_mean(correct_prediction)
          
          graph_location = tempfile.mkdtemp()
          print('Saving graph to: %s' % graph_location)
          train_writer = tf.summary.FileWriter(graph_location)
          train_writer.add_graph(tf.get_default_graph())

      #######################################################

      self.mnist = mnist
      self.y_conv = y_conv
      self.keep_prob = keep_prob
      self.y_ = y_
      self.train_step = train_step
      self.accuracy = accuracy
      self.x = x

    def train(self, niters=500):
      print("Training mnist app")

      with self.sess.graph.as_default():
        self.sess.run(tf.global_variables_initializer())
        for i in range(niters):
          batch = self.mnist.train.next_batch(50)
          if i % 100 == 0:
            train_accuracy = self.accuracy.eval(session=self.sess, feed_dict={
              self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
          self.train_step.run(session=self.sess,
                              feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

      test_accuracy = self.eval()

    def eval(self):
      print("Eval mnist app")

      # compute in batches to avoid OOM on GPUs
      with self.sess.graph.as_default():      
        accuracy_l = []
        for _ in range(20):
          batch = self.mnist.test.next_batch(500, shuffle=False)
          accuracy_l.append(self.accuracy.eval(session=self.sess,
                                               feed_dict={self.x: batch[0], 
                                                          self.y_: batch[1], 
                                                          self.keep_prob: 1.0}))
        print('test accuracy %g' % numpy.mean(accuracy_l))
        return numpy.mean(accuracy_l)

    def get_layers_to_factorize(self):
      layers = []
      
      with self.sess.graph.as_default():
        #candidates = (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fc1') + 
        #              tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fc2'))        
        candidates = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fc2')
        for i in candidates:
          if len(i.get_shape().as_list()) > 1 and "Adam" not in i.name:
            layers.append(i)
      return layers

    def get_sess(self):
      return self.sess
