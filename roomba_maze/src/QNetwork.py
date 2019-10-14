import tensorflow as tf


class QNetwork(object):
    def __init__(self, size, Env):
        self.Input = tf.placeholder(shape=[None, 360], dtype=tf.float32)
        self.imageIn = tf.reshape(self.Input, shape=[-1, 20, 20, 2])
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=16, kernel_size=[2, 2],
                                                     stride=[2, 2], padding='VALID', biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=32, kernel_size=[2, 2],
                                                     stride=[2, 2], padding='VALID', biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=256, kernel_size=[5, 5],
                                                     stride=[1, 1], padding='VALID', biases_initializer=None)
        self.fullconnect1 = tf.reshape(self.conv3, shape=[-1, 256])
        self.W1 = tf.Variable(tf.random_normal([256, size]))
        self.b1 = tf.Variable(tf.constant(0.1, shape=[size]))
        self.layer1 = tf.matmul(self.fullconnect1, self.W1) + self.b1
        self.W2 = tf.Variable(tf.random_normal([size, size]))
        self.b2 = tf.Variable(tf.constant(0.1, shape=[size]))
        self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + self.b2)
        self.layerAC, self.layerVC = tf.split(self.layer2, 2, 1)
        self.AW = tf.Variable(tf.random_normal([size // 2, Env.actions]))
        self.VW = tf.Variable(tf.random_normal([size // 2, 1]))
        self.Advantage = tf.matmul(self.layerAC, self.AW)
        self.Value = tf.matmul(self.layerVC, self.VW)

        self.Qout = self.Value + tf.subtract(self.Advantage,
                                             tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, Env.actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)  # 0.0001
        self.updateModel = self.trainer.minimize(self.loss)
