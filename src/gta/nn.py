import tensorflow as tf

class ConvNet(object):

    def __init__(self, c=1):

        # Hyperparameters
        self.mu = 0
        self.sigma = 0.1
        self.c = c
        self.weights = []
        self.biases = []
        self.convs = []
        self.activations = []
        
    def _addConv2d(
        self, 
        x, wshape, strides=(1, 1, 1, 1), 
        padding='VALID', mu=None, sigma=None, activator=tf.nn.relu,
        pooling=tf.nn.max_pool, ksize=(1, 2, 2, 1), poolingStrides=(1, 2, 2, 1), poolingPadding='VALID',
        **kw
        ):
        if mu is None: mu = self.mu
        if sigma is None: sigma = self.sigma
        W = tf.Variable(tf.truncated_normal(shape=wshape, mean=mu, stddev=sigma))
        b = tf.Variable(tf.zeros(wshape[-1]))
        conv = tf.nn.conv2d(x, W, strides=strides, padding=padding) + b
        activation = activator(conv, **kw)
        
        self.weights.append(W)
        self.biases.append(b)
        self.convs.append(conv)
        self.activations.append(activation)
        
        if pooling:
            activation = pooling(activation, ksize=ksize, strides=poolingStrides, padding=poolingPadding)
            self.activations.append(activation)
        
        return activation
    
    def _addFc(self, x, wshape, mu=None, sigma=None, activator=tf.nn.relu, **kw):
        if mu is None: mu = self.mu
        if sigma is None: sigma = self.sigma
        W = tf.Variable(tf.truncated_normal(shape=wshape, mean=mu, stddev=sigma))
        b = tf.Variable(tf.zeros(wshape[-1]))
        conv = tf.matmul(x, W) + b
        activation = activator(conv, **kw)
        
        self.weights.append(W)
        self.biases.append(b)
        self.convs.append(conv)
        self.activations.append(activation)
        
        return activation