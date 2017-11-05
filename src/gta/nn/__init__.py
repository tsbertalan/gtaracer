import tensorflow as tf

class DNN(object):

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

    @property
    def saver(self):
        if not hasattr(self, '_saver'):
            self._saver = tf.train.Saver(self.toSave)
        return self._saver

    @property
    def toSave(self):
        return self.weights + self.biases

    def save(self, sess, fname=None, doPrint=True):
        if fname is None: fname = './%s.ckpt' % type(self).__name__
        if doPrint: print('Saving model to %s ...' % fname, end=' ')
        self.saver.save(sess, fname)
        if doPrint: print('done.')

    def load(self, sess, fname=None, doPrint=True):
        if fname is None: fname = './%s.ckpt' % type(self).__name__
        if doPrint: print('Saving model to %s ...' % fname, end=' ')
        self.saver.restore(sess, fname)
        if doPrint: print('done (run tf.global_variables_initializer() now).')


class ConvNet(DNN):

    def __init__(self, c=1):

        DNN.__init__(self)

        # Hyperparameters
        self.mu = 0
        self.sigma = 0.1
        self.c = c
        self.weights = []
        self.biases = []
        self.convs = []
        self.activations = []

        super
        
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
    
