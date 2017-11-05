import sys
sys.path.append('../src')

from importlib import reload
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import tqdm

import gta.utils, os
import gta.nn

fpath = os.path.join(gta.utils.home, 'data', 'gta', 'UnifiedRecorder-1509210169.3261402.npz')

keepEids = [0, 2, 5]

with gta.utils.timeit('Loading data'):
    saved = np.load(fpath)
    DT = np.diff(saved['T'])
    X = saved['X'][1:]
    # Only keep the first 6 features (gamepad; leave out buttons)
    Y = saved['Y'][1:][:, keepEids]

def normalize(mat):
    return (mat.astype('float32') - 127.5) / 255

splits = int(len(DT) * .8), int(len(DT) * .9)

def s(a, b):
    return X[a:b], Y[a:b], DT[a:b]

X_train, Y_train, DT_train = s(0, splits[0])
X_test, Y_test, DT_test = s(splits[0], splits[1])
X_valid, Y_valid, DT_valid = s(splits[1], -1)

n_train = len(DT_train)
n_test = len(DT_test)
n_valid = len(DT_valid)

image_shape = X_train.shape[1:]

x = tf.placeholder(tf.float32, (None, *image_shape), name='images')
y = tf.placeholder(tf.float32, (None, 3), name='gamepad_axes')

class Arch(gta.nn.ConvNet):
    
    def __call__(self, x, name='predictions'):
        td = self._addConv2d
        fc = self._addFc
        #self.keep_prob = tf.placeholder_with_default(.5, shape=())
        
        x = td(x, (8, 8, self.c, 12), padding='SAME', pooling=False)
        x = td(x, (8, 8, int(x.shape[-1]), 12), padding='VALID')
        #x = tf.nn.dropout(x, self.keep_prob)
        
        x = td(x, (3, 3, int(x.shape[-1]), 16), padding='SAME')#, pooling=False)
        x = td(x, (3, 3, int(x.shape[-1]), 16), padding='VALID')
        
        x = td(x, (3, 3, int(x.shape[-1]), 32), padding='SAME')#, pooling=False)
        x = td(x, (3, 3, int(x.shape[-1]), 32), padding='VALID')
        
        x = td(x, (3, 3, int(x.shape[-1]), 64), padding='SAME')#, pooling=False)
        x = td(x, (3, 3, int(x.shape[-1]), 64), padding='VALID')
        
        x = tf.contrib.layers.flatten(x)
        
        x = fc(x, (int(x.shape[-1]), 32))
        x = fc(x, (int(x.shape[-1]), Y.shape[1]), name=name)
        
        return x

net = Arch(c=image_shape[-1])
z = net(x)

loss = tf.losses.mean_squared_error(y, z)
learning_rate = .001
training_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

EPOCHS = 150
BATCH_SIZE = 128

def evaluate(X_data, y_data, sess=None, extraFeedDict={}):
    if hasattr(net, 'keep_prob'):
        extraFeedDict.setdefault(net.keep_prob, 1.0)
    num_examples = len(X_data)
    total_accuracy = 0
    if sess is None: sess = tf.get_default_session()
        
    num = 0
    den = 0
    
    for offset in tqdm.tqdm(range(0, num_examples, BATCH_SIZE)):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        fd = {x: normalize(batch_x), y: batch_y}
        fd.update(extraFeedDict)
        run = lambda inp: sess.run(inp, feed_dict=fd)
        
        num += run(tf.reduce_sum((y - z) ** 2))
        den += len(batch_x)
        
    return num / den

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(evaluate(X_valid, Y_valid))
