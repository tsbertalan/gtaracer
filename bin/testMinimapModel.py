import sys, os
addpath = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(addpath)
addpath = os.path.join(os.path.dirname(__file__), '..', '..', 'pygta5')
sys.path.append(addpath)

from collectData import getMinimap

import testModelPyGTA

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import gta.utils, os
from gta.nn.models import SimpleConvNet as Arch

MODEL_NAME = 'otg-balanced'

# w = [1,0,0,0,0,0,0,0,0]
# s = [0,1,0,0,0,0,0,0,0]
# a = [0,0,1,0,0,0,0,0,0]
# d = [0,0,0,1,0,0,0,0,0]

# wa = [0,0,0,0,1,0,0,0,0]
# wd = [0,0,0,0,0,1,0,0,0]
# sa = [0,0,0,0,0,0,1,0,0]
# sd = [0,0,0,0,0,0,0,1,0]

#nk = [0,0,0,0,0,0,0,0,1]
n_classes = 4 + 4 + 1

image_shape = WIDTH, HEIGHT, NCHANNELS = 50, 65, 3
x = tf.placeholder(tf.float32, (None, *image_shape), name='images')
y_classlabels = tf.placeholder(tf.uint8, (None, 1), name='classLabels')

# Define tensors.
net = Arch(c=NCHANNELS, n_classes=n_classes)
logits = net(x, name='logits')
y = tf.reshape(tf.one_hot(y_classlabels, n_classes), (-1, n_classes))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
softmax_prob = tf.nn.softmax(logits)
                                      #w s a d wa wd sa sd  nk
def predict(imgs, sess=None, reweight=[1,1,1,1, 1, 1, .5, 1, .008]):
    if sess is None:
        sess = tf.get_default_session()
    out = np.stack([
        sess.run(softmax_prob, feed_dict={
            x: getMinimap(img).reshape((1, WIDTH, HEIGHT, NCHANNELS)),
            y: np.zeros((1, n_classes)),
        })
        for img in imgs
    ])
    if reweight:
        out *= reweight
    return out

saver = tf.train.Saver(net.weights + net.biases)

with tf.Session() as sess:
    saver.restore(sess, '%s.ckpt' % MODEL_NAME)
    sess.run(tf.global_variables_initializer())
    print('We have loaded a previous model!!!!')
    if __name__ == '__main__':
        testModelPyGTA.main(predict)
