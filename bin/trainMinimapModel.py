import sys, os
addpath = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(addpath)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import gta.utils, os
import gta.nn

MODEL_NAME = 'otg-balanced'
LOAD_MODEL = True
PREV_MODEL = MODEL_NAME

# w = [1,0,0,0,0,0,0,0,0]
# s = [0,1,0,0,0,0,0,0,0]
# a = [0,0,1,0,0,0,0,0,0]
# d = [0,0,0,1,0,0,0,0,0]

# wa = [0,0,0,0,1,0,0,0,0]
# wd = [0,0,0,0,0,1,0,0,0]
# sa = [0,0,0,0,0,0,1,0,0]
# sd = [0,0,0,0,0,0,0,1,0]

# nk = [0,0,0,0,0,0,0,0,1]
n_classes = 4 + 4 + 1

class Arch(gta.nn.ConvNet):
        
    def __call__(self, x, name='predictions'):
        # def td(*args, **kwargs):
        #     x = self._addConv2d(*args, **kwargs)
        #     print(x.shape)
        #     return x
        td = self._addConv2d
        fc = self._addFc
        #self.keep_prob = tf.placeholder_with_default(.5, shape=())
        
        x = td(x, (8, 8, self.c, 12), padding='SAME', pooling=False)
        x = td(x, (8, 8, int(x.shape[-1]), 12), padding='SAME')
        #x = tf.nn.dropout(x, self.keep_prob)
        
        x = td(x, (3, 3, int(x.shape[-1]), 16), padding='SAME', pooling=False)
        x = td(x, (3, 3, int(x.shape[-1]), 16), padding='SAME')
        
        x = td(x, (3, 3, int(x.shape[-1]), 32), padding='SAME', pooling=False)
        x = td(x, (3, 3, int(x.shape[-1]), 32), padding='VALID')
        
        x = td(x, (3, 3, int(x.shape[-1]), 64), padding='SAME', pooling=False)
        x = td(x, (3, 3, int(x.shape[-1]), 64), padding='VALID')
        
        x = tf.contrib.layers.flatten(x)
        
        x = fc(x, (int(x.shape[-1]), 32))
        x = fc(x, (int(x.shape[-1]), n_classes), name=name)
        
        return x

image_shape = WIDTH, HEIGHT, NCHANNELS = 50, 65, 3
x = tf.placeholder(tf.float32, (None, *image_shape), name='images')
y_classlabels = tf.placeholder(tf.uint8, (None, 1), name='classLabels')

# Define tensors.
net = Arch(c=NCHANNELS)
logits = net(x)
y = tf.reshape(tf.one_hot(y_classlabels, n_classes), (-1, n_classes))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
softmax_prob = tf.nn.softmax(logits)
prediction = tf.argmax(softmax_prob, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Set up training.
loss = tf.reduce_mean(cross_entropy)
learning_rate = .0001
training_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Set up logging.
# http://ischlag.github.io/2016/06/04/how-to-use-tensorboard/
summaries_dir = './log/%s/' % time.time()
train_writer = tf.summary.FileWriter(summaries_dir + '/train', graph=tf.get_default_graph())
test_writer = tf.summary.FileWriter(summaries_dir + '/test', graph=tf.get_default_graph())
#writer = tf.summary.FileWriter('./log/', graph=tf.get_default_graph())
# create a summary for our cost and accuracy
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy_operation)
summary_op = tf.summary.merge_all()

EPOCHS = 4
FILE_I_END = 35
BATCH_SIZE = 16

def evaluate(X_data, y_data, sess=None, extraFeedDict={}):
    if hasattr(net, 'keep_prob'):
        extraFeedDict.setdefault(net.keep_prob, 1.0)
    num_examples = len(X_data)
    total_accuracy = 0
    if sess is None: sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        fd = {x: batch_x, y: batch_y}
        fd.update(extraFeedDict)
        accuracy = sess.run(accuracy_operation, feed_dict=fd)
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

saver = tf.train.Saver(net.weights + net.biases)

def save(sess, doprint=False, fname='./%s.ckpt' % MODEL_NAME):
    if doprint: print('Saving model to %s.ckpt ...' % fname, end=' ')
    saver.save(sess, fname)
    if doprint: print('done.')

with tf.Session() as sess:
    # Load or initialize.
    if LOAD_MODEL:
        try:
            saver.restore(sess, '%s.ckpt' % MODEL_NAME)
            print('We have loaded a previous model!!!!')
        except (ValueError, tf.errors.NotFoundError):
            print('No model to be found; starting %s.ckpt from scratch.' % MODEL_NAME)
    sess.run(tf.global_variables_initializer())

    # Iterate over epochs.
    for epoch in tqdm(range(EPOCHS), unit='epoch', desc='Overall'):
        # Iterate over training files.
        data_order = [i for i in range(1, FILE_I_END+1)]
        np.random.shuffle(data_order)
        for count, i in tqdm(
            enumerate(data_order), total=len(data_order), 
            unit='file', desc='Epoch %d' % (epoch+1,)
            ):
            try:
                file_name = 'training_data-{}.npy'.format(i)
                # full file info
                train_data = np.load(file_name)

                train = train_data[:-50]
                test = train_data[-50:]

                X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
                Y = [i[1] for i in train]

                test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
                test_y = [i[1] for i in test]
                for offset in range(0, len(X), BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X[offset:end], Y[offset:end]
                    unusedTrainResult, summary = sess.run(
                        [training_operation, summary_op],
                        feed_dict={x: batch_x, y: batch_y}
                    )
                    train_writer.add_summary(summary)

                # Write log
                summary = sess.run(summary_op, feed_dict={x: test_x, y: test_y})
                test_writer.add_summary(summary, epoch*len(data_order) + count)

            except Exception as e:
                print(str(e))

        if not epoch % 3:
            save(sess)
    
    save(sess, True)