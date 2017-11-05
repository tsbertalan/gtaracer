import tensorflow as tf
import gta.nn

class SimpleConvNet(gta.nn.ConvNet):

    def __init__(self, n_classes=9, **kwargs):
        gta.nn.ConvNet.__init__(self, **kwargs)
        self.n_classes = n_classes

        
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
        x = fc(x, (int(x.shape[-1]), self.n_classes), name=name)
        
        return x