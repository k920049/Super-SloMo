import tensorflow as tf
import numpy as np



class FlowNetwork:

    def __init__(self, frame, width, height, channel, scope):

        self.scope = scope
        self.frame = frame
        self.width = width
        self.height = height
        self.channel = channel

        self._build_network()

    def _build_network(self):
        # input placeholder
        self.I_0 = tf.placeholder(dtype=tf.float32, shape=(None, self.width, self.height, self.channel))
        self.I_1 = tf.placeholder(dtype=tf.float32, shape=(None, self.width, self.height, self.channel))
        self.c_1 = tf.placeholder(dtype=tf.float32, shape=(self.frame, None, self.width, self.height, self.channel))
        self.c_2 = tf.placeholder(dtype=tf.float32, shape=(self.frame, None, self.width, self.height, self.channel))
        self.c_3 = tf.placeholder(dtype=tf.float32, shape=(self.frame, None, self.width, self.height, self.channel))
        self.c_4 = tf.placeholder(dtype=tf.float32, shape=(self.frame, None, self.width, self.height, self.channel))
        # pass the image to the flow network
        self.flow = self.flow_network(self.I_0, self.I_1)
        self.f_01 = tf.slice(self.flow, begin=(0, 0, 0, 0), size=(-1, -1, -1, 3))
        self.f_10 = tf.slice(self.flow, begin=(0, 0, 0, 3), size=(-1, -1, -1, -1))
        # compute the flow in time t
        self.f_t0 = self.c_1 * self.f_01 + self.c_2 * self.f_10
        self.f_t1 = self.c_3 * self.f_01 + self.c_4 * self.f_10

        

        self.f_t0 = tf.transpose(self.f_t0, perm=(1, 0, 2, 3, 4))
        self.f_t1 = tf.transpose(self.f_t1, perm=(1, 0, 2, 3, 4))



    def flow_network(self, I_0, I_1):
        # scope for the flow computation network
        with tf.variable_scope(self.scope) as scope:
            concat = tf.concat([I_0, I_1], axis=3)
            initializer = tf.variance_scaling_initializer()
            # first layer
            layer1 = tf.layers.conv2d(inputs=concat, filters=32, kernel_size=(7, 7), padding='same',
                                      kernel_initializer=initializer)
            layer1 = tf.nn.leaky_relu(features=layer1, alpha=0.1)
            layer1 = tf.layers.conv2d(inputs=layer1, filters=32, kernel_size=(7, 7), padding='same',
                                      kernel_initializer=initializer)
            layer1 = tf.nn.leaky_relu(features=layer1, alpha=0.1)
            # second layer
            layer2 = tf.layers.average_pooling2d(inputs=layer1, pool_size=(2, 2), strides=(2, 2), padding='same')
            layer2 = tf.layers.conv2d(inputs=layer2, filters=64, kernel_size=(5, 5), padding='same',
                                      kernel_initializer=initializer)
            layer2 = tf.nn.leaky_relu(features=layer2, alpha=0.1)
            layer2 = tf.layers.conv2d(inputs=layer2, filters=64, kernel_size=(5, 5), padding='same',
                                      kernel_initializer=initializer)
            layer2 = tf.nn.leaky_relu(features=layer2, alpha=0.1)
            # third layer
            layer3 = tf.layers.average_pooling2d(inputs=layer2, pool_size=(2, 2), strides=(2, 2), padding='same')
            layer3 = tf.layers.conv2d(inputs=layer3, filters=128, kernel_size=(3, 3), padding='same',
                                      kernel_initializer=initializer)
            layer3 = tf.nn.leaky_relu(features=layer3, alpha=0.1)
            layer3 = tf.layers.conv2d(inputs=layer3, filters=128, kernel_size=(3, 3), padding='same',
                                      kernel_initializer=initializer)
            layer3 = tf.nn.leaky_relu(features=layer3, alpha=0.1)
            # fourth layer
            layer4 = tf.layers.average_pooling2d(inputs=layer3, pool_size=(2, 2), strides=(2, 2), padding='same')
            layer4 = tf.layers.conv2d(inputs=layer4, filters=256, kernel_size=(3, 3), padding='same',
                                      kernel_initializer=initializer)
            layer4 = tf.nn.leaky_relu(features=layer4, alpha=0.1)
            layer4 = tf.layers.conv2d(inputs=layer4, filters=256, kernel_size=(3, 3), padding='same',
                                      kernel_initializer=initializer)
            layer4 = tf.nn.leaky_relu(features=layer4, alpha=0.1)
            # fifth layer
            layer5 = tf.layers.average_pooling2d(inputs=layer4, pool_size=(2, 2), strides=(2, 2), padding='same')
            layer5 = tf.layers.conv2d(inputs=layer5, filters=512, kernel_size=(3, 3), padding='same',
                                      kernel_initializer=initializer)
            layer5 = tf.nn.leaky_relu(features=layer5, alpha=0.1)
            layer5 = tf.layers.conv2d(inputs=layer5, filters=512, kernel_size=(3, 3), padding='same',
                                      kernel_initializer=initializer)
            layer5 = tf.nn.leaky_relu(features=layer5, alpha=0.1)
            # layer in the middle
            middle = tf.layers.average_pooling2d(inputs=layer5, pool_size=(2, 2), strides=(2, 2), padding='same')
            middle = tf.layers.conv2d(inputs=middle, filters=512, kernel_size=(3, 3), padding='same',
                                      kernel_initializer=initializer)
            middle = tf.nn.leaky_relu(features=middle, alpha=0.1)
            middle = tf.layers.conv2d(inputs=middle, filters=512, kernel_size=(3, 3), padding='same',
                                      kernel_initializer=initializer)
            middle = tf.nn.leaky_relu(features=middle, alpha=0.1)
            # fifth bilinear upsampling layer
            up_layer5 = tf.image.resize_bilinear(images=middle, size=(80, 45))
            up_layer5 = tf.concat([up_layer5, layer5], axis=3)
            up_layer5 = tf.layers.conv2d(inputs=up_layer5, filters=512, kernel_size=(3, 3), padding='same', kernel_initializer=initializer)
            up_layer5 = tf.nn.leaky_relu(features=up_layer5, alpha=0.1)
            up_layer5 = tf.layers.conv2d(inputs=up_layer5, filters=512, kernel_size=(3, 3), padding='same', kernel_initializer=initializer)
            up_layer5 = tf.nn.leaky_relu(features=up_layer5, alpha=0.1)
            # fourth bilinear upsampling layer
            up_layer4 = tf.image.resize_bilinear(images=up_layer5, size=(160, 90))
            up_layer4 = tf.concat([up_layer4, layer4], axis=3)
            up_layer4 = tf.layers.conv2d(inputs=up_layer4, filters=256, kernel_size=(3, 3), padding='same',
                                         kernel_initializer=initializer)
            up_layer4 = tf.nn.leaky_relu(features=up_layer4, alpha=0.1)
            up_layer4 = tf.layers.conv2d(inputs=up_layer4, filters=256, kernel_size=(3, 3), padding='same',
                                         kernel_initializer=initializer)
            up_layer4 = tf.nn.leaky_relu(features=up_layer4, alpha=0.1)
            # third bilinear upsampling layer
            up_layer3 = tf.image.resize_bilinear(images=up_layer4, size=(320, 180))
            up_layer3 = tf.concat([up_layer3, layer3], axis=3)
            up_layer3 = tf.layers.conv2d(inputs=up_layer3, filters=128, kernel_size=(3, 3), padding='same',
                                         kernel_initializer=initializer)
            up_layer3 = tf.nn.leaky_relu(features=up_layer3, alpha=0.1)
            up_layer3 = tf.layers.conv2d(inputs=up_layer3, filters=128, kernel_size=(3, 3), padding='same',
                                         kernel_initializer=initializer)
            up_layer3 = tf.nn.leaky_relu(features=up_layer3, alpha=0.1)
            # second bilinear upsampling layer
            up_layer2 = tf.image.resize_bilinear(images=up_layer3, size=(640, 360))
            up_layer2 = tf.concat([up_layer2, layer2], axis=3)
            up_layer2 = tf.layers.conv2d(inputs=up_layer2, filters=64, kernel_size=(3, 3), padding='same',
                                         kernel_initializer=initializer)
            up_layer2 = tf.nn.leaky_relu(features=up_layer2, alpha=0.1)
            up_layer2 = tf.layers.conv2d(inputs=up_layer2, filters=64, kernel_size=(3, 3), padding='same',
                                         kernel_initializer=initializer)
            up_layer2 = tf.nn.leaky_relu(features=up_layer2, alpha=0.1)
            # first bilinear upsampling layer
            up_layer1 = tf.image.resize_bilinear(images=up_layer2, size=(1280, 720))
            up_layer1 = tf.concat([up_layer1, layer1], axis=3)
            up_layer1 = tf.layers.conv2d(inputs=up_layer1, filters=32, kernel_size=(3, 3), padding='same',
                                         kernel_initializer=initializer)
            up_layer1 = tf.nn.leaky_relu(features=up_layer1, alpha=0.1)
            up_layer1 = tf.layers.conv2d(inputs=up_layer1, filters=32, kernel_size=(3, 3), padding='same',
                                         kernel_initializer=initializer)
            up_layer1 = tf.nn.leaky_relu(features=up_layer1, alpha=0.1)
            # final output layer
            output = tf.layers.conv2d(inputs=up_layer1, filters=6, kernel_size=(3, 3), padding='same',
                                           kernel_initializer=initializer)
            return output





