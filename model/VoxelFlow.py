"""
    A Tensorflow implementation of Super SloMo

    Copyright (c) 2018, Windforces
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the <ORGANIZATION> nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import tensorflow as tf

from src.Operations import l1_loss, l2_loss, bilinear_interp

class VoxelFlow:

    def __init__(self, width, height, channel, multiplier, scope):
        """
        VoxelFlow network

        In order to use this for both training and testing, you need to load the following placeholders
        self.input_placeholder: Two frames in the movie, in shape [batch_size, height, width, 2 * channel]
        self.target_placeholder: Target frame, in shape [batch_size, height, width, channel]
        self.is_train: Is the network trained right now?(For batch normalization)

        :param width: The width of the frame
        :param height: The height of the frame
        :param channel: The channel of the frame
        :param multiplier : Lagrange multiplier for the loss
        :param scope: A string specifying the variable scope
        """
        self.scope = scope
        self.width = width
        self.height = height
        self.channel = channel
        self.multiplier = multiplier

        self._build_model()

    def _build_model(self):
        """
        Build an encoder-decoder model using convolution
        :return: A tensor containing the result
        """
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.height, self.width, 2 * self.channel))
        self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.height, self.width, self.channel))
        self.is_train = tf.placeholder(dtype=tf.bool)

        initializer = tf.variance_scaling_initializer()

        with tf.variable_scope(name_or_scope=self.scope):
            # first layer
            net = tf.layers.conv2d(inputs=self.input_placeholder, filters=64, kernel_size=(5, 5), padding='same',
                                   kernel_initializer=initializer)
            net = tf.layers.batch_normalization(inputs=net, momentum=0.9997, epsilon=1e-3, training=self.is_train)
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), padding='same')
            # second layer
            net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(5, 5), padding='same',
                                   kernel_initializer=initializer)
            net = tf.layers.batch_normalization(inputs=net, momentum=0.9997, epsilon=1e-3, training=self.is_train)
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), padding='same')
            # third layer
            net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding='same',
                                   kernel_initializer=initializer)
            net = tf.layers.batch_normalization(inputs=net, momentum=0.9997, epsilon=1e-3, training=self.is_train)
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), padding='same')
            # third upsampling layer
            net = tf.image.resize_bilinear(images=net, size=(180, 320))
            net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding='same',
                                   kernel_initializer=initializer)
            net = tf.layers.batch_normalization(inputs=net, momentum=0.9997, epsilon=1e-3, training=self.is_train)
            # second upsampling layer
            net = tf.image.resize_bilinear(images=net, size=(360, 640))
            net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding='same',
                                   kernel_initializer=initializer)
            net = tf.layers.batch_normalization(inputs=net, momentum=0.9997, epsilon=1e-3, training=self.is_train)
            # first upsampling layer
            net = tf.image.resize_bilinear(images=net, size=(720, 1280))
            net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(5, 5), padding='same',
                                   kernel_initializer=initializer)
            net = tf.layers.batch_normalization(inputs=net, momentum=0.9997, epsilon=1e-3, training=self.is_train)
            # final layer
            net = tf.layers.conv2d(inputs=net, filters=3, kernel_size=(5, 5), padding='same',
                                   kernel_initializer=initializer)
            net = tf.nn.tanh(x=net)

            self.flow = net[:, :, :, 0:2]
            mask = tf.expand_dims(net[:, :, :, 2], 3)

            grid_x, grid_y = meshgrid(256, 256)
            grid_x = tf.tile(grid_x, [32, 1, 1])  # batch_size = 32
            grid_y = tf.tile(grid_y, [32, 1, 1])  # batch_size = 32

            flow = 0.5 * flow

            coor_x_1 = grid_x + flow[:, :, :, 0]
            coor_y_1 = grid_y + flow[:, :, :, 1]

            coor_x_2 = grid_x - flow[:, :, :, 0]
            coor_y_2 = grid_y - flow[:, :, :, 1]

            output_1 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1, coor_y_1)
            output_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2, coor_y_2)

            mask = 0.5 * (1.0 + mask)
            mask = tf.tile(mask, [1, 1, 1, 3])
            self.net = tf.mul(mask, output_1) + tf.mul(1.0 - mask, output_2)

    def inference(self):
        return self.net

    def loss(self):
        """
        The objective function for the network to train.
        :return: The total amount of loss
        """
        loss = l1_loss(self.net, self.target_placeholder)
        loss = loss + self.multiplier * tf.reduce_mean(tf.abs(self.flow))
        return loss






