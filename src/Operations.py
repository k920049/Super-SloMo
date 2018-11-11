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

def l1_loss(predictions, targets):
    """
    Tensorflow version of l1 loss
    :param predictions: A predicted frame in between the two frames
    :param targets: Original frame
    :return: l1 loss
    """
    loss = tf.reduce_mean(tf.abs(predictions - targets))
    return loss

def l2_loss(predictions, targets):
    """
    Tensorflow version of l2 loss
    :param predictions: A predicted frame in between the two frames
    :param targets: Original frame
    :return: l2 loss
    """
    loss = tf.reduce_mean(tf.square(predictions - targets))
    return loss

def bilinear_interp(image, x, y):
    """
    Perform bilinear sampling on im given x, y coordinates

    This function implements the differentiable sampling mechanism with
    bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
    (5).

    x,y are tensors specfying normalized coorindates [-1,1] to sample from im.
    (-1,1) means (0,0) coordinate in im. (1,1) means the most bottom right pixel.

    :param image: Tensor of size [batch_size, height, width, depth]
    :param x: Tensor of size [batch_size, height, width, 1]
    :param y: Tensor of size [batch_size, height, width, 1]
    :return: Tensor of size [batch_size, height, width, depth]
    """
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    # constants
    num_batch = tf.shape(im)[0]
    _, height, width, channels = im.get_shape().as_list()

    x = tf.to_float(x)
    y = tf.to_float(y)

    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    zero = tf.constant(0, dtype=tf.int32)

    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0

    # Sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    dim2 = width
    dim1 = width * height

    # Create base index
    base = tf.range(num_batch) * dim1
    base = tf.reshape(base, [-1, 1])
    base = tf.tile(base, [1, height * width])
    base = tf.reshape(base, [-1])

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # Use indices to look up pixels
    im_flat = tf.reshape(im, [-1, channels])
    im_flat = tf.to_float(im_flat)
    pixel_a = tf.gather(im_flat, idx_a)
    pixel_b = tf.gather(im_flat, idx_b)
    pixel_c = tf.gather(im_flat, idx_c)
    pixel_d = tf.gather(im_flat, idx_d)

    # Interpolate the values
    x1_f = tf.to_float(x1)
    y1_f = tf.to_float(y1)

    wa = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
    wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
    wc = tf.expand_dims((1.0 - (x1_f - x)) * (y1_f - y), 1)
    wd = tf.expand_dims((1.0 - (x1_f - x)) * (1.0 - (y1_f - y)), 1)

    output = tf.add_n([wa * pixel_a, wb * pixel_b,
                       wc * pixel_c, wd * pixel_d])
    output = tf.reshape(output, shape=[num_batch, height, width, channels])

    return output

def meshgrid(height, width):
    """
    Tensorflow meshgrid function.
    :param height: The height of the frame
    :param width: The width of the frame
    :return: A matrix
    """
    x_t = tf.matmul(tf.ones(shape=[height, 1]), tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1), tf.ones(shape=[1, width]))
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    # grid_x = tf.reshape(x_t_flat, [1, height, width, 1])
    # grid_y = tf.reshape(y_t_flat, [1, height, width, 1])
    grid_x = tf.reshape(x_t_flat, [1, height, width])
    grid_y = tf.reshape(y_t_flat, [1, height, width])

    return grid_x, grid_y

